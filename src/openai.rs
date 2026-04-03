use std::{
    env,
    future::Future,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
};

use anyhow::{anyhow, Context, Result};
use futures_util::StreamExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};

#[derive(Clone)]
pub struct OpenAIClient {
    client: Client,
    api_key: String,
    base_url: String,
    model: String,
    temperature: f32,
    default_system_prompt: String,
    max_output_tokens: Option<u32>,
}

impl OpenAIClient {
    pub fn from_env() -> Result<Self> {
        let api_key = env::var("OPENAI_API_KEY")
            .map_err(|_| anyhow!("OPENAI_API_KEY must be set to call the OpenAI API"))?;
        let model = env::var("OPENAI_MODEL").unwrap_or_else(|_| "gpt-4o-mini".to_string());
        let base_url = env::var("OPENAI_BASE_URL")
            .unwrap_or_else(|_| "https://api.openai.com/v1".to_string());
        let temperature = env::var("OPENAI_TEMPERATURE")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .unwrap_or(0.7);
        let default_system_prompt = env::var("OPENAI_SYSTEM_PROMPT")
            .unwrap_or_else(|_| "You are a helpful AI assistant.".to_string());
        let max_output_tokens = env::var("OPENAI_MAX_OUTPUT_TOKENS")
            .ok()
            .and_then(|v| v.parse::<u32>().ok());

        Ok(Self {
            client: Client::new(),
            api_key,
            base_url: base_url.trim_end_matches('/').to_string(),
            model,
            temperature,
            default_system_prompt,
            max_output_tokens,
        })
    }

    pub fn default_system_prompt(&self) -> &str {
        &self.default_system_prompt
    }

    pub async fn chat_completion(&self, messages: Vec<ChatMessage>) -> Result<String> {
        let request = ChatCompletionRequest {
            model: &self.model,
            messages: &messages,
            temperature: self.temperature,
            max_output_tokens: self.max_output_tokens,
            stream: false,
        };

        let url = format!("{}/chat/completions", self.base_url);
        let response = self
            .client
            .post(url)
            .bearer_auth(&self.api_key)
            .json(&request)
            .send()
            .await
            .context("failed to call OpenAI chat completions")?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(anyhow!("openai_error (status {status}): {body}"));
        }

        let parsed: ChatCompletionResponse = response
            .json()
            .await
            .context("invalid OpenAI response body")?;

        let text = parsed.into_text();
        Ok(text)
    }

    pub async fn stream_chat_completion<F, Fut>(
        &self,
        messages: Vec<ChatMessage>,
        cancel: Arc<AtomicBool>,
        mut on_token: F,
    ) -> Result<String>
    where
        F: FnMut(String) -> Fut,
        Fut: Future<Output = Result<()>>,
    {
        let request = ChatCompletionRequest {
            model: &self.model,
            messages: &messages,
            temperature: self.temperature,
            max_output_tokens: self.max_output_tokens,
            stream: true,
        };

        let url = format!("{}/chat/completions", self.base_url);
        let response = self
            .client
            .post(url)
            .bearer_auth(&self.api_key)
            .json(&request)
            .send()
            .await
            .context("failed to call OpenAI chat completions")?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(anyhow!("openai_error (status {status}): {body}"));
        }

        let mut stream = response.bytes_stream();
        let mut buffer = String::new();
        let mut accumulated = String::new();

        while let Some(chunk) = stream.next().await {
            if cancel.load(Ordering::SeqCst) {
                break;
            }
            let chunk = chunk.context("OpenAI stream closed unexpectedly")?;
            let chunk_str = String::from_utf8_lossy(&chunk);
            buffer.push_str(&chunk_str.replace('\r', ""));

            while let Some(idx) = buffer.find("\n\n") {
                let event = buffer[..idx].to_string();
                buffer.drain(..idx + 2);

                for line in event.lines() {
                    if let Some(data) = line.strip_prefix("data: ") {
                        let payload = data.trim();
                        if payload.is_empty() {
                            continue;
                        }
                        if payload == "[DONE]" {
                            return Ok(accumulated);
                        }

                        let chunk: ChatCompletionStreamResponse = serde_json::from_str(payload)
                            .map_err(|err| anyhow!("failed to parse OpenAI stream chunk: {err}: {payload}"))?;
                        let delta = chunk.into_text();
                        if delta.is_empty() {
                            continue;
                        }
                        accumulated.push_str(&delta);
                        on_token(delta).await?;
                    }
                }

                if cancel.load(Ordering::SeqCst) {
                    break;
                }
            }
        }

        Ok(accumulated)
    }
}

#[derive(Clone, Serialize)]
pub struct ChatMessage {
    role: String,
    content: String,
}

impl ChatMessage {
    pub fn system(content: String) -> Self {
        Self {
            role: "system".into(),
            content,
        }
    }

    pub fn user(content: String) -> Self {
        Self {
            role: "user".into(),
            content,
        }
    }

    pub fn assistant(content: String) -> Self {
        Self {
            role: "assistant".into(),
            content,
        }
    }

    pub fn from_role(role: String, content: String) -> Self {
        Self { role, content }
    }
}

#[derive(Serialize)]
struct ChatCompletionRequest<'a> {
    model: &'a str,
    messages: &'a [ChatMessage],
    temperature: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_output_tokens: Option<u32>,
    stream: bool,
}

#[derive(Deserialize)]
struct ChatCompletionResponse {
    choices: Vec<ChatCompletionChoice>,
}

impl ChatCompletionResponse {
    fn into_text(self) -> String {
        let mut buf = String::new();
        for choice in self.choices {
            if let Some(content) = choice.message.content {
                buf.push_str(&content.into_text());
            }
        }
        buf.trim().to_string()
    }
}

#[derive(Deserialize)]
struct ChatCompletionChoice {
    message: ChoiceMessage,
}

#[derive(Deserialize)]
struct ChoiceMessage {
    #[serde(default)]
    content: Option<MessageContent>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum MessageContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

impl MessageContent {
    fn into_text(self) -> String {
        match self {
            MessageContent::Text(text) => text,
            MessageContent::Parts(parts) => parts
                .into_iter()
                .filter_map(|part| part.text)
                .collect(),
        }
    }
}

#[derive(Deserialize)]
struct ChatCompletionStreamResponse {
    choices: Vec<StreamChoice>,
}

impl ChatCompletionStreamResponse {
    fn into_text(self) -> String {
        let mut buf = String::new();
        for choice in self.choices {
            if let Some(content) = choice.delta.content {
                buf.push_str(&content.into_text());
            }
        }
        buf
    }
}

#[derive(Deserialize)]
struct StreamChoice {
    delta: StreamDelta,
}

#[derive(Deserialize)]
struct StreamDelta {
    #[serde(default)]
    content: Option<DeltaContent>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum DeltaContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

impl DeltaContent {
    fn into_text(self) -> String {
        match self {
            DeltaContent::Text(text) => text,
            DeltaContent::Parts(parts) => parts
                .into_iter()
                .filter_map(|part| part.text)
                .collect(),
        }
    }
}

#[derive(Deserialize)]
struct ContentPart {
    #[serde(rename = "type")]
    _kind: String,
    #[serde(default)]
    text: Option<String>,
}
