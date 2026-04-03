use crate::{
    attachments::message_attachment_summaries,
    model::message::Message,
    openai::ChatMessage,
};

/// Keep only the last `max_messages` entries from the chat history.
pub fn trim_history(mut history: Vec<Message>, max_messages: usize) -> Vec<Message> {
    if history.len() <= max_messages {
        return history;
    }
    history.drain(0..history.len().saturating_sub(max_messages));
    history
}

/// Convert stored [`Message`] values into OpenAI chat messages, optionally
/// prepending a system prompt. Attachment context, when available, is appended
/// to the message body so the model can see a short description of the files.
pub fn to_openai_messages(
    system_prompt: Option<&str>,
    history: &[Message],
) -> Vec<ChatMessage> {
    let mut messages = Vec::with_capacity(history.len().saturating_add(1));

    if let Some(prompt) = system_prompt.and_then(|p| {
        let trimmed = p.trim();
        if trimmed.is_empty() {
            None
        } else {
            Some(trimmed.to_string())
        }
    }) {
        messages.push(ChatMessage::system(prompt));
    }

    for entry in history {
        match entry.role.as_str() {
            "user" | "assistant" => {
                let mut content = entry.text.clone().unwrap_or_default();
                let attachment_notes = message_attachment_summaries(&entry.attachments);
                if !attachment_notes.is_empty() {
                    if !content.trim_end().is_empty() {
                        content.push_str("\n\n");
                    }
                    content.push_str("[Attachments]\n");
                    for note in attachment_notes {
                        content.push_str("- ");
                        content.push_str(&note);
                        content.push('\n');
                    }
                }

                if content.trim().is_empty() {
                    continue;
                }

                messages.push(ChatMessage::from_role(entry.role.clone(), content));
            }
            _ => continue,
        }
    }

    messages
}
