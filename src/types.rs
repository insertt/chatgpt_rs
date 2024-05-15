use std::fmt;

#[cfg(feature = "functions")]
use crate::functions::FunctionCall;
use serde::{de, Deserialize, Serialize};

/// A role of a message sender, can be:
/// - `System`, for starting system message, that sets the tone of model
/// - `Assistant`, for messages sent by ChatGPT
/// - `User`, for messages sent by user
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize, Eq, Ord)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    /// A system message, automatically sent at the start to set the tone of the model
    System,
    /// A message sent by ChatGPT
    Assistant,
    /// A message sent by the user
    User,
    /// A message related to ChatGPT functions. Does not have much use without the `functions` feature.
    Function,
}

/// Type of the message content
#[derive(Debug, Clone, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChatMessageType {
    /// Text message
    Text,
    /// Image URL
    ImageUrl,
}

/// Container for the sent/received ChatGPT messages
#[derive(Debug, Clone, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct ChatMessage {
    /// Role of message sender
    pub role: Role,
    /// Actual content of the message
    #[serde(deserialize_with = "string_or_array")]
    pub content: Vec<ChatMessageContent>,
    /// Function call (if present)
    #[cfg(feature = "functions")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function_call: Option<FunctionCall>,
}

impl ChatMessage {
    /// Creates a new chat message
    pub fn new(role: Role, content: impl Into<Vec<ChatMessageContent>>) -> Self {
        Self {
            role,
            content: content.into(),
            #[cfg(feature = "functions")]
            function_call: None,
        }
    }

    /// Returns concatenated content of the message
    pub fn content(&self) -> String {
        self.content
            .iter()
            .map(|content| content)
            .fold(String::new(), |mut acc, elem| {
                acc.push_str(&elem.to_string());
                acc
            })
    }

    /// Returns the raw content of the message
    pub fn raw_content(&self) -> &[ChatMessageContent] {
        self.content.as_slice()
    }

    /// Converts multiple response chunks into multiple (or a single) chat messages
    #[cfg(feature = "streams")]
    pub fn from_response_chunks(chunks: Vec<ResponseChunk>) -> Vec<Self> {
        let mut result: Vec<Self> = Vec::new();
        let mut responses: Vec<(Role, String)> = Vec::new();

        for chunk in chunks {
            match chunk {
                ResponseChunk::Content {
                    delta,
                    response_index,
                } => {
                    let (_, response) = responses
                        .get_mut(response_index)
                        .expect("Invalid response chunk sequence!");

                    response.push_str(&delta);
                }
                ResponseChunk::BeginResponse {
                    role,
                    response_index: _,
                } => {
                    responses.push((role, String::with_capacity(16)));

                    let msg = ChatMessage {
                        role,
                        content: vec![],
                        #[cfg(feature = "functions")]
                        function_call: None,
                    };
                    result.push(msg);
                }
                _ => {}
            }
        }

        responses
            .into_iter()
            .map(|(role, response)| ChatMessage::new(role, &[ChatMessageContent::text(response)]))
            .collect::<Vec<_>>()
    }
}

/// Content of the message
#[derive(Debug, Clone, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ChatMessageContent {
    /// Text content
    Text {
        #[serde(deserialize_with = "deserialize_maybe_null")]
        text: String,
    },
    /// Image URL content
    ImageUrl { image_url: ImageUrlContent },
}

impl fmt::Display for ChatMessageContent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ChatMessageContent::Text { text } => write!(f, "{}", text),
            ChatMessageContent::ImageUrl {
                image_url: ImageUrlContent { url },
            } => write!(f, "{}", url),
        }
    }
}

impl ChatMessageContent {
    /// Creates a new text message content
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text { text: text.into() }
    }

    /// Creates a new image URL message content
    pub fn image_url(url: impl Into<String>) -> Self {
        Self::ImageUrl {
            image_url: ImageUrlContent { url: url.into() },
        }
    }
}

impl From<String> for ChatMessageContent {
    fn from(value: String) -> Self {
        ChatMessageContent::Text { text: value }
    }
}

impl From<&String> for ChatMessageContent {
    fn from(value: &String) -> Self {
        ChatMessageContent::Text {
            text: value.clone(),
        }
    }
}

impl From<&str> for ChatMessageContent {
    fn from(value: &str) -> Self {
        ChatMessageContent::Text {
            text: value.to_string(),
        }
    }
}

/// Image URL content
#[derive(Debug, Clone, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct ImageUrlContent {
    url: String,
}

/// A request struct sent to the API to request a message completion
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CompletionRequest<'a> {
    /// The model to be used, currently `gpt-3.5-turbo`, but may change in future
    pub model: &'a str,
    /// The message history, including the message that requires completion, which should be the last one
    pub messages: &'a Vec<ChatMessage>,
    /// Whether the message response should be gradually streamed
    pub stream: bool,
    /// The extra randomness of response
    pub temperature: f32,
    /// Controls diversity via nucleus sampling, not recommended to use with temperature
    pub top_p: f32,
    /// Controls the maximum number of tokens to generate in the completion
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    /// Determines how much to penalize new tokens based on their existing frequency so far
    pub frequency_penalty: f32,
    /// Determines how much to penalize new tokens pased on their existing presence so far
    pub presence_penalty: f32,
    /// Determines the amount of output responses
    #[serde(rename = "n")]
    pub reply_count: u32,
    /// All functions that can be called by ChatGPT
    #[cfg(feature = "functions")]
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub functions: &'a Vec<serde_json::Value>,
}

/// Represents a response from the API
#[derive(Debug, Clone, PartialEq, PartialOrd, Deserialize)]
#[serde(untagged)]
pub enum ServerResponse {
    /// An error occurred, most likely the model was just overloaded
    Error {
        /// The error that happened
        error: CompletionError,
    },
    /// Completion successfuly completed
    Completion(CompletionResponse),
}

/// An error happened while requesting completion
#[derive(Debug, Clone, PartialEq, PartialOrd, Deserialize)]
pub struct CompletionError {
    /// Message, describing the error
    pub message: String,
    /// The type of error. Example: `server_error`
    #[serde(rename = "type")]
    pub error_type: String,
}

/// A response struct received from the API after requesting a message completion
#[derive(Debug, Clone, PartialEq, PartialOrd, Deserialize)]
pub struct CompletionResponse {
    /// Unique ID of the message, but not in a UUID format.
    /// Example: `chatcmpl-6p5FEv1JHictSSnDZsGU4KvbuBsbu`
    #[serde(rename = "id")]
    pub message_id: Option<String>,
    /// Unix seconds timestamp of when the response was created
    #[serde(rename = "created")]
    pub created_timestamp: Option<u64>,
    /// The model that was used for this completion
    pub model: String,
    /// Token usage of this completion
    pub usage: TokenUsage,
    /// Message choices for this response, guaranteed to contain at least one message response
    #[serde(rename = "choices")]
    pub message_choices: Vec<MessageChoice>,
}

impl CompletionResponse {
    /// A shortcut to access the message response
    pub fn message(&self) -> &ChatMessage {
        // Unwrap is safe here, as we know that at least one message choice is provided
        &self.message_choices.first().unwrap().message
    }
}

/// A message completion choice struct
#[derive(Debug, Clone, PartialEq, PartialOrd, Deserialize)]
pub struct MessageChoice {
    /// The actual message
    pub message: ChatMessage,
    /// The reason completion was stopped
    pub finish_reason: String,
    /// The index of this message in the outer `message_choices` array
    pub index: u32,
}

/// The token usage of a specific response
#[derive(Debug, Clone, PartialEq, PartialOrd, Deserialize)]
pub struct TokenUsage {
    /// Tokens spent on the prompt message (including previous messages)
    pub prompt_tokens: u32,
    /// Tokens spent on the completion message
    pub completion_tokens: u32,
    /// Total amount of tokens used (`prompt_tokens + completion_tokens`)
    pub total_tokens: u32,
}

/// A single response chunk, returned from streamed request
#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord)]
#[cfg(feature = "streams")]
pub enum ResponseChunk {
    /// A chunk of message content
    Content {
        /// Piece of message content
        delta: String,
        /// Index of the message. Used when `reply_count` is set to more than 1 in API config
        response_index: usize,
    },
    /// Marks beginning of a new message response, with no actual content yet
    BeginResponse {
        /// The respondent's role (usually `Assistant`)
        role: Role,
        /// Index of the message. Used when `reply_count` is set to more than 1 in API config
        response_index: usize,
    },
    /// Ends a single message response response
    CloseResponse {
        /// Index of the message finished. Used when `reply_count` is set to more than 1 in API config
        response_index: usize,
    },
    /// Marks end of stream
    Done,
}

/// A part of a chunked inbound response
#[derive(Debug, Clone, Deserialize)]
#[cfg(feature = "streams")]
pub struct InboundResponseChunk {
    /// All message chunks in this response part (only one usually)
    pub choices: Vec<InboundChunkChoice>,
}

/// A single message part of a chunked inbound response
#[derive(Debug, Clone, Deserialize)]
#[cfg(feature = "streams")]
pub struct InboundChunkChoice {
    /// The part value of the response
    pub delta: InboundChunkPayload,
    /// Index of the message this chunk refers to
    pub index: usize,
}

/// Contains different chunked inbound response payloads
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
#[cfg(feature = "streams")]
pub enum InboundChunkPayload {
    /// Begins a single message by announcing roles (usually `assistant`)
    AnnounceRoles {
        /// The announced role
        role: Role,
    },
    /// Streams a part of message content
    StreamContent {
        /// The part of content
        content: String,
    },
    /// Closes a single message
    Close {},
}

fn string_or_array<'de, D>(deserializer: D) -> Result<Vec<ChatMessageContent>, D::Error>
where
    D: de::Deserializer<'de>,
{
    struct StringOrArray;

    impl<'de> de::Visitor<'de> for StringOrArray {
        type Value = Vec<ChatMessageContent>;

        fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
            formatter.write_str("string or list of ChatMessageContent")
        }

        fn visit_str<E>(self, s: &str) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            Ok(vec![ChatMessageContent::text(s.to_string())])
        }

        fn visit_seq<S>(self, seq: S) -> Result<Self::Value, S::Error>
        where
            S: de::SeqAccess<'de>,
        {
            Deserialize::deserialize(de::value::SeqAccessDeserializer::new(seq))
        }
    }

    deserializer.deserialize_any(StringOrArray)
}

fn deserialize_maybe_null<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: de::Deserializer<'de>,
{
    let buf = Option::<String>::deserialize(deserializer)?;
    Ok(buf.unwrap_or(String::new()))
}
