class DermAidError(Exception):
    """Base exception for DermAId."""
    def __init__(self, message: str = "An error occurred in DermAId."):
        self.message = message
        super().__init__(self.message)


class KnowledgeBaseError(DermAidError):
    """Raised when the knowledge base encounters an error (e.g., not found, indexing failed)."""
    def __init__(self, message: str = "Knowledge base error."):
        super().__init__(message)


class KnowledgeBaseEmptyError(KnowledgeBaseError):
    """Raised when the knowledge base has no documents."""
    def __init__(self, collection_name: str):
        super().__init__(f"Knowledge base collection '{collection_name}' is empty.")


class KnowledgeBaseQueryError(KnowledgeBaseError):
    """Raised when a retrieval query fails."""
    def __init__(self, query: str, reason: str = ""):
        msg = f"Query failed: '{query}'."
        if reason:
            msg += f" Reason: {reason}"
        super().__init__(msg)


class VisionModelError(DermAidError):
    """Raised when the vision model fails to load or process an image."""
    def __init__(self, message: str = "Vision model error."):
        super().__init__(message)


class VisionModelLoadError(VisionModelError):
    """Raised when the vision model cannot be loaded."""
    def __init__(self, model_name: str):
        super().__init__(f"Failed to load vision model: {model_name}")


class VisionImageProcessingError(VisionModelError):
    """Raised when the image cannot be processed."""
    def __init__(self, image_path: str, reason: str = ""):
        msg = f"Failed to process image '{image_path}'."
        if reason:
            msg += f" {reason}"
        super().__init__(msg)


class AgentError(DermAidError):
    """Raised when the LLM agent encounters an error."""
    def __init__(self, message: str = "Agent error."):
        super().__init__(message)


class AgentExecutionError(AgentError):
    """Raised when the agent fails to execute a query."""
    def __init__(self, query: str, reason: str = ""):
        msg = f"Agent failed to answer: '{query}'."
        if reason:
            msg += f" {reason}"
        super().__init__(msg)


class AgentToolError(AgentError):
    """Raised when a specific tool fails."""
    def __init__(self, tool_name: str, reason: str = ""):
        msg = f"Tool '{tool_name}' failed."
        if reason:
            msg += f" {reason}"
        super().__init__(msg)


class SessionError(DermAidError):
    """Raised for session (JSON KV cache) errors."""
    def __init__(self, message: str = "Session error."):
        super().__init__(message)


class SessionNotFoundError(SessionError):
    """Raised when a session ID is not found."""
    def __init__(self, session_id: str):
        super().__init__(f"Session '{session_id}' not found.")


class SessionStorageError(SessionError):
    """Raised when reading/writing session file fails."""
    def __init__(self, session_id: str, reason: str = ""):
        msg = f"Failed to store session '{session_id}'."
        if reason:
            msg += f" {reason}"
        super().__init__(msg)


class ConfigurationError(DermAidError):
    """Raised for invalid configuration."""
    def __init__(self, message: str = "Configuration error."):
        super().__init__(message)