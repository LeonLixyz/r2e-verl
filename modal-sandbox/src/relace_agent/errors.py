from typing import Any


class AgentStop(BaseException):
    """Control flow mechanism to break out of an iterator."""


class AgentError(Exception):
    """Error during agent's operation."""


class BuildError(AgentError):
    """Error during site build."""


class TestError(AgentError):
    """Error during testing of generated site."""


class DeployError(AgentError):
    """Error during deployment of generated site."""


class ToolError(Exception):
    """Tool call error that should be surfaced back to the LLM."""


class ToolInputError(ToolError):
    """Tool call was made with invalid input."""

    def __init__(self, msg: Any) -> None:
        super().__init__(f"Invalid tool input: {msg}")


class EmbeddingError(Exception):
    """Raised when an embedding-related operation fails."""


class UpsertError(Exception):
    """Raised when an upsert/delete/query operation against the vector store fails."""
