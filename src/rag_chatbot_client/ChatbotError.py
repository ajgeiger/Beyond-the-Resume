"""Raised when the chatbot API cannot be reached or refuses a call."""


class ChatbotError(RuntimeError):
    """status_code is the HTTP status, or None when the service could not be reached at all."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code

    @property
    def is_outage(self) -> bool:
        return self.status_code is None or self.status_code >= 500
