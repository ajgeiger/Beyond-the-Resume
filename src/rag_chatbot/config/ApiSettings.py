"""Where the API listens and who may call it."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ApiSettings:
    host: str = "127.0.0.1"
    port: int = 8070
    # Empty means no key is required. Set CHATBOT_API_KEY to turn auth on.
    api_key: str = field(default="", repr=False)
