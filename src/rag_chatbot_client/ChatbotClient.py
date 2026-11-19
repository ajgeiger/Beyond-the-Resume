"""A small HTTP client for the chatbot API. Needs only httpx."""

import httpx

from rag_chatbot_client.ChatbotError import ChatbotError
from rag_chatbot_client.ChatResponse import ChatResponse


class ChatbotClient:
    """ChatbotClient(url, api_key).chat(question) returns a ChatResponse.

    Every failure, from a refused connection to a 401, raises ChatbotError, so
    callers handle one exception type. Pass http to reuse a configured
    httpx.Client, such as FastAPI's TestClient in tests.
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8070",
        api_key: str = "",
        timeout: float = 30.0,
        http: httpx.Client | None = None,
    ):
        self.http = http or httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout)
        if api_key:
            self.http.headers["X-API-Key"] = api_key

    def chat(self, question: str, history: list[dict] | None = None, top_k: int | None = None) -> ChatResponse:
        body = {"question": question, "history": history or []}
        if top_k:
            body["top_k"] = top_k
        return ChatResponse.from_dict(self.call("POST", "/chat", body))

    def retrieve(self, query: str, top_k: int = 5, mode: str = "hybrid") -> dict:
        return self.call("POST", "/retrieve", {"query": query, "top_k": top_k, "mode": mode})

    def health(self) -> dict:
        return self.call("GET", "/health")

    def info(self) -> dict:
        return self.call("GET", "/info")

    def call(self, method: str, path: str, body: dict | None = None) -> dict:
        try:
            response = self.http.request(method, path, json=body)
        except httpx.HTTPError as error:
            raise ChatbotError(f"Could not reach the chatbot API: {error}") from error
        if response.status_code >= 400:
            raise ChatbotError(f"The chatbot API returned {response.status_code}.", response.status_code)
        return response.json()

    def close(self) -> None:
        self.http.close()

    def __enter__(self) -> "ChatbotClient":
        return self

    def __exit__(self, *_args) -> None:
        self.close()
