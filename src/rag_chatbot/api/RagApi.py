"""The FastAPI app: POST /chat, POST /retrieve, GET /health and GET /info."""

from fastapi import Depends, FastAPI, HTTPException

from rag_chatbot.api.ApiKeyGuard import ApiKeyGuard
from rag_chatbot.api.schemas.ChatRequestBody import ChatRequestBody
from rag_chatbot.api.schemas.ChatResponseBody import ChatResponseBody
from rag_chatbot.api.schemas.HealthResponse import HealthResponse
from rag_chatbot.api.schemas.InfoResponse import InfoResponse
from rag_chatbot.api.schemas.RetrieveRequest import RetrieveRequest
from rag_chatbot.api.schemas.RetrieveResponse import RetrieveResponse
from rag_chatbot.chat.RagRuntime import RagRuntime


class RagApi:
    """Wrap a loaded RagRuntime in HTTP routes. /health stays open so the host can probe it."""

    def __init__(self, runtime: RagRuntime):
        self.runtime = runtime
        self.guard = ApiKeyGuard(runtime.settings.api.api_key)
        self.max_query_chars = runtime.settings.guardrails.max_question_chars

    def build(self) -> FastAPI:
        app = FastAPI(title="Beyond the Resume", version="0.1.0", description="Answers about Alex Geiger's work.")
        protected = [Depends(self.guard)]

        @app.get("/health", response_model=HealthResponse)
        def health() -> HealthResponse:
            manifest = self.runtime.manifest
            return HealthResponse(
                status="ok", index_version=manifest.index_version, chunk_count=len(self.runtime.store)
            )

        @app.get("/info", response_model=InfoResponse, dependencies=protected)
        def info() -> InfoResponse:
            return InfoResponse(**self.runtime.info())

        @app.post("/chat", response_model=ChatResponseBody, dependencies=protected)
        def chat(body: ChatRequestBody) -> ChatResponseBody:
            history = [turn.model_dump() for turn in body.history]
            return ChatResponseBody(**self.runtime.chat(body.question, history, body.top_k))

        @app.post("/retrieve", response_model=RetrieveResponse, dependencies=protected)
        def retrieve(body: RetrieveRequest) -> RetrieveResponse:
            if len(body.query) > self.max_query_chars:
                raise HTTPException(status_code=422, detail=f"query is longer than {self.max_query_chars} characters.")
            return RetrieveResponse(**self.runtime.retrieve(body.query, body.top_k, body.mode, body.rerank))

        return app
