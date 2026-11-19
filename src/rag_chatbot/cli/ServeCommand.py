"""Run the API with uvicorn."""

import uvicorn

from rag_chatbot.api.RagApi import RagApi
from rag_chatbot.chat.RagRuntime import RagRuntime
from rag_chatbot.config.Settings import Settings


class ServeCommand:
    def __init__(self, settings: Settings):
        self.settings = settings

    def run(self) -> int:
        app = RagApi(RagRuntime.from_settings(self.settings)).build()
        uvicorn.run(app, host=self.settings.api.host, port=self.settings.api.port, log_level="info")
        return 0
