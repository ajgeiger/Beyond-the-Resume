"""The ASGI entry point: uvicorn rag_chatbot.api.app:app

Settings come from configs/settings.toml (or CONFIG_PATH) and the environment.
"""

from rag_chatbot.api.RagApi import RagApi
from rag_chatbot.chat.RagRuntime import RagRuntime
from rag_chatbot.config.Settings import Settings

app = RagApi(RagRuntime.from_settings(Settings.load())).build()
