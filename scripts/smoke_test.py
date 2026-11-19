"""Ask a running chatbot API one question through the client and print the result.

python scripts/smoke_test.py --url http://127.0.0.1:8070
python scripts/smoke_test.py --url https://your-user-rag-chatbot.hf.space --api-key $CHATBOT_API_KEY
"""

import argparse
import os

from rag_chatbot_client import ChatbotClient, ChatbotError


class SmokeTest:
    def __init__(self, url: str, api_key: str, question: str):
        self.client = ChatbotClient(url, api_key=api_key, timeout=60)
        self.question = question

    def run(self) -> int:
        try:
            health = self.client.health()
            info = self.client.info()
            response = self.client.chat(self.question)
        except ChatbotError as error:
            print(f"failed: {error}")
            return 1
        print(f"health: {health['status']}, index {health['index_version']}, {health['chunk_count']} chunks")
        print(f"model: {info['llm_provider']} {info['llm_model']}")
        print(f"status: {response.status} in {response.took_ms:.0f} ms")
        print(f"answer: {response.answer}")
        for citation in response.citations:
            print(f"  [{citation.number}] {citation.source} > {citation.section}")
        return 0 if response.is_answer else 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default=os.environ.get("CHATBOT_API_URL", "http://127.0.0.1:8070"))
    parser.add_argument("--api-key", default=os.environ.get("CHATBOT_API_KEY", ""))
    parser.add_argument("--question", default="What does he work on at Signature Aviation?")
    args = parser.parse_args()
    raise SystemExit(SmokeTest(args.url, args.api_key, args.question).run())


if __name__ == "__main__":
    main()
