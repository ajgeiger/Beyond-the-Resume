"""The rag-chatbot command: sources, index, serve and eval."""

import argparse
from pathlib import Path

from rag_chatbot.cli.BuildIndexCommand import BuildIndexCommand
from rag_chatbot.cli.EvaluateCommand import EvaluateCommand
from rag_chatbot.cli.ServeCommand import ServeCommand
from rag_chatbot.cli.SyncSourcesCommand import SyncSourcesCommand
from rag_chatbot.config.Settings import Settings


class CommandLine:
    def parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(prog="rag-chatbot", description="Build, serve and evaluate the chatbot.")
        parser.add_argument("--config", type=Path, default=None, help="Path to settings.toml.")
        commands = parser.add_subparsers(dest="command", required=True)
        commands.add_parser("sources", help="Copy the source documents into data/raw.")
        index = commands.add_parser("index", help="Chunk, embed and save the index when a source changed.")
        index.add_argument("--force", action="store_true", help="Rebuild even when nothing changed.")
        commands.add_parser("serve", help="Run the API.")
        evaluate = commands.add_parser("eval", help="Report recall at k and MRR.")
        evaluate.add_argument("--cases", type=Path, default=Path("evaluation/retrieval_eval.json"))
        evaluate.add_argument("--output", type=Path, default=Path("evaluation/results/retrieval_eval_results.json"))
        evaluate.add_argument("--rerank", action="store_true", help="Also score hybrid with the cross encoder.")
        return parser

    def run(self, argv: list[str] | None = None) -> int:
        args = self.parser().parse_args(argv)
        settings = Settings.load(args.config)
        if args.command == "sources":
            return SyncSourcesCommand(settings).run()
        if args.command == "index":
            return BuildIndexCommand(settings).run(args.force)
        if args.command == "serve":
            return ServeCommand(settings).run()
        return EvaluateCommand(settings).run(args.cases, args.output, args.rerank)


def main() -> None:
    raise SystemExit(CommandLine().run())
