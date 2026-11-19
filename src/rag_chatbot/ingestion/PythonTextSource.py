"""Read the human text out of a Python file, such as a resume build script."""

import ast

from rag_chatbot.ingestion.AbstractSource import AbstractSource
from rag_chatbot.ingestion.Section import Section


class PythonTextSource(AbstractSource):
    """Collect the string literals that read like sentences.

    Docstrings, file paths, font names and other short code strings are
    skipped. What is left is the text the script writes, in file order.
    """

    MIN_WORDS = 4

    def sections(self) -> list[Section]:
        tree = ast.parse(self.path.read_text(encoding="utf-8"))
        docstrings = self.docstring_ids(tree)
        paragraphs = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Constant) or not isinstance(node.value, str) or id(node) in docstrings:
                continue
            text = " ".join(node.value.split())
            if len(text.split()) >= self.MIN_WORDS and not text.startswith(("w:", "/", "<")):
                paragraphs.append(text)
        ordered = list(dict.fromkeys(paragraphs))
        return [Section(self.name, self.name, tuple(ordered))] if ordered else []

    @staticmethod
    def docstring_ids(tree: ast.AST) -> set[int]:
        found = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
                first = node.body[0]
                if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                    found.add(id(first.value))
        return found
