"""Read the human text out of a Python file, such as a resume build script."""

import ast
from pathlib import Path

from rag_chatbot.ingestion.AbstractSource import AbstractSource
from rag_chatbot.ingestion.Section import Section


class PythonTextSource(AbstractSource):
    """Collect the string literals that read like sentences.

    Class and module assignments become sections named after the variable, so
    About paragraphs stay together and are not packed with hero or contact
    copy. Docstrings, file paths, font names, leftover titles and dates are
    skipped. F-strings are joined and Profile constants are filled in when a
    Profile class is in this file or a sibling Python file. Other sentence
    strings, such as script arguments, still appear in file order.
    """

    MIN_WORDS = 4
    SENTENCE_END = (".", "!", "?")
    ABBREVIATION_END = ("Inc.", "Ltd.", "Jr.", "Sr.", "vs.")

    def __init__(self, path, name: str, assignments: tuple[str, ...] = ()):
        super().__init__(path, name)
        self.wanted_assignments = set(assignments)

    def sections(self) -> list[Section]:
        tree = ast.parse(self.path.read_text(encoding="utf-8"))
        constants = self.profile_constants()
        skip_ids = self.docstring_ids(tree) | self.joined_constant_ids(tree)
        used_ids: set[int] = set()
        sections = []
        for name, nodes in self.assignments(tree):
            if self.wanted_assignments and name not in self.wanted_assignments:
                used_ids.update(id(node) for node in nodes)
                continue
            paragraphs = self.paragraphs_from(nodes, constants, skip_ids)
            used_ids.update(id(node) for node in nodes)
            if paragraphs:
                title = f"{self.name} > {self.display_name(name)}"
                sections.append(Section(self.name, title, paragraphs))
        leftover_nodes = [
            node
            for node in ast.walk(tree)
            if self.is_text_node(node) and id(node) not in skip_ids | used_ids | self.docstring_ids(tree)
        ]
        leftover = self.paragraphs_from(leftover_nodes, constants, skip_ids)
        if leftover:
            sections.append(Section(self.name, self.name, leftover))
        return sections

    def paragraphs_from(self, nodes: list[ast.AST], constants: dict[str, str], skip_ids: set[int]) -> tuple[str, ...]:
        paragraphs = []
        for node in nodes:
            text = self.render_node(node, constants)
            text = " ".join(text.split())
            if self.is_prose(text) and not text.startswith(("w:", "/", "<")):
                paragraphs.append(text)
        return tuple(dict.fromkeys(paragraphs))

    def profile_constants(self) -> dict[str, str]:
        """String attributes of a Profile class in this file or a sibling file."""
        found: dict[str, str] = {}
        paths = [self.path, *sorted(self.path.parent.glob("*.py"))]
        seen: set[Path] = set()
        for path in paths:
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except (OSError, SyntaxError):
                continue
            found.update(self.class_string_constants(tree, "Profile"))
        return found

    @staticmethod
    def class_string_constants(tree: ast.AST, class_name: str) -> dict[str, str]:
        values: dict[str, str] = {}
        for node in tree.body:
            if not isinstance(node, ast.ClassDef) or node.name != class_name:
                continue
            for statement in node.body:
                if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
                    continue
                target = statement.targets[0]
                if isinstance(target, ast.Name) and isinstance(statement.value, ast.Constant):
                    if isinstance(statement.value.value, str):
                        values[target.id] = statement.value.value
        return values

    @classmethod
    def assignments(cls, tree: ast.AST) -> list[tuple[str, list[ast.AST]]]:
        """Name and string nodes for each module or class assignment, in order."""
        found: list[tuple[str, list[ast.AST]]] = []
        for node in tree.body:
            statements = node.body if isinstance(node, ast.ClassDef) else [node]
            for statement in statements:
                if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
                    continue
                target = statement.targets[0]
                if not isinstance(target, ast.Name):
                    continue
                nodes = cls.string_nodes(statement.value)
                if nodes:
                    found.append((target.id, nodes))
        return found

    @classmethod
    def string_nodes(cls, node: ast.AST) -> list[ast.AST]:
        if isinstance(node, (ast.Constant, ast.JoinedStr)):
            return [node] if cls.is_text_node(node) else []
        if isinstance(node, (ast.Tuple, ast.List)):
            found: list[ast.AST] = []
            for element in node.elts:
                found.extend(cls.string_nodes(element))
            return found
        return []

    @classmethod
    def render_node(cls, node: ast.AST, constants: dict[str, str]) -> str:
        if isinstance(node, ast.JoinedStr):
            return cls.render_joined(node, constants)
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        return ""

    @classmethod
    def render_joined(cls, node: ast.JoinedStr, constants: dict[str, str]) -> str:
        """Turn an f-string into plain text, filling Profile attributes when known."""
        parts = []
        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                parts.append(value.value)
            elif isinstance(value, ast.FormattedValue):
                parts.append(cls.resolve_format(value.value, constants))
        return "".join(parts)

    @staticmethod
    def resolve_format(node: ast.AST, constants: dict[str, str]) -> str:
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.value.id == "Profile":
                return constants.get(node.attr, node.attr)
        if isinstance(node, ast.Name):
            return constants.get(node.id, node.id)
        return ""

    @staticmethod
    def is_text_node(node: ast.AST) -> bool:
        return isinstance(node, ast.JoinedStr) or (isinstance(node, ast.Constant) and isinstance(node.value, str))

    @staticmethod
    def joined_constant_ids(tree: ast.AST) -> set[int]:
        """Constants that already belong to an f-string, so they are not added twice."""
        found = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.JoinedStr):
                for value in node.values:
                    if isinstance(value, ast.Constant):
                        found.add(id(value))
        return found

    @classmethod
    def is_prose(cls, text: str) -> bool:
        """Keep sentences; drop leftover titles, dates and bare skill lists."""
        if len(text.split()) < cls.MIN_WORDS:
            return False
        if any(text.endswith(ending) for ending in cls.ABBREVIATION_END):
            return False
        return text.endswith(cls.SENTENCE_END)

    @staticmethod
    def display_name(name: str) -> str:
        cleaned = name
        for suffix in ("_PARAGRAPHS", "_CARDS", "_TEXT", "_LEAD", "_TITLE", "_LABEL"):
            if cleaned.endswith(suffix):
                cleaned = cleaned[: -len(suffix)]
                break
        return cleaned.replace("_", " ").capitalize()

    @staticmethod
    def docstring_ids(tree: ast.AST) -> set[int]:
        found = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
                first = node.body[0]
                if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                    found.add(id(first.value))
        return found
