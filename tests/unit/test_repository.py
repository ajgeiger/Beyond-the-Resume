"""House rules for the repository and checks on the committed index."""

import ast
import json
import re
from pathlib import Path

import pytest

from rag_chatbot.guardrails.ContactScrubber import ContactScrubber

ROOT = Path(__file__).resolve().parents[2]
TEXT_SUFFIXES = {".py", ".md", ".toml", ".json", ".yml", ".yaml", ".txt", ".example", ".sh", ".cfg", ""}
SKIPPED_PARTS = {".git", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache", "build", "raw", "models"}


def repository_files():
    for path in ROOT.rglob("*"):
        if not path.is_file() or SKIPPED_PARTS & set(path.parts) or "egg-info" in str(path):
            continue
        if path.suffix in TEXT_SUFFIXES or path.name in ("Makefile", "Dockerfile"):
            yield path


def python_files():
    return [path for path in (ROOT / "src").rglob("*.py")]


def test_no_long_dashes_anywhere():
    offenders = [str(path) for path in repository_files() if re.search("[\u2013\u2014]", path.read_text("utf-8"))]
    assert offenders == []


def test_each_class_lives_in_a_file_named_after_it():
    for path in python_files():
        tree = ast.parse(path.read_text("utf-8"))
        classes = [node.name for node in tree.body if isinstance(node, ast.ClassDef)]
        assert classes in ([], [path.stem]), f"{path} defines {classes}"


def test_abstract_classes_start_with_abstract():
    for path in python_files():
        tree = ast.parse(path.read_text("utf-8"))
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and any(getattr(base, "id", "") == "ABC" for base in node.bases):
                assert node.name.startswith("Abstract"), node.name


def test_imports_are_at_the_top_of_each_module():
    for path in python_files():
        tree = ast.parse(path.read_text("utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                inner = [item for item in ast.walk(node) if isinstance(item, ast.Import | ast.ImportFrom)]
                assert inner == [], f"{path}:{node.name} imports inside a function"


def test_no_secrets_or_env_files_are_committed():
    assert not (ROOT / ".env").exists() or ".env" in (ROOT / ".gitignore").read_text("utf-8")
    for path in repository_files():
        assert not re.search(r"sk-[A-Za-z0-9_-]{20,}", path.read_text("utf-8")), path


def committed_chunks():
    path = ROOT / "data" / "processed" / "index" / "chunks.json"
    if not path.is_file():
        pytest.skip("No built index in the repository.")
    return json.loads(path.read_text("utf-8"))


def test_committed_index_has_no_contact_details():
    for row in committed_chunks():
        text = f"{row['section']} {row['text']}"
        assert not ContactScrubber.contains_contact(text), row["chunk_id"]
        assert "gmail" not in text.lower(), row["chunk_id"]


def test_committed_index_follows_the_content_rules():
    text = " ".join(f"{row['section']} {row['text']}" for row in committed_chunks()).lower()
    assert "ischool" not in text
    assert "lead data scientist" not in text
    for phrase in ("open to work", "open to new", "looking for a new role", "available for hire"):
        assert phrase not in text
