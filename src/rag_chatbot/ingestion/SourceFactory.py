"""Turn the entries in config/sources.toml into source readers."""

from rag_chatbot.config.SourceCatalog import SourceCatalog
from rag_chatbot.ingestion.AbstractSource import AbstractSource
from rag_chatbot.ingestion.MarkdownSource import MarkdownSource
from rag_chatbot.ingestion.PdfSource import PdfSource
from rag_chatbot.ingestion.PythonTextSource import PythonTextSource


class SourceFactory:
    @staticmethod
    def create(catalog: SourceCatalog) -> list[AbstractSource]:
        sources = []
        for spec in catalog.sources:
            if spec.kind == "pdf":
                sources.append(PdfSource(spec.path, spec.name))
            elif spec.kind == "markdown":
                sources.append(MarkdownSource(spec.path, spec.name, catalog.skipped_sections))
            elif spec.kind == "python":
                sources.append(PythonTextSource(spec.path, spec.name))
            else:
                raise ValueError(f"Unknown source kind: {spec.kind}")
        return sources
