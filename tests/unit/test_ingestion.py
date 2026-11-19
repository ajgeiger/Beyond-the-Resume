"""Readers, privacy cleaning and token based heading aware chunking."""

import pytest

from rag_chatbot.chunking.HeadingAwareChunker import HeadingAwareChunker
from rag_chatbot.config.ChunkSettings import ChunkSettings
from rag_chatbot.config.Correction import Correction
from rag_chatbot.config.SourceCatalog import SourceCatalog
from rag_chatbot.config.SourceSpec import SourceSpec
from rag_chatbot.guardrails.ContactScrubber import ContactScrubber
from rag_chatbot.ingestion.IngestionPipeline import IngestionPipeline
from rag_chatbot.ingestion.MarkdownSource import MarkdownSource
from rag_chatbot.ingestion.PdfSource import PdfSource
from rag_chatbot.ingestion.PythonTextSource import PythonTextSource
from rag_chatbot.ingestion.Section import Section
from rag_chatbot.ingestion.SourceFactory import SourceFactory
from rag_chatbot.ingestion.TextCleaner import TextCleaner


def test_markdown_keeps_heading_paths_and_skips_nested_setup(tmp_path):
    path = tmp_path / "README.md"
    path.write_text(
        "# Project\n\nA [topic model](models/x.py) for **news**.\n\n"
        "## Views\n\n### Cards\n\n| View | Purpose |\n| --- | --- |\n| Cards | Counts per group |\n\n"
        "## Setup\n\nHidden text.\n\n### Extra setup\n\nAlso hidden.\n\n"
        "## Data\n\nUseful data.\n\n```python\nprint('not prose')\n```\n",
        encoding="utf-8",
    )
    sections = MarkdownSource(path, "Readme", ("setup",)).sections()
    assert [item.title for item in sections] == ["Project", "Views > Cards", "Data"]
    assert sections[0].paragraphs == ("A topic model for news.",)
    assert sections[1].paragraphs == ("Cards. Purpose: Counts per group.",)
    assert not any("Hidden" in str(item) or "print" in str(item) for item in sections)


def test_python_reader_ignores_docstrings_and_short_literals(tmp_path):
    path = tmp_path / "build.py"
    path.write_text(
        '"""Build the resume."""\nFONT = "Calibri"\nbullet("Built forecasting models for 100 locations.")\n'
    )
    assert PythonTextSource(path, "Script").sections()[0].paragraphs == ("Built forecasting models for 100 locations.",)


def test_pdf_reader_titles_each_role_on_its_own(tmp_path, monkeypatch):
    source = PdfSource(tmp_path / "resume.pdf", "Resume")
    monkeypatch.setattr(
        source,
        "text",
        lambda: (
            "Alexander Geiger\nLead Applied AI Scientist\nPROFESSIONAL EXPERIENCE\n"
            "Signature Aviation | Orlando, FL\nLead Applied AI Scientist Jul 2026 to Present\n"
            "\u2022 Built a platform.\nSenior Data Scientist May 2022 to Jul 2026\n"
            "\u2022 Built a pricing engine.\nEDUCATION\nSyracuse University\n"
        ),
    )
    sections = source.sections()
    assert [item.title for item in sections] == [
        "Name and current title",
        "Professional experience > Signature Aviation, Orlando, FL > Lead Applied AI Scientist Jul 2026 to Present",
        "Professional experience > Signature Aviation, Orlando, FL > Senior Data Scientist May 2022 to Jul 2026",
        "Education",
    ]
    assert sections[1].paragraphs[0] == "Signature Aviation, Orlando, FL"
    assert "Built a pricing engine." in sections[2].paragraphs


@pytest.mark.parametrize("size,overlap", [(0, 0), (10, -1), (10, 10), (10, 11)])
def test_chunk_settings_reject_bad_sizes(size, overlap):
    with pytest.raises(ValueError):
        ChunkSettings(size, overlap)


def test_chunker_stays_under_size_and_within_headings(model):
    chunker = HeadingAwareChunker(ChunkSettings(16, 4, min_chunk_tokens=1), model.count_tokens)
    sections = [
        Section("Resume", "Experience", (" ".join(f"word{i}" for i in range(50)),)),
        Section("Resume", "Education", ("Syracuse University is his graduate school.",)),
    ]
    chunks = chunker.chunk(sections)
    assert all(item.token_count <= 16 for item in chunks)
    assert [item.chunk_id for item in chunks] == [f"resume-{i:03d}" for i in range(1, len(chunks) + 1)]
    assert chunks[-1].section == "Education"
    assert all("Syracuse" not in item.text for item in chunks[:-1])


def test_chunker_overlaps_adjacent_chunks(model):
    chunker = HeadingAwareChunker(ChunkSettings(10, 3, min_chunk_tokens=1), model.count_tokens)
    section = Section("Resume", "Experience", ("One two three four five six. Seven eight nine ten eleven twelve.",))
    chunks = chunker.chunk([section])
    assert len(chunks) == 2
    assert chunks[1].text.startswith("four five six.")
    assert chunks[1].token_count == 9


def test_chunker_drops_stray_short_titles(model):
    chunker = HeadingAwareChunker(ChunkSettings(40, 8, min_chunk_tokens=8), model.count_tokens)
    assert chunker.chunk([Section("Resume", "Title", ("Alexander Geiger resume.",))]) == []


@pytest.mark.parametrize(
    "text",
    ["mail person@example.com today", "Call (212) 555-0199", "Call 212-555-0199", "Call +1 212.555.0199"],
)
def test_contact_scrubber(text):
    assert ContactScrubber.contains_contact(text)
    assert not ContactScrubber.contains_contact(ContactScrubber.scrub(text))


def test_cleaner_removes_contacts_dashes_and_stale_title():
    cleaner = TextCleaner((Correction("Lead Data Scientist", "Lead Applied AI Scientist"),))
    text = cleaner.clean("Lead Data Scientist \u2014 call (212) 555-0199 or person@example.com \u2013 thanks.")
    assert "Lead Applied AI Scientist" in text
    assert not ContactScrubber.contains_contact(text)
    assert "\u2014" not in text and "\u2013" not in text


def test_pipeline_cleans_titles_and_text_before_chunking(source_file, settings, model):
    source_file.write_text(
        "# Resume\n\n## person@example.com\n\n"
        "Contact (212) 555-0199. Lead Applied AI Scientist builds forecasting models with Python.\n"
    )
    catalog = SourceCatalog((SourceSpec("Resume", "markdown", source_file),))
    pipeline = IngestionPipeline(
        SourceFactory.create(catalog), catalog, HeadingAwareChunker(settings.chunking, model.count_tokens)
    )
    chunks = pipeline.chunks()
    assert chunks
    assert all(not ContactScrubber.contains_contact(str(item.to_dict())) for item in chunks)


def test_pipeline_deduplicates_with_first_source_winning(source_file, model, settings):
    catalog = SourceCatalog(
        (SourceSpec("Resume", "markdown", source_file), SourceSpec("Second", "markdown", source_file))
    )
    pipeline = IngestionPipeline(
        SourceFactory.create(catalog), catalog, HeadingAwareChunker(settings.chunking, model.count_tokens)
    )
    assert {item.source for item in pipeline.chunks()} == {"Resume"}


def test_pipeline_reports_missing_sources(tmp_path, model, settings):
    missing = tmp_path / "missing.md"
    catalog = SourceCatalog((SourceSpec("Missing", "markdown", missing),))
    pipeline = IngestionPipeline(
        SourceFactory.create(catalog), catalog, HeadingAwareChunker(settings.chunking, model.count_tokens)
    )
    assert pipeline.chunks() == []
    assert pipeline.skipped == [str(missing)]


def test_source_factory_rejects_unknown_kinds(tmp_path):
    catalog = SourceCatalog((SourceSpec("Unknown", "banana", tmp_path / "x"),))
    with pytest.raises(ValueError, match="Unknown source kind"):
        SourceFactory.create(catalog)
