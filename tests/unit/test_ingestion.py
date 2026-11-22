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
            "Alexander Geiger\nLead Applied AI Scientist\n"
            "PROFESSIONAL SUMMARY\nLead Applied AI Scientist with 9 years in production ML.\n"
            "PROFESSIONAL EXPERIENCE\n"
            "Signature Aviation | Orlando, FL May 2022 to Present\n"
            "Lead Applied AI Scientist\n"
            "Promoted to Lead Applied AI Scientist in July 2026.\n"
            "\u2022 Built a platform powering:\n"
            "\u2022 Demand forecasting across bases.\n"
            "\u2022 Dynamic pricing for hourly parking.\n"
            "\u2022 Built the MLOps framework.\n"
            "GHD | Buffalo, NY\n"
            "Senior Data Scientist II Jul 2020 to May 2022\n"
            "\u2022 Built physical AI for pipelines.\n"
            "Data Scientist (earlier role) Aug 2018 to Jun 2019\n"
            "Earlier Experience\n"
            "Golden Oak Research | Founder | Current\n"
            "\u2022 Wrote research notes.\n"
            "EDUCATION\n"
            "Syracuse University | Master of Science in Applied Data Science 2019 to 2022\n"
            "CERTIFICATIONS\n"
            "AWS Certified Solutions Architect - Associate Issued Apr 2025, expires Apr 2028\n"
            "COMPETITIONS & ACHIEVEMENTS\n"
            "Kaggle | Competitions Expert Current\n"
            "NFL Big Data Bowl (Kaggle) Spring 2020\n"
        ),
    )
    sections = source.sections()
    titles = [item.title for item in sections]
    assert titles[0] == "Name and current title"
    assert titles[1] == "Professional summary"
    assert sections[1].header == "Professional summary"
    assert titles[2] == ("Professional experience > Signature Aviation, Orlando, FL > Lead Applied AI Scientist")
    assert titles[3] == "Professional experience > GHD, Buffalo, NY > Senior Data Scientist II"
    assert titles[4] == ("Professional experience > GHD, Buffalo, NY > Data Scientist (earlier role)")
    assert "Earlier Experience" in titles[5] and "Golden Oak Research" in titles[5]
    assert "Syracuse University" in titles[6]
    assert "AWS Certified Solutions Architect" in titles[7]
    assert titles[8].endswith("Kaggle > Competitions Expert")
    assert titles[9].endswith("NFL Big Data Bowl (Kaggle)")
    assert sections[2].header == ("Signature Aviation, Orlando, FL. Lead Applied AI Scientist. May 2022 to Present")
    assert sections[2].paragraphs[0] == "Promoted to Lead Applied AI Scientist in July 2026."
    assert "Demand forecasting across bases." in sections[2].paragraphs[1]
    assert "Dynamic pricing for hourly parking." in sections[2].paragraphs[1]
    assert sections[2].paragraphs[1].startswith("Built a platform powering:")
    assert sections[2].paragraphs[2] == "Built the MLOps framework."
    assert sections[3].header == "GHD, Buffalo, NY. Senior Data Scientist II. Jul 2020 to May 2022"
    assert sections[3].paragraphs == ("Built physical AI for pipelines.",)
    assert "Aug 2018 to Jun 2019" in sections[4].paragraphs[0]
    assert "Data Scientist (earlier role)" in sections[4].paragraphs[0]


def test_chunker_prefixes_role_header_and_keeps_paragraphs(model):
    chunker = HeadingAwareChunker(ChunkSettings(40, 8, min_chunk_tokens=1), model.count_tokens)
    header = "GHD, Buffalo, NY. Senior Data Scientist II. Jul 2020 to May 2022"
    short = "Built physical AI that detects pressure anomalies in pipelines."
    other = "A later idea about sensors and SCADA telemetry."
    chunks = chunker.chunk([Section("Resume", "Professional experience > GHD", (short, other), header=header)])
    assert len(chunks) == 1
    assert chunks[0].text.startswith(header)
    assert short in chunks[0].text and other in chunks[0].text


def test_chunker_keeps_a_fitting_paragraph_together(model):
    chunker = HeadingAwareChunker(ChunkSettings(40, 4, min_chunk_tokens=1), model.count_tokens)
    paragraph = "Built the MLOps framework for how models earn their way into production."
    other = "Made trust a production requirement by monitoring data quality and job runs."
    chunks = chunker.chunk([Section("Resume", "Signature Aviation", (paragraph, other))])
    assert any(paragraph in item.text and other in item.text for item in chunks)


def test_pipeline_keeps_a_cleaned_role_header(tmp_path, settings, model):
    path = tmp_path / "resume.md"
    path.write_text("# Role\n\nBuilt physical AI for oil and gas pipelines.\n", encoding="utf-8")
    # Markdown has no header; this checks that an empty header stays empty after cleaning.
    catalog = SourceCatalog((SourceSpec("Resume", "markdown", path),))
    pipeline = IngestionPipeline(
        SourceFactory.create(catalog), catalog, HeadingAwareChunker(settings.chunking, model.count_tokens)
    )
    sections = pipeline.sections()
    assert sections
    assert sections[0].header == ""


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
    chunker = HeadingAwareChunker(ChunkSettings(16, 6, min_chunk_tokens=1), model.count_tokens)
    # Three units: the first two fill chunk one, and the short trailing unit overlaps.
    section = Section(
        "Resume",
        "Experience",
        ("One two three four five six.", "Short tail here.", "Seven eight nine ten eleven twelve thirteen fourteen."),
    )
    chunks = chunker.chunk([section])
    assert len(chunks) == 2
    assert "One two three four five six." in chunks[0].text
    assert "Short tail here." in chunks[0].text
    assert chunks[1].text.startswith("Short tail here.")
    assert "thirteen fourteen." in chunks[1].text


def test_chunker_keeps_colon_lead_in_when_a_list_is_split(model):
    chunker = HeadingAwareChunker(ChunkSettings(80, 0, min_chunk_tokens=1), model.count_tokens)
    paragraph = (
        "Built a platform powering: Demand forecasting across bases and versions. "
        "Dynamic pricing for hourly parking across locations."
    )
    chunks = chunker.chunk(
        [Section("Resume", "Signature Aviation", (paragraph,), header="Signature Aviation, Orlando, FL")]
    )
    assert len(chunks) == 2
    assert all("Built a platform powering:" in item.text for item in chunks)
    assert any("Demand forecasting" in item.text for item in chunks)
    assert any("Dynamic pricing" in item.text for item in chunks)
    assert not all("Demand forecasting" in item.text and "Dynamic pricing" in item.text for item in chunks)


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


def test_pipeline_preserves_role_headers(tmp_path, monkeypatch, settings, model):
    path = tmp_path / "resume.pdf"
    path.write_bytes(b"%PDF-1.4 test")
    source = PdfSource(path, "Resume")
    monkeypatch.setattr(
        source,
        "text",
        lambda: (
            "PROFESSIONAL EXPERIENCE\n"
            "GHD | Buffalo, NY\n"
            "Senior Data Scientist II Jul 2020 to May 2022\n"
            "\u2022 Built physical AI that detects pressure anomalies in pipelines.\n"
        ),
    )
    catalog = SourceCatalog((SourceSpec("Resume", "pdf", tmp_path / "resume.pdf"),))
    pipeline = IngestionPipeline([source], catalog, HeadingAwareChunker(settings.chunking, model.count_tokens))
    sections = pipeline.sections()
    assert sections[0].header.startswith("GHD, Buffalo, NY. Senior Data Scientist II")
    chunks = pipeline.chunks()
    assert chunks
    assert all(item.text.startswith(sections[0].header) for item in chunks)


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
