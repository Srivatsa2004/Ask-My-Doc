"""
parsers.py
----------
Format-specific parsers for PDF, DOCX, and HTML. Each parser walks its
source file and yields a flat list of `Section` objects: contiguous runs
of text that share a page (PDF) and/or heading (all formats).

The chunker (chunker.py) doesn't know or care which format produced these
sections -- this module is the only place format-specific logic lives.
Keeping that boundary clean means adding a 4th format later (e.g. .txt,
.md, Confluence export) only touches this file.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pdfplumber
from bs4 import BeautifulSoup
from docx import Document as DocxDocument


@dataclass
class Section:
    """A contiguous span of text with the metadata needed to cite it later."""
    text: str
    heading: Optional[str] = None   # nearest heading above this text, e.g. "3.2 Refund Policy"
    page: Optional[int] = None      # 1-indexed page number (PDF only; None for docx/html)


HEADING_STYLE_PREFIXES = ("Heading", "Title")  # python-docx style names for headings


def parse_pdf(filepath: Path) -> list[Section]:
    """
    Extract text page-by-page. Within each page, a simple heuristic treats
    short, bold/large lines as headings (pdfplumber exposes font size per
    char, which we use to guess heading-ness without an ML layout model --
    good enough for the structure-aware chunking step downstream).
    """
    sections: list[Section] = []
    current_heading: Optional[str] = None

    with pdfplumber.open(str(filepath)) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            words = page.extract_words(extra_attrs=["size"])
            if not words:
                continue

            # group words into lines by their vertical position ("top")
            lines: dict[float, list] = {}
            for w in words:
                key = round(w["top"], 1)
                lines.setdefault(key, []).append(w)

            body_sizes = [w["size"] for w in words]
            median_size = sorted(body_sizes)[len(body_sizes) // 2]

            page_text_parts = []
            for top in sorted(lines.keys()):
                line_words = sorted(lines[top], key=lambda w: w["x0"])
                line_text = " ".join(w["text"] for w in line_words)
                avg_size = sum(w["size"] for w in line_words) / len(line_words)

                is_heading = (
                    avg_size > median_size * 1.15
                    and len(line_text) < 100
                    and not line_text.endswith(".")
                )

                if is_heading:
                    # flush accumulated body text as a section before starting a new heading
                    if page_text_parts:
                        sections.append(Section(
                            text="\n".join(page_text_parts), heading=current_heading, page=page_num,
                        ))
                        page_text_parts = []
                    current_heading = line_text
                else:
                    page_text_parts.append(line_text)

            if page_text_parts:
                sections.append(Section(
                    text="\n".join(page_text_parts), heading=current_heading, page=page_num,
                ))

    return [s for s in sections if s.text.strip()]


def parse_docx(filepath: Path) -> list[Section]:
    """
    Walk paragraphs in order. Paragraphs styled "Heading N" or "Title"
    become the current heading context; everything else is accumulated
    as body text under that heading until the next heading appears.
    """
    doc = DocxDocument(str(filepath))
    sections: list[Section] = []
    current_heading: Optional[str] = None
    buffer: list[str] = []

    def flush():
        if buffer:
            sections.append(Section(text="\n".join(buffer), heading=current_heading, page=None))
            buffer.clear()

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        style_name = (para.style.name if para.style else "") or ""
        if style_name.startswith(HEADING_STYLE_PREFIXES):
            flush()
            current_heading = text
        else:
            buffer.append(text)

    # also pull table content (common in enterprise docs: policy tables, spec sheets)
    for table in doc.tables:
        rows_text = []
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells]
            if any(cells):
                rows_text.append(" | ".join(cells))
        if rows_text:
            flush()
            sections.append(Section(text="\n".join(rows_text), heading=current_heading, page=None))

    flush()
    return [s for s in sections if s.text.strip()]


def parse_html(filepath: Path) -> list[Section]:
    """
    Strip boilerplate (nav/footer/script/style), then walk the DOM in
    document order treating h1-h6 as heading context, similar to the
    DOCX parser.
    """
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f.read(), "lxml")

    for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        tag.decompose()

    body = soup.body or soup
    sections: list[Section] = []
    current_heading: Optional[str] = None
    buffer: list[str] = []

    def flush():
        if buffer:
            sections.append(Section(text="\n".join(buffer), heading=current_heading, page=None))
            buffer.clear()

    HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
    BODY_TAGS = {"p", "li", "td", "th", "blockquote", "pre"}

    for el in body.find_all(list(HEADING_TAGS | BODY_TAGS)):
        text = el.get_text(strip=True, separator=" ")
        if not text:
            continue
        if el.name in HEADING_TAGS:
            flush()
            current_heading = text
        else:
            buffer.append(text)

    flush()
    return [s for s in sections if s.text.strip()]


PARSER_BY_EXTENSION = {
    ".pdf": parse_pdf,
    ".docx": parse_docx,
    ".html": parse_html,
    ".htm": parse_html,
}


def parse_file(filepath: Path) -> list[Section]:
    ext = filepath.suffix.lower()
    if ext not in PARSER_BY_EXTENSION:
        raise ValueError(f"Unsupported file type: {ext} (supported: {list(PARSER_BY_EXTENSION)})")
    return PARSER_BY_EXTENSION[ext](filepath)
