#!/usr/bin/env python3
"""
ingest.py
---------
Walks a folder of source documents (PDF/DOCX/HTML), parses + chunks each
one, and writes the results into the SQLite store. Re-running is safe:
documents are deduplicated by content hash, and re-ingesting an updated
file replaces its chunks.

Usage:
    python scripts/ingest.py --input data/raw --domain enterprise_kb
    python scripts/ingest.py --input data/raw/policy.pdf --domain enterprise_kb
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion.chunker import chunk_sections
from src.ingestion.parsers import PARSER_BY_EXTENSION, parse_file
from src.storage import Document, Store, compute_content_hash, make_doc_id


def iter_source_files(input_path: Path):
    if input_path.is_file():
        yield input_path
        return
    for ext in PARSER_BY_EXTENSION:
        yield from input_path.rglob(f"*{ext}")


def ingest_file(filepath: Path, store: Store, domain: str | None) -> dict:
    raw_bytes = filepath.read_bytes()
    content_hash = compute_content_hash(raw_bytes)
    doc_id = make_doc_id(filepath.name, content_hash)
    filetype = filepath.suffix.lower().lstrip(".")

    doc = Document(
        doc_id=doc_id, filename=filepath.name, filepath=str(filepath),
        filetype=filetype, content_hash=content_hash, domain=domain,
    )
    is_new = store.upsert_document(doc)
    if not is_new:
        return {"file": filepath.name, "status": "skipped (duplicate content)", "chunks": 0}

    sections = parse_file(filepath)
    if not sections:
        return {"file": filepath.name, "status": "warning: no extractable text", "chunks": 0}

    chunks = chunk_sections(sections, doc_id=doc_id)
    store.insert_chunks(chunks)
    return {"file": filepath.name, "status": "ingested", "chunks": len(chunks)}


def main():
    parser = argparse.ArgumentParser(description="Ingest documents into the Ask My Docs store.")
    parser.add_argument("--input", required=True, help="File or directory of PDF/DOCX/HTML files")
    parser.add_argument("--domain", default=None, help="Domain label, e.g. 'hr_policies'")
    parser.add_argument("--db", default=None, help="Override path to SQLite db")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: {input_path} does not exist", file=sys.stderr)
        sys.exit(1)

    store = Store(db_path=Path(args.db)) if args.db else Store()

    results = []
    for filepath in iter_source_files(input_path):
        try:
            results.append(ingest_file(filepath, store, args.domain))
        except Exception as e:
            results.append({"file": filepath.name, "status": f"ERROR: {e}", "chunks": 0})

    print(f"\n{'File':<35} {'Status':<30} {'Chunks':>7}")
    print("-" * 74)
    for r in results:
        print(f"{r['file']:<35} {r['status']:<30} {r['chunks']:>7}")

    stats = store.stats()
    print(f"\nStore totals: {stats['documents']} documents, {stats['chunks']} chunks")
    print(f"DB location: {store.db_path}")


if __name__ == "__main__":
    main()
