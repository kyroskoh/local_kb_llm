"""
Empty the vectordb (Chroma) and train pattern data from input DOCXs only.
No pretrained data is used. Patterns (theme + keypoints + example story style)
from the DOCXs are indexed so the system can learn how to summarize themes and
keypoints into stories. Use this to start from scratch and build pattern
knowledge from your report DOCXs only.

Run from project root with venv active. Requires .env for embeddings unless
EMBEDDING_MODEL=local.

Usage:
  python scripts/train_story_patterns.py
  python scripts/train_story_patterns.py --input-dir input/questionnaire
  python scripts/train_story_patterns.py --dry-run
"""
import logging
import os
import re
import sys
import warnings
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
    stream=sys.stderr,
)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_env() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv(_PROJECT_ROOT / ".env", override=True)
    except ImportError:
        pass


_load_env()

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import chromadb
from langchain_core.documents import Document

from src.knowledge_base import KnowledgeBase
from src.modules.questionnaire.breakout_extract import extract_breakout_keypoints

warnings.filterwarnings("ignore", message=".*unauthenticated requests.*HF Hub.*", category=UserWarning)

# Default: DOCXs in input/questionnaire (report DOCXs only; no other pretrained data)
DEFAULT_INPUT_DIR = _PROJECT_ROOT / "input" / "questionnaire"
DEFAULT_DB_DIR = _PROJECT_ROOT / "db"
STORY_PATTERNS_DOMAIN = "story_patterns"


def _sanitize_story_for_pattern(text: str, participant_names: list[str]) -> str:
    """Replace real names with (Name) and strip location phrases so pattern has no PII."""
    if not text:
        return text
    for name in participant_names:
        if name and name.strip():
            text = re.sub(re.escape(name.strip()), "(Name)", text, flags=re.IGNORECASE)
    text = re.sub(r'[\"\' ]*\(Location(?:\s+of\s+Session)?(?:\s*:\s*[^)"\']*)?\)[\"\' ]*', " ", text, flags=re.IGNORECASE)
    text = re.sub(r'[\"\' ]*\(Location\s+of\s+Session\)\s*:\s*[^."\']+[\"\' ]*', " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _theme_to_pattern_doc(theme, source_path: str) -> Document:
    """Build one pattern document per theme: theme, question, topics/keypoints, example story paragraphs."""
    parts = [
        f"Theme: {theme.title}",
        f"Question: {theme.question}",
        "Topics and keypoints:",
    ]
    for t in theme.topics:
        parts.append(f"- {t.name}")
        for k in t.keypoints:
            parts.append(f"  • {k}")
    if theme.story_paragraphs:
        parts.append("---")
        parts.append("Example story patterns (third person, theme/keypoint-based):")
        for p in theme.story_paragraphs:
            sanitized = _sanitize_story_for_pattern(p, theme.participant_names)
            if sanitized:
                parts.append(sanitized)
    content = "\n".join(parts)
    return Document(
        page_content=content,
        metadata={
            "source": source_path,
            "theme_number": theme.theme_number,
            "theme_title": theme.title,
        },
    )


def empty_vectordb(db_path: Path) -> int:
    """Delete all Chroma collections. Returns number of collections deleted."""
    client = chromadb.PersistentClient(
        path=str(db_path),
        settings=chromadb.Settings(anonymized_telemetry=False),
    )
    collections = list(client.list_collections())
    for c in collections:
        client.delete_collection(c.name)
    return len(collections)


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(
        description="Empty vectordb and train story patterns from input DOCXs only (no pretrained data).",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Directory containing report .docx files (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB_DIR,
        help=f"Chroma persist directory (default: {DEFAULT_DB_DIR})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only list DOCXs and pattern count; do not empty or write to vectordb.",
    )
    args = parser.parse_args()

    input_dir = args.input_dir if args.input_dir.is_absolute() else _PROJECT_ROOT / args.input_dir
    db_path = args.db if args.db.is_absolute() else _PROJECT_ROOT / args.db

    if not input_dir.is_dir():
        logging.error("Input directory not found: %s", input_dir)
        return 1

    docx_files = sorted(input_dir.glob("*.docx"))
    if not docx_files:
        logging.warning("No .docx files in %s", input_dir)
        if not args.dry_run:
            deleted = empty_vectordb(db_path)
            logging.info("Emptied vectordb (%d collection(s) deleted). No pattern docs to add.", deleted)
        return 0

    pattern_docs: list[Document] = []
    for path in docx_files:
        try:
            extract = extract_breakout_keypoints(path)
            for theme in extract.themes:
                pattern_docs.append(_theme_to_pattern_doc(theme, str(path)))
        except Exception as e:
            logging.warning("Skipping %s: %s", path.name, e)

    logging.info("Found %d DOCX file(s), %d pattern document(s).", len(docx_files), len(pattern_docs))

    if args.dry_run:
        for d in pattern_docs[:3]:
            logging.info("Sample metadata: %s", d.metadata)
        return 0

    deleted = empty_vectordb(db_path)
    logging.info("Emptied vectordb (%d collection(s) deleted).", deleted)

    if not pattern_docs:
        logging.info("No pattern documents to index.")
        return 0

    kb = KnowledgeBase(persist_directory=str(db_path))
    kb.init_from_documents(pattern_docs, domain=STORY_PATTERNS_DOMAIN)
    logging.info("Indexed %d pattern document(s) into domain %r.", len(pattern_docs), STORY_PATTERNS_DOMAIN)
    return 0


if __name__ == "__main__":
    sys.exit(main())
