"""
Test story generation from Breakout Group Discussion keypoints using the same
training data as the app (input/questionnaire/). Run from project root with venv active.

Uses the LLM from .env (LLM_MODEL_PATH), e.g. local Qwen GGUF; no OpenAI key needed for story generation.
OPENAI_API_KEY is only required if you want to load training data (embeddings) from input/questionnaire/.

Usage:
  python scripts/test_breakout_stories.py "input/questionnaire/CSS Report_2025_10_01_Grp 1 Lyon.docx"
  python scripts/test_breakout_stories.py path/to/report.docx --max 2
"""
import asyncio
import os
import sys
from pathlib import Path

# Reduce Hugging Face hub noise on Windows (symlinks warning)
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# Project root and load .env first (same as main.py)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_env() -> None:
    """Load .env from project root so EMBEDDING_MODEL and LLM_MODEL_PATH are set."""
    try:
        from dotenv import load_dotenv
        load_dotenv(_PROJECT_ROOT / ".env", override=True)
    except ImportError:
        pass


_load_env()

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.app import QUESTIONNAIRE_INPUT_DIR
from src.knowledge_base import KnowledgeBase
from src.modules.questionnaire import extract_breakout_keypoints, generate_stories_from_breakout


def _resolve_llm_path_from_env() -> str | None:
    """Read LLM_MODEL_PATH from env and resolve relative paths against project root."""
    raw = os.environ.get("LLM_MODEL_PATH")
    if not raw or not raw.strip():
        return None
    raw = raw.strip()
    if raw.lower() in ("openai", ""):
        return None
    if raw.startswith("hf:"):
        return raw
    # Resolve relative paths against project root so the path is valid regardless of cwd
    p = Path(raw)
    if not p.is_absolute():
        p = (_PROJECT_ROOT / raw).resolve()
    return str(p)


def _main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python scripts/test_breakout_stories.py <report.docx> [--max N]", file=sys.stderr)
        sys.exit(1)
    docx_path = Path(sys.argv[1])
    max_stories = None
    if "--max" in sys.argv:
        idx = sys.argv.index("--max")
        if idx + 1 < len(sys.argv):
            try:
                max_stories = int(sys.argv[idx + 1])
            except ValueError:
                pass
    if not docx_path.is_file():
        print(f"File not found: {docx_path}", file=sys.stderr)
        sys.exit(1)
    try:
        extract = extract_breakout_keypoints(docx_path)
    except Exception as e:
        print(f"Extract failed: {e}", file=sys.stderr)
        sys.exit(1)
    if not extract.themes:
        print("No themes found in Breakout Group Discussion section.", file=sys.stderr)
        sys.exit(1)
    _load_env()
    llm_model_path = _resolve_llm_path_from_env()
    using_local_embeddings = (os.environ.get("EMBEDDING_MODEL") or "").strip().lower() not in ("", "openai")
    kb = KnowledgeBase()
    try:
        kb.init_from_directory(QUESTIONNAIRE_INPUT_DIR)
    except Exception as e:
        err_lower = str(e).lower()
        if not using_local_embeddings and not os.environ.get("OPENAI_API_KEY") and ("api_key" in err_lower or "openai" in err_lower):
            print(
                "Note: Training data not loaded (OPENAI_API_KEY required for OpenAI embeddings). Set EMBEDDING_MODEL=local in .env to use local embeddings.",
                file=sys.stderr,
            )
        else:
            print(f"Note: Training data not loaded: {e}. Stories will use keypoints only.", file=sys.stderr)
        kb = None
    async def run() -> list:
        return await generate_stories_from_breakout(
            extract, kb=kb, domain="", llm_model_path=llm_model_path, max_stories=max_stories
        )
    try:
        stories = asyncio.run(run())
    except Exception as e:
        err = str(e).lower()
        if "llama-cpp-python is not installed" in err or "llama_cpp" in err:
            print(
                str(e),
                "\nOn Windows, if 'pip install -r requirements-llm.txt' fails to build, try Python 3.11/3.12 for a pre-built wheel, or install Visual Studio Build Tools (Desktop development with C++).",
                file=sys.stderr,
            )
        elif "api_key" in err or "openai_api_key" in err:
            llm_env = os.environ.get("LLM_MODEL_PATH", "")
            resolved = llm_model_path if llm_model_path else "(none)"
            print(
                f"LLM fallback to OpenAI failed (no API key). LLM_MODEL_PATH in env: {llm_env!r} → resolved: {resolved}. "
                "Set LLM_MODEL_PATH in .env to a local GGUF path (e.g. ./models/Qwen2.5-0.5B-Instruct-GGUF.gguf) or hf:Qwen/Qwen2.5-0.5B-Instruct-GGUF. Ensure the file exists or install: pip install -r requirements-llm.txt",
                file=sys.stderr,
            )
        else:
            print(f"Story generation failed: {e}", file=sys.stderr)
        sys.exit(1)
    for s in stories:
        print(f"--- Theme #{s.theme_number}: {s.theme_title} | {s.topic_name} ---")
        print(s.story_text)
        print()
    print(f"Generated {len(stories)} story(ies).")


if __name__ == "__main__":
    _main()
