"""
Generate one story per theme from Breakout Group Discussion (aggregates all topics
under each theme). Same training data and LLM as test_breakout_stories.py.
Run from project root with venv active.

Uses the LLM from .env (LLM_MODEL_PATH). OPENAI_API_KEY is required for
loading training data (embeddings) unless EMBEDDING_MODEL=local.

Usage:
  python scripts/test_theme_stories.py "input/questionnaire/CSS Report_2025_10_01_Grp 1 Lyon.docx"
  python scripts/test_theme_stories.py path/to/report.docx --max 2
  python scripts/test_theme_stories.py path/to/report.docx --theme 1
  python scripts/test_theme_stories.py path/to/report.docx --max-words 60
  python scripts/test_theme_stories.py path/to/report.docx --write-prompts   # write each theme prompt to output/questionnaire/prompts/
"""
import asyncio
import logging
import os
import sys
import warnings
from pathlib import Path

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s: %(name)s: %(message)s",
    stream=sys.stderr,
)

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
warnings.filterwarnings("ignore", message=".*unauthenticated requests.*HF Hub.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*local_dir_use_symlinks.*deprecated.*", category=UserWarning)

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

from src.knowledge_base import DEFAULT_INPUT_DIR, KnowledgeBase
from src.modules.questionnaire.breakout_extract import BreakoutExtract, ThemeBlock, TopicKeypoints
from src.modules.questionnaire import (
    INPUT_SUBDIR,
    extract_breakout_keypoints,
    generate_stories_from_breakout,
    write_stories_to_file,
)

QUESTIONNAIRE_INPUT_DIR = DEFAULT_INPUT_DIR / INPUT_SUBDIR


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
    p = Path(raw)
    if not p.is_absolute():
        p = (_PROJECT_ROOT / raw).resolve()
    return str(p)


def _extract_to_one_story_per_theme(extract: BreakoutExtract) -> BreakoutExtract:
    """Convert extract so each theme has one topic (aggregated keypoints) for one story per theme."""
    themes_one = []
    for t in extract.themes:
        keypoints: list[str] = [t.question] if t.question else []
        for top in t.topics:
            keypoints.append(top.name)
            keypoints.extend(top.keypoints)
        topic = TopicKeypoints(name=t.title, keypoints=keypoints, mentions=None)
        themes_one.append(
            ThemeBlock(
                theme_number=t.theme_number,
                title=t.title,
                question=t.question,
                topics=[topic],
                participant_names=t.participant_names,
                story_paragraphs=t.story_paragraphs,
            )
        )
    return BreakoutExtract(themes=themes_one)


def _main() -> None:
    if len(sys.argv) < 2:
        print(
            "Usage: python scripts/test_theme_stories.py <report.docx> [--max N] [--theme N]",
            file=sys.stderr,
        )
        sys.exit(1)
    docx_path = Path(sys.argv[1])
    max_stories = None
    theme_only = None
    max_words = 130
    write_prompts_to = None
    if "--write-prompts" in sys.argv:
        write_prompts_to = _PROJECT_ROOT / "output" / "questionnaire" / "prompts" / docx_path.stem
    if "--max" in sys.argv:
        idx = sys.argv.index("--max")
        if idx + 1 < len(sys.argv):
            try:
                max_stories = int(sys.argv[idx + 1])
            except ValueError:
                pass
    if "--max-words" in sys.argv:
        idx = sys.argv.index("--max-words")
        if idx + 1 < len(sys.argv):
            try:
                max_words = int(sys.argv[idx + 1])
            except ValueError:
                pass
    if "--theme" in sys.argv:
        idx = sys.argv.index("--theme")
        if idx + 1 < len(sys.argv):
            try:
                theme_only = int(sys.argv[idx + 1])
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
    if theme_only is not None:
        themes_subset = [t for t in extract.themes if t.theme_number == theme_only]
        if not themes_subset:
            print(
                f"No theme with number {theme_only}. Available: {[t.theme_number for t in extract.themes]}",
                file=sys.stderr,
            )
            sys.exit(1)
        extract = BreakoutExtract(themes=themes_subset)
    extract = _extract_to_one_story_per_theme(extract)
    _load_env()
    llm_model_path = _resolve_llm_path_from_env()
    using_local_embeddings = (os.environ.get("EMBEDDING_MODEL") or "").strip().lower() not in ("", "openai")
    kb = KnowledgeBase()
    try:
        kb.init_from_directory(QUESTIONNAIRE_INPUT_DIR)
        print("Using training data from input/questionnaire/ for story context.", file=sys.stderr)
    except Exception as e:
        err_lower = str(e).lower()
        if not using_local_embeddings and not os.environ.get("OPENAI_API_KEY") and (
            "api_key" in err_lower or "openai" in err_lower
        ):
            print(
                "Note: Training data not loaded (OPENAI_API_KEY required for OpenAI embeddings). "
                "Set EMBEDDING_MODEL=local in .env to use local embeddings.",
                file=sys.stderr,
            )
        else:
            print(f"Note: Training data not loaded: {e}. Stories will use keypoints only.", file=sys.stderr)
        kb = None
    async def run() -> list:
        return await generate_stories_from_breakout(
            extract,
            kb=kb,
            domain="",
            llm_model_path=llm_model_path,
            max_stories=max_stories,
            max_words=max_words,
            write_prompts_to=write_prompts_to,
        )
    try:
        stories = asyncio.run(run())
    except Exception as e:
        err = str(e).lower()
        if "llama-cpp-python is not installed" in err or "llama_cpp" in err:
            print(
                str(e),
                "\nOn Windows, if 'pip install -r requirements-llm.txt' fails to build, try Python 3.11/3.12 "
                "for a pre-built wheel, or install Visual Studio Build Tools (Desktop development with C++).",
                file=sys.stderr,
            )
        elif "api_key" in err or "openai_api_key" in err:
            llm_env = os.environ.get("LLM_MODEL_PATH", "")
            resolved = llm_model_path if llm_model_path else "(none)"
            print(
                f"LLM fallback to OpenAI failed (no API key). LLM_MODEL_PATH in env: {llm_env!r} → resolved: {resolved}. "
                "Set LLM_MODEL_PATH in .env to a local GGUF path or hf:Qwen/.... Ensure the file exists or "
                "install: pip install -r requirements-llm.txt",
                file=sys.stderr,
            )
        else:
            print(f"Story generation failed: {e}", file=sys.stderr)
        sys.exit(1)
    for s in stories:
        print(f"--- Theme #{s.theme_number}: {s.theme_title} | {s.topic_name} ---")
        print(s.story_text)
        print()
    out_path = write_stories_to_file(stories, docx_path, module_name="questionnaire")
    print(f"Generated {len(stories)} theme story(ies) (max {max_words} words each).")
    print(f"Output written to: {out_path}")
    if write_prompts_to is not None:
        print(f"Prompts written to: {write_prompts_to}")


if __name__ == "__main__":
    _main()
