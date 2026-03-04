"""
Extract keypoints/keywords from the Breakout Group Discussion section of a report DOCX.
Output is structured for story generation: themes, questions, topics, and keypoints.
"""
from dataclasses import dataclass, field
from pathlib import Path
import re

from langchain_community.document_loaders import Docx2txtLoader


SECTION_MARKERS = (
    "breakout group discussion",
    "b. breakout group discussion",
)
THEME_PATTERN = re.compile(r"^Theme\s*#\s*(\d+)\s*(.+)$", re.IGNORECASE)
QUESTION_PATTERN = re.compile(r"^Question\s+asked\s*:\s*(.+)$", re.IGNORECASE)
# Topic line: "1. Topic Name (N)" or "1. Topic Name" or "Number of Mentions: N"
TOPIC_HEAD_PATTERN = re.compile(r"^(\d+)\.\s+(.+?)(?:\s*\((\d+)\))?\s*$")
MENTIONS_PATTERN = re.compile(r"(?:Number of mentions?|Mentions?)\s*:\s*(\d+)", re.IGNORECASE)
BULLET_PATTERN = re.compile(r"^[\s•\-*]+\s*(.+)$")


@dataclass
class TopicKeypoints:
    """One topic row: name, keypoint phrases, and mention count."""

    name: str
    keypoints: list[str] = field(default_factory=list)
    mentions: int | None = None


@dataclass
class ThemeBlock:
    """One theme: title, question, list of topics with keypoints, optional participant names and story paragraphs from Stories section."""

    theme_number: int
    title: str
    question: str
    topics: list[TopicKeypoints] = field(default_factory=list)
    participant_names: list[str] = field(default_factory=list)
    story_paragraphs: list[str] = field(default_factory=list)


@dataclass
class BreakoutExtract:
    """Full extraction result for the Breakout Group Discussion section."""

    themes: list[ThemeBlock] = field(default_factory=list)
    raw_section: str = ""

    def to_dict(self) -> dict:
        """Serialize for JSON or story-generation input."""
        return {
            "themes": [
                {
                    "theme_number": t.theme_number,
                    "title": t.title,
                    "question": t.question,
                    "topics": [
                        {
                            "name": p.name,
                            "keypoints": p.keypoints,
                            "mentions": p.mentions,
                        }
                        for p in t.topics
                    ],
                    "participant_names": t.participant_names,
                    "story_paragraphs": t.story_paragraphs,
                }
                for t in self.themes
            ],
        }


def _load_docx_text(path: str | Path) -> str:
    """Load a DOCX file and return its full text (no chunking)."""
    loader = Docx2txtLoader(str(path))
    docs = loader.load()
    if not docs:
        return ""
    return docs[0].page_content if hasattr(docs[0], "page_content") else str(docs[0])


def _extract_story_paragraphs_from_theme_block(block_text: str) -> list[str]:
    """
    Parse the Stories subsection: number, story paragraph, name line, location line, repeated.
    Return list of story paragraph texts in order.
    """
    idx = block_text.lower().find("stories:")
    if idx == -1:
        return []
    after_stories = block_text[idx + len("stories:") :].strip()
    if not after_stories:
        return []
    segments = [s.strip() for s in re.split(r"\n\s*\n", after_stories) if s.strip()]
    paragraphs: list[str] = []
    for k in range(len(segments) // 4):
        i = 1 + 4 * k
        if i < len(segments):
            story = segments[i]
            if story and not re.match(r"^\d+$", story) and len(story) > 20:
                paragraphs.append(story)
    return paragraphs


def _extract_story_names_from_theme_block(block_text: str) -> list[str]:
    """
    Parse the Stories subsection: number, story paragraph, name line, location line, repeated.
    Return list of participant name lines in order (one per story).
    """
    idx = block_text.lower().find("stories:")
    if idx == -1:
        return []
    after_stories = block_text[idx + len("stories:") :].strip()
    if not after_stories:
        return []
    segments = [s.strip() for s in re.split(r"\n\s*\n", after_stories) if s.strip()]
    names: list[str] = []
    for k in range(len(segments) // 4):
        i = 2 + 4 * k
        if i < len(segments):
            candidate = segments[i]
            if candidate and not re.match(r"^\d+$", candidate) and len(candidate) <= 120:
                names.append(candidate)
    return names


def _find_breakout_section(full_text: str) -> str:
    """Return the substring that contains the Breakout Group Discussion section."""
    text_lower = full_text.lower()
    start = -1
    for marker in SECTION_MARKERS:
        idx = text_lower.find(marker)
        if idx != -1:
            start = idx
            break
    if start == -1:
        return ""
    # Optionally stop at next top-level section (e.g. "C. " or "Appendix" or next "Theme #" after a blank block)
    section = full_text[start:]
    return section


def _parse_breakout_section(section_text: str) -> BreakoutExtract:
    """Parse the Breakout Group Discussion section into themes and topics with keypoints."""
    result = BreakoutExtract(raw_section=section_text)
    lines = section_text.split("\n")
    current_theme: ThemeBlock | None = None
    current_topic: TopicKeypoints | None = None
    i = 0

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Theme #N Title
        m_theme = THEME_PATTERN.match(stripped)
        if m_theme:
            if current_theme and current_topic:
                current_theme.topics.append(current_topic)
            if current_theme:
                result.themes.append(current_theme)
            current_theme = ThemeBlock(
                theme_number=int(m_theme.group(1)),
                title=m_theme.group(2).strip(),
                question="",
            )
            current_topic = None
            i += 1
            continue

        # Question asked: ...
        m_q = QUESTION_PATTERN.match(stripped)
        if m_q and current_theme is not None:
            current_theme.question = m_q.group(1).strip()
            i += 1
            continue

        # Numbered topic: "1. Safety and Security (10)" or "1. Topic Name"
        m_topic = TOPIC_HEAD_PATTERN.match(stripped)
        if m_topic and current_theme is not None:
            if current_topic is not None:
                current_theme.topics.append(current_topic)
            mentions = int(m_topic.group(3)) if m_topic.group(3) else None
            current_topic = TopicKeypoints(
                name=m_topic.group(2).strip(),
                keypoints=[],
                mentions=mentions,
            )
            i += 1
            continue

        # "Number of Mentions: N" on its own line
        m_mentions = MENTIONS_PATTERN.search(stripped)
        if m_mentions and current_topic is not None:
            current_topic.mentions = current_topic.mentions or int(m_mentions.group(1))
            i += 1
            continue

        # Bullet keypoint under current topic
        m_bullet = BULLET_PATTERN.match(stripped)
        if m_bullet and current_topic is not None and stripped:
            current_topic.keypoints.append(m_bullet.group(1).strip())
            i += 1
            continue

        # Continuation of topic name (no number): sometimes the topic name wraps or keypoints are on next lines without bullets
        if current_topic is not None and stripped and not re.match(r"^\d+\.", stripped):
            # If it looks like a short phrase, treat as keypoint; else skip
            if len(stripped) < 120 and stripped.lower() not in ("stories", "story", "theme"):
                current_topic.keypoints.append(stripped)
        i += 1

    if current_theme is not None:
        if current_topic is not None:
            current_theme.topics.append(current_topic)
        result.themes.append(current_theme)

    # Second pass: parse participant names and story paragraphs from each theme's Stories subsection
    theme_blocks = re.split(r"\n\s*Theme\s*#\s*\d+", section_text, flags=re.IGNORECASE)
    for i, theme in enumerate(result.themes):
        block_index = i + 1
        if block_index < len(theme_blocks):
            block = theme_blocks[block_index]
            theme.participant_names = _extract_story_names_from_theme_block(block)
            theme.story_paragraphs = _extract_story_paragraphs_from_theme_block(block)

    return result


def extract_breakout_keypoints(docx_path: str | Path) -> BreakoutExtract:
    """
    Load a report DOCX and extract keypoints/keywords from the Breakout Group Discussion section.

    Args:
        docx_path: Path to the .docx file (e.g. CSS Report_2025_10_01_Grp 1 Lyon.docx).

    Returns:
        BreakoutExtract with themes, each containing topics and keypoints for story generation.
    """
    path = Path(docx_path)
    if path.suffix.lower() != ".docx":
        raise ValueError(f"Expected a .docx file, got {path.suffix}")
    full_text = _load_docx_text(path)
    section = _find_breakout_section(full_text)
    return _parse_breakout_section(section)


if __name__ == "__main__":
    import json
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.modules.questionnaire.breakout_extract <path/to/report.docx> [--json]", file=sys.stderr)
        sys.exit(1)
    docx_path = sys.argv[1]
    out_json = "--json" in sys.argv
    try:
        result = extract_breakout_keypoints(docx_path)
        if out_json:
            print(json.dumps(result.to_dict(), indent=2))
        else:
            for t in result.themes:
                print(f"Theme #{t.theme_number}: {t.title}")
                print(f"  Question: {t.question}")
                for p in t.topics:
                    mentions = f" (mentions: {p.mentions})" if p.mentions is not None else ""
                    print(f"  - {p.name}{mentions}")
                    for k in p.keypoints:
                        print(f"      • {k}")
                print()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
