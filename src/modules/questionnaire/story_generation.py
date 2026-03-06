"""
Generate short narrative stories from Breakout Group Discussion keypoints.
Uses the same LLM and optional KB (training data) as the questionnaire module.
"""
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from langchain_core.prompts import PromptTemplate

logger = logging.getLogger(__name__)

from ...knowledge_base import KnowledgeBase, _create_llm

from .breakout_extract import BreakoutExtract, ThemeBlock, TopicKeypoints


STORY_TEMPLATE = """Write one short narrative story for a report. Tailor the story to the theme statement and keypoints from this document only. Each document has its own themes and keypoint style—use the Theme, Question, Topic, and Keypoints provided below exactly as given; do not substitute content from other themes or from generic examples.

Theme: {theme_title}
Question asked: {question}
Topic (category): {topic_name}
Keypoints (participants' answers; mention count = how often raised—use for emphasis):
{keypoints}

Example story pattern(s) from this document (follow this style and structure when writing the new story):
{example_story_pattern}

{kb_context}

Critical: The theme statement above ("{theme_title}" / "{question}") and the keypoints are from this document. Your story must reflect that theme statement and only those keypoints. Base the story only on the Theme, Question, Topic, and Keypoints above. Do not use any other knowledge: no real names (except the placeholder "(Name)"), no place names or location in the story—do not include any location (no "(Location: ...)", no real places). Do not invent or copy names, locations, or people from anywhere else.

Output only the story. Do not output any explanation, meta-commentary, or phrases like "I am an AI/assistant". If the theme is implied by the keypoints, write the story—do not comment on whether it was addressed. Story structure:
- One continuous paragraph of prose only. No headings, bullet points, lists, or echoed topic/keypoint labels. Use the literal "(Name)" for the participant. {pronoun_instruction} The entire story must be in third person only. Do not use first person anywhere—no I, me, my, we, our, us. Even when reporting what the participant said, paraphrase in third person (e.g. "(Name) shared that being a Singaporean means..." or "(Name) hoped to see..."); never write "I think", "we want", "in my experience", or quote the participant saying "I" or "we".
- Length: summarize in at most {max_words} words. Be compact and precise; every sentence must add new information. Do not repeat any phrase, clause, or sentence.
- Opening: Start with "(Name)" and a phrase that directly reflects the theme statement and keypoints above (e.g. "(Name) shared that [from keypoints]" or "(Name), who [brief context], shared that [from keypoints]."). The opening must align with the theme and keypoints provided, not with a different or generic theme.
- After the opening: 1–3 more sentences with concrete details drawn only from the keypoints above. Follow the pronoun rule above for referring to (Name). Do not include any location in the story (no place names, no "(Location of Session)", no location phrases). Do not repeat the same or similar wording.
- Wrong: mixing pronouns for (Name); using I, me, my, we, our, us; including any location or place name; content that does not match the theme statement or keypoints above; listing keypoints verbatim; repeating words or sentences; adding real names or places from any source. Right: one paragraph, one consistent way to refer to (Name) as specified above, content from this document's theme and keypoints only, no location, no other names or places except "(Name)", no repetition.

Your reply must be exactly one paragraph of prose starting with "(Name)". No headings, no lists, no echoed keypoints, no meta-commentary—only the story text. Third person only: never use I, me, my, we, our, or us anywhere in the story. Stay within {max_words} words. Important: Write the full paragraph and end with a complete sentence. Do not stop mid-sentence or mid-thought. Do not write "I am an AI/assistant" or similar."""

# Minimum length for a valid story; shorter output is treated as failed (e.g. small models often emit EOS after 1–2 tokens).
MIN_STORY_LENGTH = 100


def _sanitize_story_output(text: str) -> str:
    """Remove model meta-output: 'I am an AI/assistant', meta-commentary, and literal placeholders."""
    if not text:
        return text
    # Remove literal placeholder if model echoed it
    text = text.replace("[from keypoints]", "").strip()
    # Strip any location from story: no location in output
    text = re.sub(r'[\"\' ]*\(Location(?:\s+of\s+Session)?(?:\s*:\s*[^)"\']*)?\)[\"\' ]*', " ", text, flags=re.IGNORECASE)
    text = re.sub(r'[\"\' ]*\(Location\s+of\s+Session\)\s*:\s*[^."\']+[\"\' ]*', " ", text, flags=re.IGNORECASE)
    text = re.sub(r'[\"\' ]*Location\s+of\s+Session\s*:\s*[^."\']+[\"\' ]*', " ", text, flags=re.IGNORECASE)
    # Remove orphaned "credited:" or "credited: "" left after stripping location
    text = re.sub(r'\s+credited\s*:\s*["\']?\s*', " ", text, flags=re.IGNORECASE)
    text = re.sub(r'\s*["\']\s*$', "", text)
    text = re.sub(r"\s+", " ", text).strip()
    # Truncate only at clear meta-commentary (not narrative "critical"); avoid cutting story mid-sentence
    for marker in (
        r"\s+Critical:\s+This theme",
        r"\s+Critical:\s+The key points",
        r"\s+The key points are:",
        r"\s+Opening:",
        r"\s+After the opening:",
    ):
        match = re.search(marker, text, re.IGNORECASE)
        if match:
            text = text[: match.start()].rstrip()
    lines = text.split("\n")
    out: list[str] = []
    skip_until_next_paragraph = False
    skip_meta_block = False
    for line in lines:
        stripped = line.strip()
        # Drop lines that are only "I am an AI/assistant" or close variants
        if re.match(r"^[\"']?I\s+am\s+(an\s+)?(AI|language\s+model)(\s*/\s*assistant)?[\"']?\.?$", stripped, re.IGNORECASE):
            continue
        # Drop paragraph that starts with meta-commentary (e.g. "However, critical:" ... "Therefore, I will use...")
        if "However, critical:" in stripped or "Therefore, I will use a generic opening" in stripped:
            skip_until_next_paragraph = True
            continue
        if skip_until_next_paragraph:
            if not stripped:
                skip_until_next_paragraph = False
            continue
        # Drop meta lines: "Critical:...", "The key points are:", "Opening:", "After the opening:", bullet-only lines
        if re.match(r"^(Critical:|The key points are:|Opening:|After the opening:)", stripped, re.IGNORECASE):
            skip_meta_block = True
            continue
        if skip_meta_block:
            if not stripped or (stripped.startswith("- ") and len(stripped) < 80):
                continue
            if re.match(r"^- ", stripped) and "key point" in stripped.lower():
                continue
            skip_meta_block = False
        # Drop line that is only meta (training context / not directly addressed)
        if stripped and ("was not directly addressed by this document's training" in stripped or "not explicitly stated" in stripped) and len(stripped) < 200:
            continue
        out.append(line)
    text = "\n".join(out).strip()
    # Collapse multiple newlines and trim each paragraph
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    text = " ".join(paragraphs) if paragraphs else text
    text = _rewrite_first_person_to_third(text)
    return text


def _rewrite_first_person_to_third(text: str) -> str:
    """Rewrite first-person pronouns to third person (Name) so story never contains I, we, my, our, me, us."""
    if not text or len(text) < 10:
        return text
    # Contractions first (word boundary)
    text = re.sub(r"\bI'm\b", "(Name) is", text, flags=re.IGNORECASE)
    text = re.sub(r"\bI've\b", "(Name) has", text, flags=re.IGNORECASE)
    text = re.sub(r"\bI'll\b", "(Name) will", text, flags=re.IGNORECASE)
    text = re.sub(r"\bwe're\b", "(Name) is", text, flags=re.IGNORECASE)
    text = re.sub(r"\bwe've\b", "(Name) has", text, flags=re.IGNORECASE)
    text = re.sub(r"\bwe'll\b", "(Name) will", text, flags=re.IGNORECASE)
    # Possessive and object forms
    text = re.sub(r"\bour\b", "their", text, flags=re.IGNORECASE)
    text = re.sub(r"\bmy\b", "(Name)'s", text, flags=re.IGNORECASE)
    text = re.sub(r"\bme\b", "(Name)", text, flags=re.IGNORECASE)
    text = re.sub(r"\bus\b", "(Name)", text, flags=re.IGNORECASE)
    text = re.sub(r"\bwe\b", "(Name)", text, flags=re.IGNORECASE)
    text = re.sub(r"\bI\b", "(Name)", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()


# Default max words for compact, summarized stories (configurable via generate_stories_from_breakout).
DEFAULT_MAX_STORY_WORDS = 130


def _first_n_words(s: str, n: int = 8) -> str:
    """Normalize and return first n words for fingerprinting (catches near-duplicate sentences)."""
    words = re.sub(r"\s+", " ", s.lower().strip()).split()
    return " ".join(words[:n]) if words else ""


def _truncate_to_max_words(text: str, max_words: int) -> str:
    """Return text truncated to at most max_words. Ends at a word boundary; prefers ending at a sentence."""
    if not text or max_words <= 0:
        return text
    words = text.split()
    if len(words) <= max_words:
        return text
    truncated = " ".join(words[:max_words])
    last_sentence = max(truncated.rfind(". "), truncated.rfind("! "), truncated.rfind("? "))
    if last_sentence > len(truncated) // 2:
        return truncated[: last_sentence + 1].strip()
    return truncated.strip()


def _deduplicate_repeated_phrases(text: str) -> str:
    """Remove duplicate and near-duplicate sentences and long repeated phrases to reduce LLM repetition."""
    if not text or len(text) < 50:
        return text
    # Split on sentence boundaries (period, exclamation, question)
    parts = re.split(r"(?<=[.!?])\s+", text)
    if len(parts) <= 1:
        return _remove_repeated_word_phrases(text)
    out: list[str] = []
    seen_fingerprints: set[str] = set()
    for p in parts:
        s = p.strip()
        if not s or len(s) < 10:
            continue
        # Skip if identical to an earlier sentence
        if s in out:
            continue
        # Skip if first 8 words match an earlier sentence (near-duplicate)
        fp = _first_n_words(s, 8)
        if fp and fp in seen_fingerprints:
            continue
        # Skip if this sentence is a fragment of a longer one we already kept
        if any(len(kept) > len(s) and s in kept for kept in out):
            continue
        out.append(s)
        if fp:
            seen_fingerprints.add(fp)
    result = " ".join(out).strip()
    if not result:
        return text
    return _remove_repeated_word_phrases(result)


def _remove_repeated_word_phrases(text: str, min_phrase_words: int = 10) -> str:
    """Remove long phrases that repeat; iterate until no change (handles multiple repeats)."""
    while True:
        words = text.split()
        if len(words) < min_phrase_words * 2:
            break
        changed = False
        for n in range(min(min_phrase_words, len(words) // 2), 0, -1):
            if n >= len(words):
                continue
            phrase = " ".join(words[:n])
            rest = " ".join(words[n:])
            pos = rest.find(phrase)
            if pos == -1:
                continue
            if pos == 0:
                result = " ".join(words[:n]) + " " + rest[len(phrase) :].strip()
            else:
                result = " ".join(words[:n]) + " " + rest[:pos].strip()
            new_text = result.strip() if result.strip() else text
            if new_text != text:
                text = new_text
                changed = True
            break
        if not changed:
            break
    return text


def _format_keypoints(topic: TopicKeypoints) -> str:
    """Format topic keypoints for the prompt; include mention count when present (confidence/rating)."""
    header = topic.name
    if topic.mentions is not None:
        header = f"{topic.name} ({topic.mentions} mentions)"
    if not topic.keypoints:
        return header
    lines = [f"- {k}" for k in topic.keypoints]
    return header + "\n" + "\n".join(lines)


def _sanitize_example_story(text: str, participant_names: list[str]) -> str:
    """Normalize example story for prompt: use (Name), no location, trim."""
    if not text or not text.strip():
        return ""
    t = text.strip()
    for name in participant_names:
        if name and name.strip():
            t = re.sub(re.escape(name.strip()), "(Name)", t, flags=re.IGNORECASE)
    t = re.sub(r'\(Location(?:\s+of\s+Session)?(?:\s*:\s*[^)"\']*)?\)', " ", t, flags=re.IGNORECASE)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _format_example_story_pattern(theme: ThemeBlock) -> str:
    """Format up to 2 example story paragraphs from this theme for prompt tailoring. Empty if none."""
    paragraphs = getattr(theme, "story_paragraphs", None) or []
    if not paragraphs:
        return "(No example stories from this document.)"
    names = getattr(theme, "participant_names", None) or []
    sanitized: list[str] = []
    for i, p in enumerate(paragraphs[:2]):
        if not p or not p.strip():
            continue
        s = _sanitize_example_story(p, names)
        if s:
            sanitized.append(f"Example {len(sanitized) + 1}: {s}")
    if not sanitized:
        return "(No example stories from this document.)"
    return "\n\n".join(sanitized)


def _get_pronoun_instruction(participant_pronoun: str | None) -> str:
    """Return prompt instruction for referring to (Name). When gender is unknown, use (Name) only; when known, use he or she consistently."""
    if participant_pronoun == "he":
        return "Refer to (Name) with he/him/his only; use consistently throughout (e.g. 'He shared that...', 'He added that...'). Do not use she or they."
    if participant_pronoun == "she":
        return "Refer to (Name) with she/her only; use consistently throughout (e.g. 'She shared that...', 'She added that...'). Do not use he or they."
    return "When referring to the participant, use (Name) only—do not use he, she, or they. Use '(Name) shared that...', '(Name) added that...', '(Name) credited...', '(Name) saw...'."


# Fixed context for story prompt: do not use vectordb; training data is for summarization patterns only, not for names/locations.
_STORY_CONTEXT_NO_KB = (
    "Base the story only on the Theme, Question, Topic, and Keypoints above. "
    "Do not retrieve or use other documents to add names, locations, or details. "
    "(Training data may inform summarization style elsewhere; for this story use only the theme and keypoints provided.)"
)


@dataclass
class StoryResult:
    """One generated story with metadata."""

    theme_number: int
    theme_title: str
    topic_name: str
    story_text: str


# Project root: go up from src/modules/questionnaire/story_generation.py
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
# Output directory at project root: output/<module_name>/ (e.g. output/questionnaire/)
_OUTPUT_DIR = _PROJECT_ROOT / "output"


def write_stories_to_file(
    stories: list[StoryResult],
    input_docx_path: str | Path,
    module_name: str = "questionnaire",
) -> Path:
    """
    Write story results to a .txt file under <project_root>/output/<module_name>/.
    Filename is the input DOCX stem with a timestamp suffix (YYYYMMDD_HHMMSS).
    """
    out_dir = _OUTPUT_DIR / module_name
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(input_docx_path).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"{stem}_{timestamp}.txt"
    lines: list[str] = []
    for s in stories:
        lines.append(f"--- Theme #{s.theme_number}: {s.theme_title} | {s.topic_name} ---")
        lines.append(s.story_text)
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def _resolve_kb_context(
    kb: KnowledgeBase | None,
    theme: ThemeBlock,
    topic: TopicKeypoints,
    domain: str,
) -> str:
    """Return fixed instruction for story prompt. KB/vectordb is not used; stories are based only on theme and keypoints."""
    return _STORY_CONTEXT_NO_KB


def _first_participant_name_from_extract(extract: BreakoutExtract) -> str | None:
    """Return the first non-empty participant name from any theme so one name is used for the whole report."""
    for theme in extract.themes:
        for name in theme.participant_names:
            if name and name.strip():
                return name.strip()
    return None


def _build_prompt_variables(
    theme: ThemeBlock,
    topic: TopicKeypoints,
    topic_index: int,
    kb: KnowledgeBase | None,
    domain: str,
    max_words: int,
    participant_pronoun: str | None,
    report_participant_name: str | None,
) -> dict:
    """Build the prompt variables dict used for one story. Used for invocation and for writing prompt to file."""
    keypoints_str = _format_keypoints(topic)
    kb_context = _resolve_kb_context(kb, theme, topic, domain)
    example_story_pattern = _format_example_story_pattern(theme)
    pronoun_instruction = _get_pronoun_instruction(participant_pronoun)
    return {
        "theme_title": theme.title,
        "question": theme.question,
        "topic_name": topic.name,
        "keypoints": keypoints_str,
        "example_story_pattern": example_story_pattern,
        "kb_context": kb_context,
        "max_words": max_words,
        "pronoun_instruction": pronoun_instruction,
    }


async def _generate_one_story(
    chain: object,
    theme: ThemeBlock,
    topic: TopicKeypoints,
    topic_index: int,
    kb: KnowledgeBase | None,
    domain: str,
    max_words: int,
    participant_pronoun: str | None = None,
    report_participant_name: str | None = None,
    write_prompts_to: Path | None = None,
) -> StoryResult:
    """Generate a single story for one topic. Replaces (Name) with participant name when available from DOCX."""
    prompt_vars = _build_prompt_variables(
        theme, topic, topic_index, kb, domain, max_words, participant_pronoun, report_participant_name
    )
    if write_prompts_to is not None:
        write_prompts_to.mkdir(parents=True, exist_ok=True)
        prompt_text = STORY_TEMPLATE.format(**prompt_vars)
        path = write_prompts_to / f"prompt_theme{theme.theme_number}_topic{topic_index}.txt"
        path.write_text(prompt_text, encoding="utf-8")
        logger.info("Wrote prompt for theme %s topic %s to %s", theme.theme_number, topic_index, path)
    participant_name: str | None = report_participant_name
    if participant_name is None and topic_index < len(theme.participant_names):
        participant_name = theme.participant_names[topic_index].strip() or None
    try:
        out = await chain.ainvoke(prompt_vars)
        text = (out.content if hasattr(out, "content") else str(out)).strip()
        text = _sanitize_story_output(text)
        if len(text) < MIN_STORY_LENGTH:
            logger.warning(
                "Story too short (%d chars) for theme %s topic %s; treating as failed.",
                len(text),
                theme.title,
                topic.name,
            )
            return StoryResult(
                theme_number=theme.theme_number,
                theme_title=theme.title,
                topic_name=topic.name,
                story_text="[Story generation failed.]",
            )
        text = _deduplicate_repeated_phrases(text)
        text = _truncate_to_max_words(text, max_words)
        if participant_name:
            text = text.replace("(Name)", participant_name)
        return StoryResult(
            theme_number=theme.theme_number,
            theme_title=theme.title,
            topic_name=topic.name,
            story_text=text,
        )
    except Exception as e:
        logger.warning(
            "Story generation failed for theme %s topic %s: %s",
            theme.title,
            topic.name,
            e,
            exc_info=True,
        )
        return StoryResult(
            theme_number=theme.theme_number,
            theme_title=theme.title,
            topic_name=topic.name,
            story_text="[Story generation failed.]",
        )


async def generate_stories_from_breakout(
    extract: BreakoutExtract,
    kb: KnowledgeBase | None = None,
    domain: str = "",
    llm_model_path: str | None = None,
    max_stories: int | None = None,
    max_words: int = DEFAULT_MAX_STORY_WORDS,
    participant_pronoun: str | None = None,
    write_prompts_to: Path | None = None,
) -> list[StoryResult]:
    """
    Generate narrative stories from breakout keypoints using the same LLM. Stories are based only on theme and keypoints; the vectordb/KB is not used.

    Training data (if present elsewhere) is for understanding summarization patterns only—not for constructing names, locations, or other details. No retrieval is performed.

    Args:
        extract: Result of extract_breakout_keypoints(docx_path).
        kb: Unused; kept for API compatibility. No vectordb retrieval is performed.
        domain: Unused; kept for API compatibility.
        llm_model_path: Optional LLM path; same semantics as KnowledgeBase.
        max_stories: Cap number of stories (default all topics); use for quick tests.
        max_words: Maximum word count per story for compact, summarized output (default 130).
        participant_pronoun: When gender is known, "he" or "she" for consistent pronouns; when None (gender unknown), use (Name) only and do not use pronouns.

    Returns:
        List of StoryResult, one per topic (or up to max_stories).
    """
    # Allow enough tokens so the model can complete the story (avoid mid-sentence cutoff).
    # n_ctx must fit prompt + completion; use 4096 so long prompts + 2k completion don't truncate.
    story_max_tokens = max(2048, max_words * 5)
    llm = _create_llm(llm_model_path, max_tokens=story_max_tokens, n_ctx=4096)
    prompt = PromptTemplate(
        template=STORY_TEMPLATE,
        input_variables=["theme_title", "question", "topic_name", "keypoints", "example_story_pattern", "kb_context", "max_words", "pronoun_instruction"],
    )
    chain = prompt | llm
    report_participant_name = _first_participant_name_from_extract(extract)
    results: list[StoryResult] = []
    count = 0
    for theme in extract.themes:
        topics = theme.topics
        if not topics:
            # Fallback: no numbered topics parsed (e.g. DOCX format differs); generate one story per theme
            topics = [TopicKeypoints(name=theme.title, keypoints=[theme.question] if theme.question else [])]
        for topic_index, topic in enumerate(topics):
            if max_stories is not None and count >= max_stories:
                return results
            result = await _generate_one_story(
                chain, theme, topic, topic_index, kb, domain, max_words,
                participant_pronoun=participant_pronoun,
                report_participant_name=report_participant_name,
                write_prompts_to=write_prompts_to,
            )
            results.append(result)
            count += 1
    return results
