"""
Generate short narrative stories from Breakout Group Discussion keypoints.
Uses the same LLM and optional KB (training data) as the questionnaire module.
"""
import logging
import re
from dataclasses import dataclass

from langchain_core.prompts import PromptTemplate

logger = logging.getLogger(__name__)

from ...knowledge_base import KnowledgeBase, _create_llm

from .breakout_extract import BreakoutExtract, ThemeBlock, TopicKeypoints


STORY_TEMPLATE = """Write one short narrative story for a report, in the style of the training examples. Assume the point of view of a breakout participant; use the theme, question, and keypoints below (and any relevant context) to create a concrete, specific story.

Theme: {theme_title}
Question asked: {question}
Topic (category): {topic_name}
Keypoints (participants' answers; mention count = how often raised—use for emphasis):
{keypoints}
{kb_context}

Story structure (match the pattern in training data; your reply must be only the story—no introduction, no "I am an AI/assistant"):
- One continuous paragraph of prose only. No headings, bullet points, lists, or echoed topic/keypoint labels. Use the literal "(Name)" for the participant (not "He/She"). Use third person only (he/his or she/her for (Name)); never I, me, my, we, our.
- Length: summarize in at most {max_words} words. Be compact and precise; every sentence must add new information. Do not repeat any phrase, clause, or sentence.
- Opening (choose one style from training examples):
  - "(Name) shared of how [topic or theme in a few words]. [Next sentence: He/She shared that... or He/She added that...]"
  - "(Name), who [brief context e.g. has been living in Singapore for X years], shared that [specific observation]."
  - "(Name), a [brief role e.g. father of 2], shared his/her experience of how [topic]. He/She shared that [concrete example]."
- After the opening: 1–3 more sentences with concrete details (e.g. a specific example, number, or situation). Use "He shared that...", "She added that...", "He credited...", "She saw..." as in the training examples. Use "(Location of Session)" only if relevant. Do not repeat the same or similar wording.
- Training-style example: "(Name) shared of how Singapore is solution-focused. She shared that during Covid-19 she saw the government implement different measures to resolve the situation. She credited Singapore's success to good strategic planning by the leaders." Wrong: "(He/She) shared..."; wrong: listing keypoints; wrong: generic "I am proud"; wrong: repeating words, phrases, or sentences. Right: one paragraph, third person, concrete example, no repetition.

Your reply must be exactly one paragraph of prose starting with "(Name)" (e.g. "(Name) shared of how..." or "(Name), who..., shared that..."). No headings, no lists, no echoed keypoints—only the story text. Stay within {max_words} words and do not repeat any phrase or sentence."""

# Minimum length for a valid story; shorter output is treated as failed (e.g. small models often emit EOS after 1–2 tokens).
MIN_STORY_LENGTH = 100
# Default max words for compact, summarized stories (configurable via generate_stories_from_breakout).
DEFAULT_MAX_STORY_WORDS = 300


def _deduplicate_repeated_phrases(text: str) -> str:
    """Remove consecutive duplicate sentences or long repeated fragments to reduce LLM repetition."""
    if not text or len(text) < 50:
        return text
    # Split on sentence boundaries (period, space, optional quote)
    parts = re.split(r"(?<=[.!?])\s+", text)
    if len(parts) <= 1:
        return text
    out: list[str] = [parts[0]]
    for p in parts[1:]:
        s = p.strip()
        if not s:
            continue
        # Skip if identical to previous sentence (consecutive duplicate)
        if out and s == out[-1]:
            continue
        # Skip if this sentence is a duplicate of an earlier one (same text already seen)
        if s in out:
            continue
        out.append(s)
    result = " ".join(out).strip()
    return result if result else text


def _format_keypoints(topic: TopicKeypoints) -> str:
    """Format topic keypoints for the prompt; include mention count when present (confidence/rating)."""
    header = topic.name
    if topic.mentions is not None:
        header = f"{topic.name} ({topic.mentions} mentions)"
    if not topic.keypoints:
        return header
    lines = [f"- {k}" for k in topic.keypoints]
    return header + "\n" + "\n".join(lines)


def _get_kb_context(kb: KnowledgeBase, theme_title: str, topic_name: str, keypoints: list[str], domain: str = "", k: int = 3) -> str:
    """Retrieve relevant training-data context for this topic."""
    try:
        query = f"{theme_title} {topic_name} " + " ".join(keypoints[:5])
        docs = kb.similarity_search(query, k=k, domain=domain, domain_as_query_hint=True)
        if not docs:
            return ""
        seen = set()
        parts = []
        for doc in docs:
            content = (doc.page_content or "").strip()[:500]
            if content and content[:80] not in seen:
                seen.add(content[:80])
                parts.append(content)
        if not parts:
            return ""
        return "Relevant context from training data:\n" + "\n\n".join(parts)
    except Exception:
        return ""


@dataclass
class StoryResult:
    """One generated story with metadata."""

    theme_number: int
    theme_title: str
    topic_name: str
    story_text: str


def _resolve_kb_context(
    kb: KnowledgeBase | None,
    theme: ThemeBlock,
    topic: TopicKeypoints,
    domain: str,
) -> str:
    """Resolve kb_context for the prompt; returns fallback if no KB or empty."""
    if kb is None:
        return "(No additional context.)"
    ctx = _get_kb_context(kb, theme.title, topic.name, topic.keypoints, domain=domain)
    return ctx if ctx else "(No additional context.)"


async def _generate_one_story(
    chain: object,
    theme: ThemeBlock,
    topic: TopicKeypoints,
    topic_index: int,
    kb: KnowledgeBase | None,
    domain: str,
    max_words: int,
) -> StoryResult:
    """Generate a single story for one topic. Replaces (Name) with participant name when available from DOCX."""
    keypoints_str = _format_keypoints(topic)
    kb_context = _resolve_kb_context(kb, theme, topic, domain)
    participant_name: str | None = None
    if topic_index < len(theme.participant_names):
        participant_name = theme.participant_names[topic_index].strip() or None
    try:
        out = await chain.ainvoke({
            "theme_title": theme.title,
            "question": theme.question,
            "topic_name": topic.name,
            "keypoints": keypoints_str,
            "kb_context": kb_context,
            "max_words": max_words,
        })
        text = (out.content if hasattr(out, "content") else str(out)).strip()
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
) -> list[StoryResult]:
    """
    Generate narrative stories from breakout keypoints using the same trained data (KB) and LLM.

    Args:
        extract: Result of extract_breakout_keypoints(docx_path).
        kb: Optional knowledge base (e.g. initialized from input/questionnaire/); used for context.
        domain: Domain for KB retrieval when using flat layout.
        llm_model_path: Optional LLM path; same semantics as KnowledgeBase.
        max_stories: Cap number of stories (default all topics); use for quick tests.
        max_words: Maximum word count per story for compact, summarized output (default 300).

    Returns:
        List of StoryResult, one per topic (or up to max_stories).
    """
    llm = _create_llm(llm_model_path)
    prompt = PromptTemplate(
        template=STORY_TEMPLATE,
        input_variables=["theme_title", "question", "topic_name", "keypoints", "kb_context", "max_words"],
    )
    chain = prompt | llm
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
            result = await _generate_one_story(chain, theme, topic, topic_index, kb, domain, max_words)
            results.append(result)
            count += 1
    return results
