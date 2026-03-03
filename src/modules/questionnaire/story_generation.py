"""
Generate short narrative stories from Breakout Group Discussion keypoints.
Uses the same LLM and optional KB (training data) as the questionnaire module.
"""
import logging
from dataclasses import dataclass

from langchain_core.prompts import PromptTemplate

logger = logging.getLogger(__name__)

from ...knowledge_base import KnowledgeBase, _create_llm

from .breakout_extract import BreakoutExtract, ThemeBlock, TopicKeypoints


STORY_TEMPLATE = """Assume the point of view of a breakout participant based on the theme, question, and keypoints below. Write one short narrative story for a report.

Theme: {theme_title}
Question asked: {question}
Topic (category): {topic_name}
Keypoints (participants' answers; mention count = how often raised—use for emphasis):
{keypoints}
{kb_context}

Story structure (your reply must be only this—no introduction, no "I am an AI/assistant", no description of the task):
- Output format: One continuous paragraph of prose only. No headings, no bullet points, no lists, no line breaks between sentences, no numbers, no repetition of topic names or keypoint labels. Do not echo the keypoints or theme data—weave the ideas into flowing sentences only. Start with the literal "(Name)" (not "He/She").
- (Name) is the participant (e.g. Lyon). The whole story is about (Name) in the third person. Never use I, me, my, we, or our.
- First sentence: "(Name) shared about [brief topic summary]." or "(Name) shared of how [brief topic summary]."
- Next 2–4 sentences: Refer to (Name) with third-person pronouns (he/his or she/her). Use "(Location of Session)" only if relevant. One short paragraph only (2–5 sentences total).
- Example: "(Name) shared about safety. He said that security is his priority." Wrong: "(He/She) shared..."; wrong: listing keypoints or topic names. Right: one block of prose.

Your reply must be exactly one paragraph of prose, starting with "(Name) shared about" or "(Name) shared of how". No headings, no lists, no echoed keypoints—only the story text."""

# Minimum length for a valid story; shorter output is treated as failed (e.g. small models often emit EOS after 1–2 tokens).
MIN_STORY_LENGTH = 100


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
    kb: KnowledgeBase | None,
    domain: str,
) -> StoryResult:
    """Generate a single story for one topic."""
    keypoints_str = _format_keypoints(topic)
    kb_context = _resolve_kb_context(kb, theme, topic, domain)
    try:
        out = await chain.ainvoke({
            "theme_title": theme.title,
            "question": theme.question,
            "topic_name": topic.name,
            "keypoints": keypoints_str,
            "kb_context": kb_context,
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
) -> list[StoryResult]:
    """
    Generate narrative stories from breakout keypoints using the same trained data (KB) and LLM.

    Args:
        extract: Result of extract_breakout_keypoints(docx_path).
        kb: Optional knowledge base (e.g. initialized from input/questionnaire/); used for context.
        domain: Domain for KB retrieval when using flat layout.
        llm_model_path: Optional LLM path; same semantics as KnowledgeBase.
        max_stories: Cap number of stories (default all topics); use for quick tests.

    Returns:
        List of StoryResult, one per topic (or up to max_stories).
    """
    llm = _create_llm(llm_model_path)
    prompt = PromptTemplate(
        template=STORY_TEMPLATE,
        input_variables=["theme_title", "question", "topic_name", "keypoints", "kb_context"],
    )
    chain = prompt | llm
    results: list[StoryResult] = []
    count = 0
    for theme in extract.themes:
        topics = theme.topics
        if not topics:
            # Fallback: no numbered topics parsed (e.g. DOCX format differs); generate one story per theme
            topics = [TopicKeypoints(name=theme.title, keypoints=[theme.question] if theme.question else [])]
        for topic in topics:
            if max_stories is not None and count >= max_stories:
                return results
            result = await _generate_one_story(chain, theme, topic, kb, domain)
            results.append(result)
            count += 1
    return results
