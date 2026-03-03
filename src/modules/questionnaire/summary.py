"""
Generate summarized context from a filled questionnaire: what the user agreed to
and the user's background. Uses the knowledge base (training data) for context
when available.
"""
from dataclasses import dataclass

from langchain_core.prompts import PromptTemplate

from ...knowledge_base import KnowledgeBase, _create_llm


SUMMARY_TEMPLATE = """You are summarizing a user's filled questionnaire for records.

Questionnaire answers:
{questionnaire_text}

Relevant context from documents (terms, policies, agreements):
{kb_context}
{template_context}
{domain_focus}

Produce exactly two short sections:

1) WHAT THE USER AGREED TO: Summarize what the user has agreed to, based on their answers and the document context above. If there is no agreement-related content, write "No specific agreements indicated."

2) USER BACKGROUND: Summarize the user's background (role, experience, or other relevant details) from the questionnaire. If none is provided, write "No background details provided."

Use clear, neutral language. Output only the two sections with these exact headers.
If a domain focus was provided above, tailor the summary to that domain."""


def _format_questionnaire(questionnaire: dict[str, str]) -> str:
    """Turn a questionnaire dict into readable text for the prompt."""
    if not questionnaire:
        return "(No answers provided)"
    lines = [f"- {k}: {v}" for k, v in questionnaire.items()]
    return "\n".join(lines)


def _get_kb_context(kb: KnowledgeBase, domain: str = "", k: int = 4) -> str:
    """Retrieve relevant chunks from the given domain's collection."""
    try:
        docs_agreement = kb.similarity_search(
            "agreement terms conditions policy consent", k=k, domain=domain
        )
        docs_background = kb.similarity_search(
            "role experience background qualifications", k=2, domain=domain
        )
        seen = set()
        combined = []
        for doc in docs_agreement + docs_background:
            if doc.page_content and doc.page_content[:100] not in seen:
                seen.add(doc.page_content[:100])
                combined.append(doc.page_content.strip())
        if not combined:
            return "(No relevant document context in the knowledge base.)"
        return "\n\n---\n\n".join(combined[:6])
    except Exception:
        return "(Knowledge base unavailable or empty.)"


@dataclass
class QuestionnaireSummaryResult:
    """Structured result of questionnaire summary generation."""

    agreed_summary: str
    background_summary: str

    def to_display(self) -> str:
        """Single formatted string for UI display."""
        return (
            f"**What you agreed to**\n\n{self.agreed_summary}\n\n"
            f"**Your background**\n\n{self.background_summary}"
        )


def _parse_llm_output(text: str) -> QuestionnaireSummaryResult:
    """Parse LLM output into agreed_summary and background_summary."""
    agreed = ""
    background = ""
    current = "agreed"
    for line in text.split("\n"):
        line_lower = line.strip().lower()
        if "what the user agreed to" in line_lower or "what you agreed" in line_lower:
            current = "agreed"
            after = line.split(":", 1)[-1].strip() if ":" in line else ""
            if after:
                agreed = after
            continue
        if "user background" in line_lower or "your background" in line_lower:
            current = "background"
            after = line.split(":", 1)[-1].strip() if ":" in line else ""
            if after:
                background = after
            continue
        if not line.strip():
            continue
        if current == "agreed":
            agreed = f"{agreed}\n{line}".strip() if agreed else line.strip()
        else:
            background = f"{background}\n{line}".strip() if background else line.strip()
    return QuestionnaireSummaryResult(
        agreed_summary=agreed or "No summary produced.",
        background_summary=background or "No summary produced.",
    )


def _domain_focus_text(domain: str, domain_display_name: str = "") -> str:
    """Instruction line for the prompt when a domain is selected."""
    if not domain:
        return ""
    label = domain_display_name or domain.replace("_", " ").title()
    return f"Domain focus: Summarize in the context of \"{label}\"."


async def generate_summary(
    questionnaire: dict[str, str],
    kb: KnowledgeBase | None = None,
    domain: str = "",
    domain_display_name: str = "",
    template_document_context: str = "",
    llm_model_path: str | None = None,
) -> QuestionnaireSummaryResult:
    """
    Generate a summarized context from a filled questionnaire: what the user
    agreed to and the user's background. Uses the given domain's training data
    (or closest-related when data is flat). Template can be PDF/DOCX/etc. for extra context.

    Args:
        questionnaire: Map of field names to answers (e.g. {"name": "Jane", "role": "Engineer"}).
        kb: Optional knowledge base; if provided, relevant chunks are used (domain as query hint when flat).
        domain: Domain id or label; biases retrieval to closest-related content when training data is not domain-based.
        domain_display_name: Optional display name for the domain in the prompt.
        template_document_context: Optional text from a template document (PDF, DOCX, etc.) for the prompt.
        llm_model_path: Optional path to GGUF model; same semantics as KnowledgeBase.

    Returns:
        QuestionnaireSummaryResult with agreed_summary and background_summary.
    """
    questionnaire_text = _format_questionnaire(questionnaire)
    kb_context = (
        _get_kb_context(kb, domain=domain) if kb else "(No knowledge base provided.)"
    )
    template_context = (
        f"Template or reference document:\n{template_document_context}"
        if template_document_context
        else ""
    )
    domain_focus = _domain_focus_text(domain, domain_display_name)

    prompt = PromptTemplate(
        template=SUMMARY_TEMPLATE,
        input_variables=["questionnaire_text", "kb_context", "template_context", "domain_focus"],
    )
    llm = _create_llm(llm_model_path)
    chain = prompt | llm
    result = await chain.ainvoke({
        "questionnaire_text": questionnaire_text,
        "kb_context": kb_context,
        "template_context": template_context,
        "domain_focus": domain_focus,
    })
    if hasattr(result, "content"):
        text = result.content
    else:
        text = str(result)
    return _parse_llm_output(text)
