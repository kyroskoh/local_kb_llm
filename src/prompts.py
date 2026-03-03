"""Prompt template for QA with sources."""
from langchain_core.prompts import PromptTemplate

QA_SOURCES_TEMPLATE = """Use the following pieces of context to answer the question at the end. If you don't know the answer based on the context, just say that you don't know; don't try to make up an answer. Keep the answer concise and cite the sources.

{summaries}

Question: {question}

Answer with sources:"""


def prompt_template() -> PromptTemplate:
    """Return the prompt used by the QA-with-sources chain (stuff chain uses 'summaries')."""
    return PromptTemplate(
        template=QA_SOURCES_TEMPLATE,
        input_variables=["summaries", "question"],
    )
