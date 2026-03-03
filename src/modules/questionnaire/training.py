"""
Add and persist new training data for the questionnaire module, sorted by domain.
"""
from pathlib import Path

from ...knowledge_base import KnowledgeBase
from ...input_domains import save_training_document


def add_and_save_training_example(
    kb: KnowledgeBase,
    module_input_dir: str | Path,
    domain: str,
    text: str,
    source: str = "generated",
    save_filename: str | None = None,
) -> Path | None:
    """
    Add generated or new training text to the KB for the domain and optionally
    save it to the module's domain folder so it can be reloaded later.

    Args:
        kb: Knowledge base to add the document to.
        module_input_dir: Module input root (e.g. input/questionnaire/).
        domain: Domain to sort the example into (e.g. legal, hr).
        text: Training text content.
        source: Source label for the document metadata.
        save_filename: If set, save content to this filename under
            <module_input_dir>/<domain>/ (e.g. generated_2024-01-15.txt).

    Returns:
        Path to the saved file if save_filename was set, else None.
    """
    kb.add_training_text(domain=domain, text=text, source=source)
    if save_filename:
        return save_training_document(
            module_input_dir, domain, text, save_filename
        )
    return None
