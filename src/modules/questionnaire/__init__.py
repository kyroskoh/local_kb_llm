"""
Questionnaire module: generate summarized context from a filled questionnaire
(what the user agreed to + user background) using domain training data.
Training data lives under input/questionnaire/<domain>/ (or flat with domains.json).
"""
from .summary import (
    QuestionnaireSummaryResult,
    generate_summary,
)
from .training import add_and_save_training_example

# Subfolder under input/ for this module's training data (input/questionnaire/<domain>/)
INPUT_SUBDIR = "questionnaire"

__all__ = [
    "INPUT_SUBDIR",
    "add_and_save_training_example",
    "generate_summary",
    "QuestionnaireSummaryResult",
]
