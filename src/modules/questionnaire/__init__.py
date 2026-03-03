"""
Questionnaire module: generate summarized context from a filled questionnaire
(what the user agreed to + user background) using domain training data.
Training data lives under input/questionnaire/<domain>/.
"""
from .summary import (
    QuestionnaireSummaryResult,
    generate_summary,
)

# Subfolder under input/ for this module's training data (input/questionnaire/<domain>/)
INPUT_SUBDIR = "questionnaire"

__all__ = ["INPUT_SUBDIR", "generate_summary", "QuestionnaireSummaryResult"]
