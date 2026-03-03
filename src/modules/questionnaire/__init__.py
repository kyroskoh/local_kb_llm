"""
Questionnaire module: generate summarized context from a filled questionnaire
(what the user agreed to + user background) using domain training data.
Training data lives under input/questionnaire/<domain>/ (or flat with domains.json).
"""
from .breakout_extract import (
    BreakoutExtract,
    ThemeBlock,
    TopicKeypoints,
    extract_breakout_keypoints,
)
from .story_generation import (
    StoryResult,
    generate_stories_from_breakout,
)
from .summary import (
    QuestionnaireSummaryResult,
    generate_summary,
)
from .training import add_and_save_training_example

# Subfolder under input/ for this module's training data (input/questionnaire/<domain>/)
INPUT_SUBDIR = "questionnaire"

__all__ = [
    "BreakoutExtract",
    "INPUT_SUBDIR",
    "ThemeBlock",
    "TopicKeypoints",
    "StoryResult",
    "add_and_save_training_example",
    "extract_breakout_keypoints",
    "generate_stories_from_breakout",
    "generate_summary",
    "QuestionnaireSummaryResult",
]
