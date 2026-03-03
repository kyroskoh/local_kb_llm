# Questionnaire module – training data

Training data can be **domain-based** (subfolders) or **flat** (all files here). Template and sample data can be PDF, DOCX, TXT, or EPUB (same as document_loader).

**Domain-based:** place domain subfolders (e.g. `legal/`, `hr/`). Each can have:
- Supported docs: PDF, DOCX, TXT, EPUB
- Optional `domain.json`, `questionnaire_template.json`, or **template.pdf / template.docx** as the questionnaire template

**Flat:** put all PDF/DOCX/TXT/EPUB files directly here and add **domains.json** with `{"domains": ["legal", "hr", ...]}` so the app can still use domain labels for closest-related retrieval. Optionally add **template.pdf** (or template.docx) at this level.

**Add new training data:** use `add_training_text` / `save_training_document` or `add_and_save_training_example` to generate and sort new data into a domain (folder created if needed).

---

**Breakout Group Discussion (report DOCX):** For report DOCX files (e.g. `CSS Report_2025_10_01_Grp 1 Lyon.docx`) that contain a "Breakout Group Discussion" section with themes, questions, and topic keypoints, you can extract keypoints for story generation:

```bash
# From project root, with venv activated
python -m src.modules.questionnaire.breakout_extract "input/questionnaire/CSS Report_2025_10_01_Grp 1 Lyon.docx"
# JSON output (for downstream story generation):
python -m src.modules.questionnaire.breakout_extract "input/questionnaire/CSS Report_2025_10_01_Grp 1 Lyon.docx" --json
```

Output is structured by theme → topics → keypoints (and optional mention counts) for use in generating stories from the discussion points.

**Test story generation (same training data):** From project root with venv active, run the test script to extract keypoints, load training data from `input/questionnaire/`, and generate narrative stories via the same LLM/KB as the app:

```bash
python scripts/test_breakout_stories.py "input/questionnaire/CSS Report_2025_10_01_Grp 1 Lyon.docx"
# Limit to first N stories (e.g. 2) for a quick run:
python scripts/test_breakout_stories.py "input/questionnaire/CSS Report_2025_10_01_Grp 1 Lyon.docx" --max 2
```
