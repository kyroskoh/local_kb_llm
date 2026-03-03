# Questionnaire module – training data

Training data can be **domain-based** (subfolders) or **flat** (all files here). Template and sample data can be PDF, DOCX, TXT, or EPUB (same as document_loader).

**Domain-based:** place domain subfolders (e.g. `legal/`, `hr/`). Each can have:
- Supported docs: PDF, DOCX, TXT, EPUB
- Optional `domain.json`, `questionnaire_template.json`, or **template.pdf / template.docx** as the questionnaire template

**Flat:** put all PDF/DOCX/TXT/EPUB files directly here and add **domains.json** with `{"domains": ["legal", "hr", ...]}` so the app can still use domain labels for closest-related retrieval. Optionally add **template.pdf** (or template.docx) at this level.

**Add new training data:** use `add_training_text` / `save_training_document` or `add_and_save_training_example` to generate and sort new data into a domain (folder created if needed).
