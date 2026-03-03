# Training data – expertise domains

Training data is organized by **module** and optionally **domain**: `input/<module>/` or `input/<module>/<domain>/`.

Training data can be **domain-based** (subfolders per domain) or **flat** (all files in the module folder). The app uses the selected domain to retrieve **closest-related** content (domain as query hint when data is flat). You can **generate and add** new training data and sort it into a domain.

## Layout

```
input/
├── questionnaire/
│   ├── legal/                 domain-based
│   │   ├── domain.json
│   │   ├── questionnaire_template.json   (or template.pdf / template.docx)
│   │   ├── terms.pdf
│   │   └── policy.docx
│   ├── hr/
│   ├── domains.json          flat: list of domain labels (copy from domains.example.json)
│   ├── template.pdf          optional module-level template (PDF/DOCX/TXT/EPUB)
│   └── *.pdf, *.docx, ...    flat: all docs in module root
├── another_module/
│   └── ...
├── domain.example.json
├── domains.example.json
└── questionnaire_template.example.json
```

- **domain.json** (optional): `{"name": "Legal", "description": "..."}`. Copy from `domain.example.json`.
- **questionnaire_template.json** or **template.pdf / template.docx** (optional): Questionnaire template (same format as filled questionnaire, or any supported doc format). Template and sample training data can be provided later in PDF/DOCX/etc.
- **domains.json** (optional, flat layout): `{"domains": ["legal", "hr"]}`. Copy from `domains.example.json`. When training data is not in domain folders, this lists domain labels for closest-related retrieval.

## In the app

- **Load training data from input/questionnaire/**: Indexes domain subfolders (if any) or all files in the module root (flat). Domain dropdown shows loaded domains or labels from `domains.json`.
- **Chat** and **Questionnaire summary**: Choose a domain; retrieval uses that domain’s collection or (when flat) domain as a query hint for closest-related content.
- **Create/generate new training data**: Use `kb.add_training_text(domain, text, source)` and `save_training_document(module_input_dir, domain, content, filename)` or `add_and_save_training_example` from the questionnaire module to add and optionally save new content under a domain.

Programmatic: `kb.init_from_directory("input/questionnaire")`, `kb.add_training_text(domain, text, source)`, `save_training_document(module_input_dir, domain, content, filename)`.
