# Questionnaire module – training data

Place **domain subfolders** here (e.g. `legal/`, `hr/`). Each domain folder contains:

- Supported documents: PDF, DOCX, TXT, EPUB
- Optional `domain.json` (name, description)
- Optional `questionnaire_template.json` (same format as filled questionnaire)

Example:

```
input/questionnaire/
├── legal/
│   ├── domain.json
│   ├── questionnaire_template.json
│   ├── terms.pdf
│   └── policy.docx
└── hr/
    ├── domain.json
    └── handbook.pdf
```

Copy `domain.example.json` and `questionnaire_template.example.json` from `input/` into each domain folder as needed.
