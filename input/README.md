# Training data – expertise domains

Training data is organized by **module** and **domain**: `input/<module>/<domain>/`.

Each module is a local LLM expertise area (e.g. questionnaire, another_module). Each domain is a subfolder under that module with supported documents (PDF, DOCX, TXT, EPUB) and optional config.

## Layout

```
input/
├── questionnaire/           ← questionnaire module
│   ├── legal/                 domain: legal
│   │   ├── domain.json
│   │   ├── questionnaire_template.json
│   │   ├── terms.pdf
│   │   └── policy.docx
│   └── hr/
│       ├── domain.json
│       └── handbook.pdf
├── another_module/          ← other modules (placeholder)
│   └── <domain>/
├── domain.example.json     (copy into <module>/<domain>/)
└── questionnaire_template.example.json
```

- **domain.json** (optional): `{"name": "Legal", "description": "..."}`. Copy from `domain.example.json`.
- **questionnaire_template.json** (optional, questionnaire module): Same format as the filled questionnaire. Copy from `questionnaire_template.example.json`. See `questionnaire/README.md`.

## In the app

- **Load training data from input/questionnaire/**: Indexes all domains under `input/questionnaire/` (e.g. legal, hr). Domain dropdowns then list each loaded domain.
- **Chat** and **Questionnaire summary**: Choose a domain to focus on that domain’s documents.

Programmatic: `kb.init_from_directory("input/questionnaire")` returns the list of loaded domain ids for the questionnaire module.
