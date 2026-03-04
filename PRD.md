# Product Requirements Document (PRD)

## Local Knowledge Base with LLMs

**Version:** 1.0  
**Status:** Living document

---

## 1. Vision and goals

Build a **local knowledge base** that:

- Ingests your own documents (PDF, DOCX, TXT, EPUB) and answers questions from them with source grounding.
- Supports **expertise modules** (e.g. questionnaire) with optional **domain** structure for focused retrieval and summarization.
- Uses a **local LLM** (GGUF) by default for QA, with optional OpenAI; embeddings remain OpenAI-based.
- Lets users **generate and add** new training data and **sort it by domain** for future runs.

Inspired by [Build Local Knowledge Base Using LLMs](https://cellsummer.github.io/views/2023-07-23-Build-Local_Knowledge-Base-With-LLMs.html).

---

## 2. User personas and outcomes

| Persona | Outcome |
|--------|---------|
| **Content owner** | Load documents once (by domain or flat), then ask questions and get answers grounded in those docs. |
| **Questionnaire operator** | Select a domain, paste or fill a questionnaire, and get a concise summary of what the user agreed to and their background. |
| **Developer** | Extend with new modules or domains; add training data programmatically and persist it under the right domain. |

---

## 3. Scope

### In scope

- Document ingestion: PDF, DOCX, TXT, EPUB (via LangChain).
- Vector store: Chroma with persistent storage; multi-collection per domain when domain-based.
- QA: Local GGUF (e.g. Qwen2.5) or OpenAI; answers cite sources or say “I don’t know” when not in context.
- **Questionnaire module**: Summarize filled questionnaires (agreed + background) using module training data; support template as JSON or PDF/DOCX/TXT/EPUB. **Breakout story generation**: Extract themes, questions, topics, and keypoints from the Breakout Group Discussion section of report DOCX; generate short narrative stories per topic (or per theme) using the same LLM and optional training data. Stories are tailored to each document’s theme statement and keypoints (document-agnostic; each DOCX has its own themes and keypoint style). Participant names from the DOCX Stories subsection replace the (Name) placeholder when available; stories are constrained by max words (default 300) and post-processed to remove repeated sentences and phrases.
- **Input layout**: `input/<module>/<domain>/` (domain-based) or flat `input/<module>/` with optional `domains.json` for domain labels.
- **Closest-related retrieval**: When data is flat, selected domain is used as a query hint so retrieval is domain-biased.
- **Add/sort training data**: API to add new text to a domain and optionally save it under `input/<module>/<domain>/`.
- **Embeddings**: OpenAI or local (sentence-transformers) via `EMBEDDING_MODEL` in `.env`.
- Web UI: NiceGUI (Chat + Questionnaire summary, domain selectors, load training data).

### Out of scope (current)

- Authentication / multi-tenancy.
- Other modules beyond questionnaire (structure is ready; implementation is placeholder).

---

## 4. Input layout and config files

Training data lives under **input/** and is organized by **module** and optionally **domain**.

### 4.1 Two config patterns

| Need | File | Location | Purpose |
|------|------|----------|---------|
| **Domain-based layout** (subfolders per domain) | **domain.example.json** | Copy as `input/<module>/<domain>/domain.json` (e.g. `input/questionnaire/legal/domain.json`) | Gives that domain a **display name** and **description** (e.g. "Legal", "Legal compliance and agreements"). One file **per domain folder**. |
| **Flat layout** (all files in module root) | **domains.example.json** | Copy as `input/<module>/domains.json` (e.g. `input/questionnaire/domains.json`) | Lists **domain labels** (e.g. `["legal", "hr", "compliance"]`) so the app can still show domain options and use them as **query hints** for closest-related retrieval. **One file per module**. |

**Why both?**

- **domain.json** = “describe this one domain folder.” Use when you have `legal/`, `hr/` etc.
- **domains.json** = “here are the domain labels for this module when there are no subfolders.” Use when all files sit in the module root.

### 4.2 Other input assets

- **questionnaire_template.json** (or **template.pdf / template.docx**): Questionnaire structure or reference doc; can live in a domain folder or at module level. Same supported formats as document_loader.
- **questionnaire_template.example.json**: Example JSON template for questionnaire fields (copy into domain folders or adapt).

---

## 5. Functional requirements

### 5.1 Knowledge base

- Load documents from a path or from a directory (domain-based or flat).
- Persist vectors in Chroma under `db/`.
- Support multiple collections (one per domain when domain-based).
- QA and similarity search support an optional **domain** (and, when flat, domain as query hint).

### 5.2 Questionnaire module

- Generate a short summary from a filled questionnaire: **what the user agreed to** and **user background**.
- Use module training data (and optional template doc) as context; respect selected domain (collection or query hint).
- Support template as JSON (field list) or as a document (PDF/DOCX/TXT/EPUB).
- “Use template” in UI: load empty template fields for the selected domain so the user can fill and generate.
- **Breakout story generation**: Parse report DOCX Breakout section into themes, questions, topics, and keypoints; optionally parse participant names from the Stories subsection; generate one narrative story per topic (or one per theme when aggregated). Story content is tailored to the document’s theme statement and keypoints (no fixed theme list; each DOCX may have different themes and keypoint structure). Max-word limit and repetition removed in post-processing; (Name) placeholder replaced by parsed participant name when available.

### 5.3 Training data and domains

- **Domain-based**: Subfolders under `input/<module>/`; each subfolder with supported docs is a domain; optional `domain.json` per folder.
- **Flat**: All supported docs in `input/<module>/`; optional `domains.json` to define domain labels for UI and retrieval.
- **Add and sort**: API to add a training text to a domain and optionally save it under `input/<module>/<domain>/<filename>`.

### 5.4 UI

- Chat: upload docs (or load from input), choose domain, ask questions.
- Questionnaire summary: choose domain, optional “Use template”, fill/paste questionnaire, generate summary.
- Load training data from `input/questionnaire/`; domain dropdown reflects loaded domains or `domains.json` when flat.

---

## 6. Non-functional requirements

- **Config**: No hardcoded secrets; use `.env` for API keys.
- **Code**: Modular (e.g. `src/modules/questionnaire/`), type hints, clear separation between KB, input discovery, and modules.
- **Docs**: README (setup, usage, why domain vs domains), `input/README.md`, and this PRD.

---

## 7. References

- [LangChain](https://python.langchain.com/)
- [Chroma](https://www.trychroma.com/)
- [Build Local Knowledge Base Using LLMs](https://cellsummer.github.io/views/2023-07-23-Build-Local_Knowledge-Base-With-LLMs.html)
- Project `README.md` and `input/README.md`
