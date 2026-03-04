# Local Knowledge Base with LLMs

A local knowledge base application built with [LangChain](https://python.langchain.com/), [Chroma](https://www.trychroma.com/), and OpenAI. Ingest your own documents (PDF, DOCX, TXT, EPUB), then ask questions and get answers grounded in your data—with source citations. The app says "I don't know" when the context doesn't support an answer, reducing hallucination.

Inspired by [Build Local Knowledge Base Using LLMs](https://cellsummer.github.io/views/2023-07-23-Build-Local_Knowledge-Base-With-LLMs.html).

## Features

- **Document loaders**: PDF, DOCX, TXT, EPUB via LangChain
- **Chunking**: Recursive character text splitter (500 chars, 100 overlap)
- **Vector store**: Chroma with persistent storage in `db/`
- **Embeddings**: OpenAI embeddings
- **QA**: **Local GGUF by default** (e.g. Qwen2.5-0.5B-Instruct at `models/Qwen2.5-0.5B-Instruct-GGUF.gguf`); configurable via `LLM_MODEL_PATH` or fallback to OpenAI
- **UI**: NiceGUI chat interface—upload documents, ask questions, and generate questionnaire summaries
- **Training data (input/)**: Layout is **input/<module>/<domain>/** or flat **input/<module>/** with **domains.json**. Questionnaire template can be JSON or **PDF/DOCX/TXT/EPUB** (same as document_loader). Training data can be flat; the app uses the selected domain as a **query hint** for closest-related retrieval. **Generate and sort** new data with `add_training_text` / `save_training_document` or `add_and_save_training_example`.
- **Breakout story generation**: Scripts to extract themes/topics/keypoints from report DOCX (Breakout section) and generate short narrative stories (one per topic or per theme). Stories are tailored to each document’s theme statement and keypoints (document-agnostic; each DOCX can have different themes and keypoint style). When gender is unknown, the participant is referred to as **(Name)** only (no pronouns); when gender is known, pass `participant_pronoun="he"` or `"she"` for consistent third-person pronouns. Stories are based only on theme and keypoints (vectordb/KB is not used; training data elsewhere is for summarization patterns only, not for names or locations). Participant names from the DOCX replace (Name) when available. Max-word limit (default 130); post-processing removes repeated sentences/phrases and strips meta-commentary (e.g. "Critical:", "Opening:"). Output is written to **output/<module_name>/** at project root (e.g. **output/questionnaire/**) as **<input_docx_stem>_<timestamp>.txt**.

## Setup

1. **Clone and enter the project**

   ```bash
   cd local_kb_llm
   ```

2. **Create a virtual environment (recommended)**

   ```bash
   python -m venv .venv
   .venv\Scripts\activate   # Windows
   # source .venv/bin/activate  # macOS/Linux
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   **Optional — local GGUF LLM:** To use a local model (e.g. Qwen GGUF), install `pip install -r requirements-llm.txt`. On Windows, building from source requires [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) (Desktop development with C++); or use Python 3.11/3.12 where a pre-built wheel may be available. You can skip it and set `LLM_MODEL_PATH=openai` to use OpenAI for QA.

4. **Configure environment**

   Copy the example env file:

   ```bash
   copy .env.example .env   # Windows
   # cp .env.example .env   # macOS/Linux
   ```

   To run **without the OpenAI API** (local embeddings + local LLM), set in `.env`:
   - `EMBEDDING_MODEL=local` (uses sentence-transformers; install with `pip install sentence-transformers` if not already in requirements)
   - `LLM_MODEL_PATH` to your local GGUF path or `hf:repo_id`

   To use OpenAI for embeddings or QA, set `OPENAI_API_KEY=sk-your-key-here` and optionally `EMBEDDING_MODEL=openai` or `LLM_MODEL_PATH=openai`. Get a key at [OpenAI API keys](https://platform.openai.com/api-keys).

5. **LLM for QA (configurable)**

   QA uses a **local GGUF model by default**. Place the model at `models/Qwen2.5-0.5B-Instruct-GGUF.gguf` (or set `LLM_MODEL_PATH` in `.env` to another path).

   - **Default**: Download from [Hugging Face – Qwen2.5-0.5B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF) (e.g. `q4_k_m` or `q8_0`), save as `models/Qwen2.5-0.5B-Instruct-GGUF.gguf`. If the file is missing, the app falls back to OpenAI for QA.
   - **Hugging Face (auto-download)**: In `.env`, set `LLM_MODEL_PATH=hf:Qwen/Qwen2.5-0.5B-Instruct-GGUF` or `hf:repo_id:filename.gguf` to use `Llama.from_pretrained()` and `create_chat_completion` (requires `pip install -r requirements-llm.txt`).
   - **Custom path**: In `.env`, set `LLM_MODEL_PATH=/path/to/your.gguf`.
   - **Use OpenAI for QA**: In `.env`, set `LLM_MODEL_PATH=openai` or `LLM_MODEL_PATH=`.

   **Embeddings:** Set `EMBEDDING_MODEL=local` in `.env` to use a local model (sentence-transformers); unset or `openai` uses OpenAI.

## Usage

1. **Run the app**

   From the project root:

   ```bash
   python main.py
   ```

2. **Open the UI**

   In your browser go to: **http://localhost:8080**

3. **Build the knowledge base**

   - **Option A**: Put documents in **input/questionnaire/** per domain (e.g. `input/questionnaire/legal/`, `input/questionnaire/hr/`). See `input/README.md`. Click **Load training data from input/questionnaire/** to index all domains.
   - **Option B**: In the Chat tab, use **Upload document** to add a file (stored as the default domain).

4. **Ask questions (Chat tab)**

   - Choose **Domain** to restrict QA to one domain’s documents (or Default). Type a question and press Enter or click **Send**.

5. **Questionnaire summary**

   - Go to the **Questionnaire summary** tab. Choose **Domain** (the questionnaire template is the same format as that domain’s training data). Click **Use template** to load the domain’s template (empty fields), fill in the candidate’s info, then click **Generate summary**. The app produces **what the user agreed to** and **the user's background**, tailored to the selected domain.

6. **Add more documents**

   - In the Chat tab, after the first upload, an **Add another document** control appears so you can extend the knowledge base.

7. **Breakout story generation (scripts)**

   From the project root (no vectordb used for stories; theme and keypoints only):

   ```bash
   # One story per topic (from Breakout section themes/topics)
   python scripts/test_breakout_stories.py "input/questionnaire/CSS Report_2025_Grp1.docx" [--max N] [--theme N] [--max-words N]

   # One story per theme (topics aggregated per theme)
   python scripts/test_theme_stories.py "input/questionnaire/CSS Report_2025_Grp1.docx" [--max N] [--theme N] [--max-words N]
   ```

   Stories use the same LLM; vectordb/KB is not used—stories are based only on theme and keypoints. Each story is tailored to that document’s theme statement and keypoints (no fixed theme list). When gender is unknown, the participant is referred to as (Name) only; when known, pass `participant_pronoun="he"` or `"she"` to the API for consistent pronouns. Participant names from the DOCX Stories subsection replace (Name) when available. Output is limited to a configurable word count (default 130) and post-processed to remove repeated sentences/phrases and meta-commentary. Results are written to **output/questionnaire/** at project root as **<input_docx_stem>_YYYYMMDD_HHMMSS.txt**. See `PRD.md` for scope.

## Why two example config files in `input/`?

| File | Purpose | When to use |
|------|---------|-------------|
| **domain.example.json** | **Per-domain** metadata: display name and description for **one** domain folder. Copy into each `input/<module>/<domain>/` (e.g. `input/questionnaire/legal/domain.json`). | **Domain-based layout**: you have subfolders like `legal/`, `hr/`; each can have its own `domain.json` so the UI shows "Legal" instead of "legal". |
| **domains.example.json** | **Per-module** list of domain labels. Copy as `input/<module>/domains.json` (e.g. `input/questionnaire/domains.json`). Tells the app which domain options to show when there are **no** domain subfolders. | **Flat layout**: all training files live in the module root (no `legal/`, `hr/` subfolders); the app still offers domain choices and uses them as query hints for closest-related retrieval. |

- Use **domain.example.json** when you organize data by domain folders.
- Use **domains.example.json** when you keep all files in one folder but still want domain-based retrieval.

See `input/README.md` and `PRD.md` for full context.

## Project layout

```
local_kb_llm/
├── main.py              # Entry point (loads .env, runs UI)
├── requirements.txt
├── .env.example
├── README.md
├── PRD.md               # Product requirements
├── input/               # Training data: input/<module>/<domain>/ (e.g. input/questionnaire/legal/)
│   ├── domain.example.json   # Copy into each domain folder (domain-based layout)
│   ├── domains.example.json  # Copy as <module>/domains.json (flat layout)
│   ├── questionnaire/
│   └── another_module/
├── db/                  # Chroma persistence (created at first run)
├── output/              # Generated story output (e.g. output/questionnaire/*.txt)
├── scripts/             # CLI scripts (test_breakout_stories.py, test_theme_stories.py)
└── src/
    ├── __init__.py
    ├── app.py           # NiceGUI chat UI
    ├── document_loader.py  # PDF, DOCX, TXT, EPUB loaders + splitter
    ├── input_domains.py    # Discover modules and domains under input/<module>/
    ├── knowledge_base.py   # Chroma (multi-collection per domain) + embeddings + QA chain
    ├── modules/         # Expertise modules (each: input/<module>/<domain>/)
    │   ├── questionnaire/  # Questionnaire summary; breakout extract + story generation; output at project root output/<module>/
    │   └── another_module/ # Placeholder
    └── prompts.py       # QA-with-sources prompt
```

## Programmatic use

You can use the knowledge base without the UI:

```python
from dotenv import load_dotenv
load_dotenv()

from src.knowledge_base import KnowledgeBase

# Default: local GGUF at models/Qwen2.5-0.5B-Instruct-GGUF.gguf
kb = KnowledgeBase()

# Custom GGUF path
# kb = KnowledgeBase(llm_model_path="./models/my-model.gguf")

# Force OpenAI for QA
# kb = KnowledgeBase(llm_model_path="openai")

# Option A: init from a single file (resets collection)
kb.init_from_path("path/to/document.pdf")

# Option B: add more documents later
kb.add_path("path/to/another.docx")

# Ask a question
import asyncio
answer = asyncio.run(kb.ask("What is the main topic of the document?"))
print(answer)

# Load questionnaire module (domain-based or flat)
from src.modules.questionnaire import generate_summary, add_and_save_training_example
loaded = kb.init_from_directory("input/questionnaire")  # ["legal", "hr"] or [""] if flat
result = asyncio.run(generate_summary(
    {"name": "Jane", "role": "Engineer", "agreed_terms": "yes"},
    kb=kb, domain="legal", domain_display_name="Legal",
))
print(result.agreed_summary)
print(result.background_summary)

# Add new training data and sort to a domain (saves to input/questionnaire/legal/ if filename given)
add_and_save_training_example(
    kb, "input/questionnaire", "legal", "New training text.", source="generated",
    save_filename="generated_2024-01-15.txt",
)
```

## Tech stack

- **LangChain** – document loaders, text splitters, QA chain
- **langchain-chroma** – Chroma vector store integration
- **langchain-openai** – OpenAI embeddings (and optional OpenAI chat model)
- **langchain-community** – ChatLlamaCpp for local GGUF models
- **llama-cpp-python** – local inference for GGUF (e.g. Qwen2.5)
- **Chroma** – vector DB (DuckDB + Parquet persistence)
- **NiceGUI** – web UI

## License

Use and modify as needed for your own projects.
