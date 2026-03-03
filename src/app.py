"""
NiceGUI chat UI for the local knowledge base and questionnaire summary.
"""
import asyncio
import json
import tempfile
from pathlib import Path

from nicegui import ui

from .input_domains import (
    discover_domain_labels,
    discover_domains,
    load_questionnaire_template,
    load_template_document,
)
from .knowledge_base import (
    DEFAULT_INPUT_DIR,
    KnowledgeBase,
    DEFAULT_PERSIST_DIR,
)
from .modules.questionnaire import INPUT_SUBDIR as QUESTIONNAIRE_INPUT_SUBDIR
from .modules.questionnaire import generate_summary

# Questionnaire module training data: input/questionnaire/<domain>/
QUESTIONNAIRE_INPUT_DIR = DEFAULT_INPUT_DIR / QUESTIONNAIRE_INPUT_SUBDIR
QUESTIONNAIRE_INPUT_DIR.mkdir(parents=True, exist_ok=True)

# Default placeholder when no domain template is loaded
DEFAULT_QUESTIONNAIRE_PLACEHOLDER = '{"name": "", "role": "", "agreed_terms": ""}'


def _domain_select_options(kb: KnowledgeBase) -> list[tuple[str, str]]:
    """Build (label, value) options: Default plus loaded domains, or domain labels from domains.json when flat."""
    options: list[tuple[str, str]] = [("Default", "")]
    infos = discover_domains(QUESTIONNAIRE_INPUT_DIR)
    loaded = kb.list_domains()
    for domain_id in loaded:
        if domain_id == "":
            continue
        label = next((d.display_name for d in infos if d.id == domain_id), domain_id)
        options.append((label, domain_id))
    if not options or (len(loaded) == 1 and loaded[0] == ""):
        for label in discover_domain_labels(QUESTIONNAIRE_INPUT_DIR):
            if not any(v == label for _, v in options):
                options.append((label, label))
    return options


def run_app(
    persist_directory: str = DEFAULT_PERSIST_DIR,
    collection_name: str = "local_kb",
    llm_model_path: str | None = None,
) -> None:
    """Run the NiceGUI chat application."""
    kb = KnowledgeBase(
        collection_name=collection_name,
        persist_directory=persist_directory,
        llm_model_path=llm_model_path,
    )

    @ui.page("/")
    def index():
        ui.label("Local Knowledge Base").classes("text-2xl font-bold mb-4")
        ui.markdown(
            "Ask questions about your documents, or load training data from the input folder and generate questionnaire summaries."
        ).classes("text-gray-600 mb-4")

        domain_select_ref: dict = {}

        def on_load_input():
            try:
                loaded = kb.init_from_directory(QUESTIONNAIRE_INPUT_DIR)
                if loaded == [""]:
                    ui.notify("Training data loaded (default).", type="positive")
                else:
                    ui.notify(f"Loaded domains: {', '.join(loaded)}", type="positive")
                opts = _domain_select_options(kb)
                for sel in domain_select_ref.values():
                    sel.options = opts
                    sel.update()
            except NotADirectoryError as e:
                ui.notify(str(e), type="warning")
            except ValueError as e:
                ui.notify(str(e), type="warning")
            except Exception as e:
                ui.notify(str(e), type="negative")

        with ui.row().classes("gap-2 mb-4"):
                ui.button(
                "Load training data from input/questionnaire/",
                on_click=on_load_input,
            ).props("outline")

        with ui.tabs().classes("w-full") as tabs:
            chat_tab = ui.tab("Chat")
            questionnaire_tab = ui.tab("Questionnaire summary")
        with ui.tab_panels(tabs, value=chat_tab).classes("w-full"):
            with ui.tab_panel(chat_tab):
                upload_area = ui.column().classes("w-full")
                messages = ui.column().classes("w-full gap-2 mt-4")

                def on_upload(e):
                    if not getattr(e, "file", None):
                        return
                    name = e.file.name or "document"
                    suffix = Path(name).suffix.lower().lstrip(".")
                    if suffix not in ("pdf", "docx", "txt", "epub"):
                        ui.notify(
                            "Unsupported format. Use PDF, DOCX, TXT, or EPUB.",
                            type="warning",
                        )
                        return
                    try:
                        with tempfile.NamedTemporaryFile(
                            delete=False, suffix=f".{suffix}"
                        ) as tmp:
                            e.file.save(tmp.name)
                            path = tmp.name
                        with messages:
                            ui.spinner(size="sm")
                            ui.label(f"Loading {name}...")
                        ui.run_coroutine(
                            _load_doc(kb, path, messages, upload_area, name),
                            on_exception=lambda ex: ui.notify(str(ex), type="negative"),
                        )
                    except Exception as ex:
                        ui.notify(str(ex), type="negative")

                def _parse_and_index(kb_instance, path: str, name: str) -> int:
                    from .document_loader import parse_document

                    try:
                        docs = parse_document(path)
                    finally:
                        Path(path).unlink(missing_ok=True)
                    if not docs:
                        return 0
                    try:
                        kb_instance.init_from_documents(docs)
                    except Exception:
                        kb_instance.add_documents(docs)
                    return len(docs)

                async def _load_doc(kb_instance, path: str, msg_container, upload_ui, name: str):
                    count = await asyncio.to_thread(
                        _parse_and_index, kb_instance, path, name
                    )
                    if count == 0:
                        ui.notify("No content extracted from document.", type="warning")
                        return
                    msg_container.clear()
                    upload_ui.clear()
                    ui.notify(
                        f"Loaded {count} chunks from {name}. You can ask questions."
                    )
                    with upload_ui:
                        ui.upload(
                            label="Add another document (PDF, DOCX, TXT, EPUB)",
                            on_upload=on_upload,
                        ).props("accept=.pdf,.docx,.txt,.epub")

                with upload_area:
                    ui.upload(
                        label="Upload document (PDF, DOCX, TXT, EPUB)",
                        on_upload=on_upload,
                    ).props("accept=.pdf,.docx,.txt,.epub")

                chat_domain_select = ui.select(
                    options=_domain_select_options(kb),
                    value="",
                    label="Domain (for QA)",
                ).classes("w-48")
                domain_select_ref["chat"] = chat_domain_select

                async def send_question():
                    q = query_input.value.strip()
                    if not q:
                        return
                    query_input.value = ""
                    with messages:
                        with ui.row().classes("items-start gap-2"):
                            ui.label("You:").classes("font-medium text-blue-600")
                            ui.label(q).classes("flex-1")
                    with messages:
                        with ui.row().classes("items-start gap-2"):
                            ui.label("KB:").classes("font-medium text-green-600")
                            spinner = ui.spinner(size="sm")
                            reply_label = ui.label("...")
                    try:
                        answer = await kb.ask(q, domain=chat_domain_select.value or "")
                        spinner.set_visibility(False)
                        reply_label.set_text(answer)
                    except Exception as ex:
                        spinner.set_visibility(False)
                        reply_label.set_text(f"Error: {ex}")
                        reply_label.classes("text-red-600")

                with ui.row().classes("w-full gap-2 mt-4"):
                    query_input = ui.input(
                        placeholder="Ask a question about your documents...",
                    ).classes("flex-1").on(
                        "keydown.enter", lambda: ui.run_coroutine(send_question())
                    )
                    ui.button("Send", on_click=lambda: ui.run_coroutine(send_question()))

            with ui.tab_panel(questionnaire_tab):
                ui.markdown(
                    "Use **Domain** to match the questionnaire template for that domain (same format as its training data). "
                    "Fill in the template and click **Generate summary**."
                ).classes("text-gray-600 mb-4")
                questionnaire_domain_select = ui.select(
                    options=_domain_select_options(kb),
                    value="",
                    label="Domain",
                ).classes("w-48")
                domain_select_ref["questionnaire"] = questionnaire_domain_select
                questionnaire_ta = ui.textarea(
                    label="Questionnaire (JSON)",
                    placeholder=DEFAULT_QUESTIONNAIRE_PLACEHOLDER,
                ).classes("w-full")

                def on_domain_change():
                    domain_id = questionnaire_domain_select.value or ""
                    if not domain_id:
                        questionnaire_ta.placeholder = DEFAULT_QUESTIONNAIRE_PLACEHOLDER
                        questionnaire_ta.update()
                        return
                    tpl = load_questionnaire_template(QUESTIONNAIRE_INPUT_DIR, domain_id)
                    if tpl:
                        questionnaire_ta.placeholder = json.dumps(tpl.empty_filled(), indent=2)
                    else:
                        questionnaire_ta.placeholder = DEFAULT_QUESTIONNAIRE_PLACEHOLDER
                    questionnaire_ta.update()

                questionnaire_domain_select.on("change", on_domain_change)

                def on_use_template():
                    domain_id = questionnaire_domain_select.value or ""
                    if domain_id:
                        tpl = load_questionnaire_template(QUESTIONNAIRE_INPUT_DIR, domain_id)
                        if tpl:
                            questionnaire_ta.value = json.dumps(tpl.empty_filled(), indent=2)
                            questionnaire_ta.update()
                            ui.notify("Template loaded. Fill in the values.", type="positive")
                            return
                    ui.notify("Select a domain that has a questionnaire_template.json.", type="warning")

                with ui.row().classes("gap-2"):
                    ui.button("Use template", on_click=on_use_template).props("outline")
                summary_result = ui.column().classes("w-full mt-4")

                async def on_generate_summary():
                    raw = questionnaire_ta.value.strip()
                    if not raw:
                        ui.notify("Enter questionnaire JSON.", type="warning")
                        return
                    try:
                        data = json.loads(raw)
                    except json.JSONDecodeError as e:
                        ui.notify(f"Invalid JSON: {e}", type="negative")
                        return
                    if not isinstance(data, dict):
                        ui.notify("JSON must be an object (key-value pairs).", type="warning")
                        return
                    questionnaire = {str(k): str(v) for k, v in data.items()}
                    summary_result.clear()
                    with summary_result:
                        sp = ui.spinner(size="sm")
                        lab = ui.label("Generating summary...")
                    domain_id = questionnaire_domain_select.value or ""
                    domain_label = ""
                    if domain_id:
                        for d in discover_domains(QUESTIONNAIRE_INPUT_DIR):
                            if d.id == domain_id:
                                domain_label = d.display_name
                                break
                        if not domain_label:
                            domain_label = domain_id
                    template_context = load_template_document(
                        QUESTIONNAIRE_INPUT_DIR, domain_id
                    )
                    try:
                        result = await generate_summary(
                            questionnaire,
                            kb=kb,
                            domain=domain_id,
                            domain_display_name=domain_label,
                            template_document_context=template_context,
                            llm_model_path=llm_model_path,
                        )
                        sp.set_visibility(False)
                        lab.set_text("")
                        summary_result.clear()
                        with summary_result:
                            ui.markdown(result.to_display())
                    except Exception as ex:
                        sp.set_visibility(False)
                        lab.set_text("")
                        summary_result.clear()
                        with summary_result:
                            ui.label(f"Error: {ex}").classes("text-red-600")

                ui.button(
                    "Generate summary",
                    on_click=lambda: ui.run_coroutine(on_generate_summary()),
                )

    ui.run(
        title="Local Knowledge Base",
        reload=False,
        port=8080,
    )
