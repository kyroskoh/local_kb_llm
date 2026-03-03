"""
Knowledge base: Chroma vector store + configurable embeddings (OpenAI or local) + QA chain.
Supports OpenAI (ChatOpenAI), local GGUF (ChatLlamaCpp), or Hugging Face
(Llama.from_pretrained + create_chat_completion) for the LLM.
"""
import os
from pathlib import Path
from typing import Any

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI

from .document_loader import parse_document, SUPPORTED_EXTENSIONS
from .input_domains import discover_domains
from .prompts import prompt_template

# Default collection and persist directory
DEFAULT_COLLECTION = "local_kb"
DEFAULT_PERSIST_DIR = "db"

# Default directory for training data (input folder)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_DIR = _PROJECT_ROOT / "input"

# Default local LLM: Qwen2.5-0.5B-Instruct GGUF (relative to project root)
DEFAULT_LLM_MODEL_PATH = _PROJECT_ROOT / "models" / "Qwen2.5-0.5B-Instruct-GGUF.gguf"

# Sentinel: set LLM_MODEL_PATH to this to force OpenAI instead of local GGUF
USE_OPENAI = "openai"
OPENAI_QA_MODEL = "gpt-3.5-turbo"

# Prefix for Hugging Face: LLM_MODEL_PATH=hf:repo_id or hf:repo_id:filename
HF_PREFIX = "hf:"

# Default local embedding model (no API key); set EMBEDDING_MODEL=openai to use OpenAI
DEFAULT_LOCAL_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _create_embeddings() -> Embeddings:
    """Create embeddings from env: EMBEDDING_MODEL=openai or unset → OpenAI; else local HuggingFace model."""
    model = (os.environ.get("EMBEDDING_MODEL") or "").strip()
    if model.lower() in ("", "openai"):
        return OpenAIEmbeddings()
    name = DEFAULT_LOCAL_EMBEDDING_MODEL if model.lower() == "local" else model
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(
            model_name=name,
            model_kwargs={"device": "cpu"},
        )
    except ImportError as e:
        raise ImportError(
            "Local embeddings require langchain-huggingface and sentence-transformers. "
            "Install with: pip install langchain-huggingface sentence-transformers"
        ) from e


def _parse_hf_spec(path: str) -> tuple[str, str | None] | None:
    """If path is hf:repo_id or hf:repo_id:filename, return (repo_id, filename or None)."""
    if not path.startswith(HF_PREFIX):
        return None
    rest = path[len(HF_PREFIX) :].strip()
    if ":" in rest:
        repo_id, filename = rest.split(":", 1)
        return repo_id.strip(), filename.strip() or None
    return rest.strip(), None


def _default_gguf_from_repo(repo_id: str) -> str:
    """Pick a GGUF filename from a Hugging Face repo when LLM_MODEL_PATH is hf:repo_id without :filename."""
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        files = api.list_repo_files(repo_id)
        gguf = [f for f in files if f.endswith(".gguf")]
        if not gguf:
            raise RuntimeError(
                f"No .gguf file found in repo {repo_id!r}. "
                "Specify one with LLM_MODEL_PATH=hf:repo_id:filename.gguf"
            )
        # Prefer a small quant (q4_k_m) if present, else first
        preferred = [f for f in gguf if "q4_k_m" in f or "q4k_m" in f]
        return preferred[0] if preferred else gguf[0]
    except Exception as e:
        if isinstance(e, RuntimeError):
            raise
        raise RuntimeError(
            f"Could not list files in repo {repo_id!r}: {e}. "
            "Set LLM_MODEL_PATH=hf:repo_id:filename.gguf with an explicit filename."
        ) from e


def _resolve_llm_path(config: str | None) -> str | None:
    """
    Resolve LLM path from config. Returns path string to use, or None for OpenAI.
    - None or unset → default GGUF path (may not exist)
    - "" or USE_OPENAI → force OpenAI
    - path string → use that path
    """
    if config is not None and config.strip().lower() == USE_OPENAI:
        return None
    if config is not None and config.strip() == "":
        return None
    if config is not None and config.strip():
        return config.strip()
    return str(DEFAULT_LLM_MODEL_PATH)


class _LlamaChatModel(BaseChatModel):
    """LangChain chat model wrapping llama-cpp-python Llama using create_chat_completion."""

    llama_llm: Any  # Llama instance (public name so Pydantic v2 does not treat as private)
    temperature: float = 0.2
    max_tokens: int = 512
    repeat_penalty: float = 1.2  # Reduce repetitive phrasing (e.g. in story generation)

    class Config:
        arbitrary_types_allowed = True

    def _message_to_dict(self, msg: BaseMessage) -> dict[str, str]:
        role = "user"
        if hasattr(msg, "type"):
            if msg.type == "human":
                role = "user"
            elif msg.type == "ai":
                role = "assistant"
            elif msg.type == "system":
                role = "system"
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        return {"role": role, "content": content}

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        openai_messages = [self._message_to_dict(m) for m in messages]
        response = self.llama_llm.create_chat_completion(
            messages=openai_messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            repeat_penalty=self.repeat_penalty,
        )
        content = ""
        if response and "choices" in response and response["choices"]:
            choice = response["choices"][0]
            if "message" in choice and "content" in choice["message"]:
                content = choice["message"]["content"] or ""
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])

    @property
    def _llm_type(self) -> str:
        return "llama_cpp_chat"


def _create_llm(llm_model_path: str | None = None) -> BaseChatModel:
    """Create the chat model: Hugging Face, local GGUF, or OpenAI."""
    path = _resolve_llm_path(llm_model_path)
    if not path:
        return ChatOpenAI(model=OPENAI_QA_MODEL, temperature=0.2)

    hf = _parse_hf_spec(path)
    if hf is not None:
        repo_id, filename = hf
        if not filename:
            filename = _default_gguf_from_repo(repo_id)
        try:
            from llama_cpp import Llama
        except ImportError as e:
            raise RuntimeError(
                "LLM_MODEL_PATH is set to a Hugging Face model (hf:...) but llama-cpp-python is not installed. "
                "Install with: pip install -r requirements-llm.txt"
            ) from e
        try:
            # n_ctx must fit prompt + completion; story prompts can exceed 512 tokens
            llm = Llama.from_pretrained(
                repo_id=repo_id,
                filename=filename,
                n_ctx=2048,
            )
            return _LlamaChatModel(llama_llm=llm, temperature=0.2, max_tokens=512)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load Hugging Face model {repo_id!r} (file {filename!r}): {e}. "
                "Ensure llama-cpp-python is installed (pip install -r requirements-llm.txt) and network is available. "
                "To pick a specific file use: LLM_MODEL_PATH=hf:repo_id:filename.gguf"
            ) from e

    if Path(path).exists():
        from langchain_community.chat_models.llamacpp import ChatLlamaCpp

        return ChatLlamaCpp(
            model_path=path,
            temperature=0.2,
            n_ctx=2048,
            max_tokens=512,
            verbose=False,
        )
    return ChatOpenAI(model=OPENAI_QA_MODEL, temperature=0.2)


class KnowledgeBase:
    """Local knowledge base backed by Chroma and configurable embeddings (OpenAI or local)."""

    def __init__(
        self,
        collection_name: str = DEFAULT_COLLECTION,
        persist_directory: str = DEFAULT_PERSIST_DIR,
        llm_model_path: str | None = None,
    ):
        """
        Args:
            llm_model_path: Path to GGUF file for local QA, or None to use
                default (models/Qwen2.5-0.5B-Instruct-GGUF.gguf). Set to ""
                or "openai" to force OpenAI instead of local.
        """
        self.collection_name = collection_name
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.llm_model_path = llm_model_path
        self._vectorstores: dict[str, Chroma] = {}

    def _client_settings(self) -> chromadb.PersistentClient:
        return chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=chromadb.Settings(anonymized_telemetry=False),
        )

    def _embeddings(self) -> Embeddings:
        return _create_embeddings()

    def _collection_name_for_domain(self, domain: str) -> str:
        """Full Chroma collection name for a domain (empty string = default)."""
        if not domain:
            return self.collection_name
        return f"{self.collection_name}_{domain}"

    def _resolve_domain(self, domain: str) -> str:
        """If domain is '' and exactly one domain is loaded, use it; else return domain."""
        if domain != "":
            return domain
        domains = self.list_domains()
        if len(domains) == 1:
            return domains[0]
        return domain

    def _get_vectorstore(self, domain: str = "") -> Chroma:
        """Get or create the vectorstore for the given domain."""
        domain = self._resolve_domain(domain)
        if domain not in self._vectorstores:
            client = self._client_settings()
            coll = self._collection_name_for_domain(domain)
            self._vectorstores[domain] = Chroma(
                collection_name=coll,
                embedding_function=self._embeddings(),
                client=client,
            )
        return self._vectorstores[domain]

    def list_domains(self) -> list[str]:
        """Return list of domains that have been loaded (have a vectorstore)."""
        return sorted(self._vectorstores.keys())

    def init_from_documents(self, docs: list[Document], domain: str = "") -> Chroma:
        """Create or reset the collection for the given domain and add documents."""
        client = self._client_settings()
        coll = self._collection_name_for_domain(domain)
        try:
            client.delete_collection(coll)
        except Exception:
            pass
        self._vectorstores[domain] = Chroma.from_documents(
            documents=docs,
            embedding=self._embeddings(),
            collection_name=coll,
            client=client,
        )
        return self._vectorstores[domain]

    def init_from_path(self, path: str, doc_type: str | None = None, domain: str = "") -> Chroma:
        """Load a single document from path and initialize the vector store for the domain."""
        docs = parse_document(path, doc_type=doc_type)
        return self.init_from_documents(docs, domain=domain)

    def init_from_directory(
        self, directory: str | Path, domain: str | None = None
    ) -> list[str]:
        """
        Load training data from a directory. Supports domain-focused structure.

        - If domain is None and directory contains subfolders with supported files,
          each subfolder is treated as a domain (e.g. input/legal/, input/hr/).
          Documents are indexed into separate collections per domain.
        - If domain is None and directory contains supported files at top level,
          they are indexed into the default domain ("").
        - If domain is set, only that subfolder (or the directory itself) is
          indexed into the given domain name.

        Returns:
            List of domain ids that were loaded (e.g. ["legal", "hr"] or [""]).
        """
        path = Path(directory)
        if not path.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory}")

        if domain is not None:
            all_docs: list[Document] = []
            for file_path in sorted(path.iterdir()):
                if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                    try:
                        all_docs.extend(parse_document(str(file_path)))
                    except Exception:
                        continue
            if not all_docs:
                raise ValueError(f"No supported documents found in {directory}")
            self.init_from_documents(all_docs, domain=domain)
            return [domain]

        domains_found = discover_domains(path)
        if domains_found:
            loaded: list[str] = []
            for info in domains_found:
                all_docs = []
                for file_path in sorted(info.path.iterdir()):
                    if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                        try:
                            all_docs.extend(parse_document(str(file_path)))
                        except Exception:
                            continue
                if all_docs:
                    self.init_from_documents(all_docs, domain=info.id)
                    loaded.append(info.id)
            if not loaded:
                raise ValueError(f"No supported documents found in any subfolder of {directory}")
            return loaded

        all_docs = []
        for file_path in sorted(path.iterdir()):
            if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                try:
                    all_docs.extend(parse_document(str(file_path)))
                except Exception:
                    continue
        if not all_docs:
            raise ValueError(f"No supported documents found in {directory}")
        self.init_from_documents(all_docs, domain="")
        return [""]

    def add_documents(self, docs: list[Document], domain: str = "") -> None:
        """Add more documents to the existing collection for the given domain."""
        vs = self._get_vectorstore(domain)
        vs.add_documents(docs)

    def add_path(self, path: str, doc_type: str | None = None, domain: str = "") -> None:
        """Load a single document and add it to the collection for the domain."""
        docs = parse_document(path, doc_type=doc_type)
        self.add_documents(docs, domain=domain)

    def add_training_text(
        self, domain: str, text: str, source: str = "generated"
    ) -> None:
        """
        Add a single training text to the given domain's collection (creates
        the collection if needed). Use to sort generated or new data into a domain.
        """
        doc = Document(page_content=text.strip(), metadata={"source": source})
        self.add_documents([doc], domain=domain)

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        domain: str = "",
        domain_as_query_hint: bool = True,
    ) -> list[Document]:
        """
        Return the k most similar document chunks. Uses the domain's collection
        when present; if training data is flat (single collection), uses domain
        as a query prefix to get closest-related content.
        """
        resolved_domain = (
            domain
            if domain and domain in self._vectorstores
            else self._resolve_domain("")
        )
        if domain_as_query_hint and domain and domain != resolved_domain:
            query = f"{domain} {query}".strip()
        vs = self._get_vectorstore(resolved_domain)
        return vs.similarity_search(query=query, k=k)

    async def ask(self, query: str, k: int = 4, domain: str = "") -> str:
        """
        Answer the question using the knowledge base. Returns only the answer string.
        Says "I don't know" when the context does not contain relevant information.
        Uses local GGUF by default (configurable); falls back to OpenAI if file missing.
        When multiple domains are loaded, pass domain= to focus on one.
        """
        from langchain.chains.qa_with_sources import load_qa_with_sources_chain

        relevant_docs = self.similarity_search(query, k=k, domain=domain)
        llm = _create_llm(self.llm_model_path)
        prompt = prompt_template()
        chain = load_qa_with_sources_chain(
            llm=llm,
            chain_type="stuff",
            prompt=prompt,
            verbose=False,
        )
        result = await chain.ainvoke(
            {"input_documents": relevant_docs, "question": query},
        )
        return result.get("output_text", "").strip()
