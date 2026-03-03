"""
Knowledge base: Chroma vector store + OpenAI embeddings + QA chain.
Supports OpenAI (ChatOpenAI) or a local GGUF model (ChatLlamaCpp) for the LLM.
"""
from pathlib import Path

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
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


def _create_llm(llm_model_path: str | None = None) -> BaseChatModel:
    """Create the chat model: local GGUF if path exists, otherwise OpenAI."""
    path = _resolve_llm_path(llm_model_path)
    if path and Path(path).exists():
        from langchain_community.chat_models.llamacpp import ChatLlamaCpp

        return ChatLlamaCpp(
            model_path=path,
            temperature=0.2,
            n_ctx=2048,
            max_tokens=512,
            verbose=False,
        )
    return ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2)


class KnowledgeBase:
    """Local knowledge base backed by Chroma and OpenAI embeddings."""

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

    def _embeddings(self) -> OpenAIEmbeddings:
        return OpenAIEmbeddings()

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

    def similarity_search(self, query: str, k: int = 4, domain: str = "") -> list[Document]:
        """Return the k most similar document chunks from the given domain's collection."""
        vs = self._get_vectorstore(domain)
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
