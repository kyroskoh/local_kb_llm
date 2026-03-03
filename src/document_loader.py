"""
Document loaders and text splitting for the local knowledge base.
Supports PDF, DOCX, TXT, and EPUB via LangChain loaders.
"""
from pathlib import Path

# Extensions supported by parse_document (lowercase with leading dot)
SUPPORTED_EXTENSIONS = (".pdf", ".docx", ".txt", ".epub")

from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredEPubLoader,
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter


def _default_text_splitter() -> TextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )


def parse_pdf(path: str, text_splitter: TextSplitter | None = None) -> list[Document]:
    """Load and chunk a PDF file."""
    splitter = text_splitter or _default_text_splitter()
    loader = PyPDFLoader(path)
    pages = loader.load()
    return splitter.split_documents(pages)


def parse_docx(path: str, text_splitter: TextSplitter | None = None) -> list[Document]:
    """Load and chunk a DOCX file."""
    splitter = text_splitter or _default_text_splitter()
    loader = Docx2txtLoader(path)
    docs = loader.load()
    return splitter.split_documents(docs)


def parse_txt(path: str, text_splitter: TextSplitter | None = None) -> list[Document]:
    """Load and chunk a plain text file."""
    splitter = text_splitter or _default_text_splitter()
    loader = TextLoader(path, encoding="utf-8", autodetect_encoding=True)
    docs = loader.load()
    return splitter.split_documents(docs)


def parse_epub(path: str, text_splitter: TextSplitter | None = None) -> list[Document]:
    """Load and chunk an EPUB file."""
    splitter = text_splitter or _default_text_splitter()
    loader = UnstructuredEPubLoader(path)
    docs = loader.load()
    return splitter.split_documents(docs)


def parse_document(path: str, doc_type: str | None = None) -> list[Document]:
    """
    Parse a document by path and optional type; type is inferred from suffix if omitted.
    Supported: pdf, docx, txt, epub.
    """
    suffix = Path(path).suffix.lower().lstrip(".")
    kind = (doc_type or suffix).lower()

    parsers = {
        "pdf": parse_pdf,
        "docx": parse_docx,
        "txt": parse_txt,
        "epub": parse_epub,
    }
    if kind not in parsers:
        raise NotImplementedError(
            f"Document type '{kind}' is not supported. Use one of: {list(parsers)}"
        )
    return parsers[kind](path)
