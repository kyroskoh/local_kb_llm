"""
Discover and describe input training data: modules and domains.
Layout: input/<module>/<domain>/ (e.g. input/questionnaire/legal/).
Each domain folder contains supported doc types (PDF, DOCX, TXT, EPUB)
and optional domain.json, questionnaire_template.json.
"""
import json
from dataclasses import dataclass
from pathlib import Path

from .document_loader import SUPPORTED_EXTENSIONS

# Filename for optional per-domain config (name, description)
DOMAIN_CONFIG_FILENAME = "domain.json"

# Filename for questionnaire template: same format as filled questionnaire (field ids)
QUESTIONNAIRE_TEMPLATE_FILENAME = "questionnaire_template.json"

# Base names for template document (PDF, DOCX, TXT, EPUB); first match wins
TEMPLATE_DOC_BASES = ("template", "questionnaire_template")
# Extensions for template docs (exclude .json)
TEMPLATE_DOC_EXTENSIONS = SUPPORTED_EXTENSIONS

# Optional file at module level to list domain labels when training data is flat
DOMAINS_LIST_FILENAME = "domains.json"


@dataclass
class DomainInfo:
    """Metadata for a single domain (subfolder with training docs)."""

    id: str
    path: Path
    name: str
    description: str
    file_count: int

    @property
    def display_name(self) -> str:
        """Label for UI: custom name or capitalized id."""
        return self.name or self.id.replace("_", " ").title()


def _has_supported_files(directory: Path) -> bool:
    """True if directory contains at least one file with a supported extension."""
    if not directory.is_dir():
        return False
    for p in directory.iterdir():
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
            return True
    return False


def _load_domain_config(domain_path: Path) -> tuple[str, str]:
    """Load optional domain.json; return (name, description)."""
    config_path = domain_path / DOMAIN_CONFIG_FILENAME
    if not config_path.is_file():
        return ("", "")
    try:
        with open(config_path, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return ("", "")
        name = data.get("name", "")
        description = data.get("description", "")
        return (str(name).strip(), str(description).strip())
    except (json.JSONDecodeError, OSError):
        return ("", "")


def discover_domains(input_dir: str | Path) -> list[DomainInfo]:
    """
    Discover domain subfolders under input_dir. Each subfolder that contains
    at least one supported document (PDF, DOCX, TXT, EPUB) is a domain.
    Optionally reads domain.json in each subfolder for name and description.

    Args:
        input_dir: Path to the input root (e.g. input/).

    Returns:
        List of DomainInfo, one per domain subfolder with supported files.
    """
    root = Path(input_dir)
    if not root.is_dir():
        return []

    domains: list[DomainInfo] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        if not _has_supported_files(child):
            continue
        name, description = _load_domain_config(child)
        file_count = sum(
            1
            for p in child.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        domains.append(
            DomainInfo(
                id=child.name,
                path=child,
                name=name,
                description=description,
                file_count=file_count,
            )
        )
    return domains


def list_domain_ids(input_dir: str | Path) -> list[str]:
    """Return ordered list of domain ids under input_dir (convenience)."""
    return [d.id for d in discover_domains(input_dir)]


def discover_modules(input_dir: str | Path) -> list[str]:
    """
    Discover module subfolders under input_dir. Each immediate subdirectory
    is a module (e.g. questionnaire, another_module). Training data for that
    module lives under input/<module>/<domain>/.

    Args:
        input_dir: Path to the input root (e.g. input/).

    Returns:
        List of module names (subfolder names) that exist.
    """
    root = Path(input_dir)
    if not root.is_dir():
        return []
    return sorted(
        child.name
        for child in root.iterdir()
        if child.is_dir() and not child.name.startswith(".")
    )


@dataclass
class QuestionnaireTemplate:
    """Questionnaire template for a domain: same format as the filled questionnaire (field ids)."""

    field_ids: list[str]
    labels: dict[str, str]  # id -> label for UI

    def empty_filled(self) -> dict[str, str]:
        """Return a dict with all template keys and empty strings (for candidate to fill)."""
        return dict.fromkeys(self.field_ids, "")


def load_questionnaire_template(
    input_dir: str | Path, domain_id: str
) -> QuestionnaireTemplate | None:
    """
    Load the questionnaire template for a domain. The template defines the same
    format as the training data for that domain; when a candidate fills in
    those fields, generate_summary can use it.

    Template file: <input_dir>/<domain_id>/questionnaire_template.json
    Format: {"fields": ["name", "role", ...]} or
            {"fields": [{"id": "name", "label": "Full name"}, ...]}

    Returns:
        QuestionnaireTemplate with field_ids and optional labels, or None if missing.
    """
    root = Path(input_dir)
    domain_path = root / domain_id
    template_path = domain_path / QUESTIONNAIRE_TEMPLATE_FILENAME
    if not template_path.is_file():
        return None
    try:
        with open(template_path, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(data, dict) or "fields" not in data:
        return None
    raw = data["fields"]
    if not isinstance(raw, list):
        return None
    field_ids: list[str] = []
    labels: dict[str, str] = {}
    for item in raw:
        if isinstance(item, str):
            field_ids.append(item)
            labels[item] = item.replace("_", " ").title()
        elif isinstance(item, dict) and "id" in item:
            fid = str(item["id"]).strip()
            if fid:
                field_ids.append(fid)
                labels[fid] = str(item.get("label", fid)).strip() or fid.replace("_", " ").title()
    if not field_ids:
        return None
    return QuestionnaireTemplate(field_ids=field_ids, labels=labels)
