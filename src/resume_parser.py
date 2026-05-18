"""Utilities for loading real-world resume files."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

from pypdf import PdfReader


SUPPORTED_RESUME_EXTENSIONS = {".pdf", ".txt"}


def extract_pdf_text(path: Path) -> str:
    """Extract text from a PDF resume."""
    reader = PdfReader(str(path))
    page_text = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(page_text).strip()


def extract_pdf_text_from_bytes(data: bytes) -> str:
    """Extract text from an uploaded PDF resume."""
    reader = PdfReader(BytesIO(data))
    page_text = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(page_text).strip()


def extract_resume_text(path: Path) -> str:
    """Extract text from a supported resume file."""
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return extract_pdf_text(path)
    if suffix == ".txt":
        return path.read_text(encoding="utf-8")

    raise ValueError(f"Unsupported resume type: {path.suffix}")


def extract_resume_text_from_bytes(filename: str, data: bytes) -> str:
    """Extract text from an uploaded resume by filename."""
    suffix = Path(filename).suffix.lower()

    if suffix == ".pdf":
        return extract_pdf_text_from_bytes(data)
    if suffix == ".txt":
        return data.decode("utf-8", errors="ignore")

    raise ValueError(f"Unsupported resume type: {suffix}")


def collect_resume_paths(paths: list[Path]) -> list[Path]:
    """Expand input files/directories into supported resume file paths."""
    resume_paths: list[Path] = []

    for path in paths:
        if path.is_dir():
            for extension in SUPPORTED_RESUME_EXTENSIONS:
                resume_paths.extend(path.rglob(f"*{extension}"))
        elif path.is_file() and path.suffix.lower() in SUPPORTED_RESUME_EXTENSIONS:
            resume_paths.append(path)
        else:
            raise ValueError(f"Resume path is not a supported file or directory: {path}")

    return sorted(set(resume_paths))
