import hashlib
import shutil
from pathlib import Path

UPLOADS_DIR = Path("data/uploads")


def store_pdf(source_path: str | Path) -> tuple[str, Path]:
    """Copy PDF to uploads dir. Return (doc_id, saved_path)."""
    source = Path(source_path)
    if not source.exists():
        raise FileNotFoundError(f"PDF not found: {source}")
    if source.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a .pdf file, got: {source.suffix}")

    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

    doc_id = hashlib.md5(source.read_bytes()).hexdigest()[:12]
    dest = UPLOADS_DIR / f"{doc_id}_{source.name}"
    if not dest.exists():
        shutil.copy2(source, dest)

    return doc_id, dest
