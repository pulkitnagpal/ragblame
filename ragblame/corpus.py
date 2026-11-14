"""Load the fetched docs pages from disk.

In RAG, the *corpus* is the set of documents the bot is allowed to search. Each
page becomes one Doc; later stages cut Docs into chunks, embed the chunks and
index them so a question can find the right passage.
"""
from __future__ import annotations

from pathlib import Path

from ragblame.types import Doc

SOURCE_PREFIX = "Source: "


def split_source(raw: str) -> tuple[str, str]:
    """Split a saved page into (url, body). The fetch script puts 'Source: <url>' on line 1."""
    first, _, rest = raw.partition("\n")
    if first.startswith(SOURCE_PREFIX):
        return first[len(SOURCE_PREFIX):].strip(), rest
    return "", raw


def first_heading(text: str, fallback: str) -> str:
    """The first '# ' heading in the page, or the fallback if there isn't one."""
    for line in text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return fallback


def doc_id(path: Path, root: Path) -> str:
    """The page's path relative to the corpus root, without '.md', with forward slashes."""
    return path.relative_to(root).with_suffix("").as_posix()


def load_doc(path: Path, root: Path) -> Doc:
    """Read one .md page into a Doc."""
    url, body = split_source(path.read_text(encoding="utf-8"))
    text = body.strip()
    return Doc(id=doc_id(path, root), title=first_heading(text, path.stem), url=url, text=text)


def load_docs(root: Path) -> list[Doc]:
    """Every .md page under root, sorted by id. Other files (LICENSE, NOTICE, .txt) are ignored."""
    root = Path(root)
    docs = [load_doc(path, root) for path in root.rglob("*.md") if path.is_file()]
    return sorted(docs, key=lambda d: d.id)
