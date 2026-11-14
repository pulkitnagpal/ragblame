"""Chunking: cutting docs into pieces, and finding where the true answer landed among those pieces."""
from __future__ import annotations

import re
from dataclasses import dataclass

from ragblame.types import Chunk, Doc, Gold


def gold_chunks(chunks: list[Chunk], gold: Gold) -> list[Chunk]:
    """Chunks whose span overlaps the gold span, in document order."""
    return [c for c in chunks if c.doc_id == gold.doc_id and c.start < gold.end and c.end > gold.start]


def is_intact(chunks: list[Chunk], gold: Gold) -> bool:
    """True when some single chunk contains the whole gold span."""
    return any(c.doc_id == gold.doc_id and c.start <= gold.start and c.end >= gold.end for c in chunks)


# --- Fixed-size chunking ---------------------------------------------------------------
#
# The naive default many real bots use: count off N words, cut, step forward, repeat. It
# knows nothing about the document's structure, so a cut can land in the middle of a
# sentence, a list or a table row. The overlap (the last few words of one chunk repeated at
# the start of the next) softens that a little: a fact cut near a boundary may still sit
# whole in one of the two chunks.

_WORD = re.compile(r"\S+")   # a "word" is any run of non-space characters


def fixed_chunks(docs: list[Doc], words: int = 220, overlap: int = 20) -> list[Chunk]:
    """Cut each doc into windows of `words` words, each starting `words - overlap` words after
    the one before. Chunk text is exactly doc.text[start:end], with no context prefix."""
    if not 0 <= overlap < words:
        raise ValueError(f"need 0 <= overlap < words, got overlap={overlap}, words={words}")
    chunks: list[Chunk] = []
    for doc in docs:
        chunks.extend(_fixed_doc_chunks(doc, words, overlap))
    return chunks


def _fixed_doc_chunks(doc: Doc, words: int, overlap: int) -> list[Chunk]:
    """Fixed-size windows over one doc. A doc shorter than `words` is one chunk; an empty doc is none."""
    spans = [m.span() for m in _WORD.finditer(doc.text)]   # (start, end) of every word
    chunks: list[Chunk] = []
    first = 0
    while first < len(spans):
        last = min(first + words, len(spans)) - 1
        start, end = spans[first][0], spans[last][1]
        chunks.append(Chunk(f"{doc.id}#{len(chunks)}", doc.id, start, end, doc.text[start:end]))
        if last == len(spans) - 1:
            break
        first += words - overlap
    return chunks


# --- Heading-aware chunking --------------------------------------------------------------
#
# The usual "fix": follow the document's own structure. Each markdown section becomes a
# chunk, and the chunk text starts with its heading path ("Usage and limits > Pull rate
# limits") so the embedder and the generator know what the chunk is about even when its
# own words don't say. Long sections are split, but only between whole blocks (paragraphs,
# list items, code blocks, table rows), and a piece of a table repeats the table's header
# so its columns still mean something.

_HEADING = re.compile(r"^(#{1,6})[ \t]+(.*?)[ \t#]*$")
_FENCE = re.compile(r"^\s*(```|~~~)")
_LIST_ITEM = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
_TABLE_ROW = re.compile(r"^\s*\|")
_TABLE_SEPARATOR = re.compile(r"^\s*\|?\s*:?-{3,}")


@dataclass
class _Line:
    start: int        # character offsets of the line in the doc, without its newline
    end: int
    text: str
    in_code: bool     # inside a fenced code block, where "# ..." is a comment, not a heading


@dataclass
class _Section:
    path: tuple[str, ...]   # headings from the top level down to this section
    lines: list[_Line]


@dataclass
class _Block:
    """A piece of a section that is never split: a paragraph, list item, code block or table row."""
    start: int
    end: int
    kind: str                 # "heading", "paragraph", "item", "code", "table_header" or "row"
    table: int = 0            # which table of the section a row belongs to (0: not a table)
    words: int = 0


def heading_chunks(docs: list[Doc], max_words: int = 300, overlap: int = 30) -> list[Chunk]:
    """Split each doc at its markdown headings, prefix each chunk with its heading path, and
    split sections longer than `max_words` between whole blocks, overlapping by up to
    `overlap` words of whole blocks."""
    if max_words <= 0 or overlap < 0:
        raise ValueError(f"need max_words > 0 and overlap >= 0, got {max_words}, {overlap}")
    chunks: list[Chunk] = []
    for doc in docs:
        n = 0
        for section in _sections(doc):
            blocks = _blocks(doc, section.lines)
            for piece in _pack(blocks, max_words, overlap):
                start, end = blocks[piece[0]].start, blocks[piece[-1]].end
                text = _prefix(doc, section, blocks, piece) + doc.text[start:end]
                chunks.append(Chunk(f"{doc.id}#{n}", doc.id, start, end, text))
                n += 1
    return chunks


def _lines(text: str) -> list[_Line]:
    """Every line of the text with its offsets, marking the lines of fenced code blocks
    (the fences included)."""
    lines, pos, in_code = [], 0, False
    for raw in text.split("\n"):
        fence = bool(_FENCE.match(raw))
        lines.append(_Line(pos, pos + len(raw), raw, in_code or fence))
        if fence:
            in_code = not in_code
        pos += len(raw) + 1
    return lines


def _heading(line: _Line) -> tuple[int, str] | None:
    """(level, title) when the line is a markdown heading outside code, else None."""
    match = None if line.in_code else _HEADING.match(line.text)
    return (len(match.group(1)), match.group(2)) if match else None


def _sections(doc: Doc) -> list[_Section]:
    """Cut the doc into sections at its headings, each knowing its heading path. A section
    with nothing under its heading (say an H1 straight before an H2) joins the next one."""
    sections = [_Section((), [])]            # the text before the first heading
    stack: list[tuple[int, str]] = []        # open headings: (level, title)
    for line in _lines(doc.text):
        heading = _heading(line)
        if heading:
            # A heading closes every open heading at its level or deeper.
            stack = [h for h in stack if h[0] < heading[0]] + [heading]
            sections.append(_Section(tuple(title for _, title in stack), []))
        sections[-1].lines.append(line)
    merged: list[_Section] = []
    carried: list[_Line] = []
    for i, section in enumerate(sections):
        lines = carried + section.lines
        if _heading_only(section.lines) and i < len(sections) - 1:
            carried = lines
        else:
            carried = []
            merged.append(_Section(section.path, lines))
    return merged


def _heading_only(lines: list[_Line]) -> bool:
    """True when the lines hold nothing but headings and blank lines."""
    return all(not line.text.strip() or _heading(line) for line in lines)


def _blocks(doc: Doc, lines: list[_Line]) -> list[_Block]:
    """Group a section's lines into blocks that are never split, then glue each heading to the
    block after it so no chunk ends on a bare heading."""
    blocks: list[_Block] = []
    text_block: _Block | None = None   # an open paragraph or list item the next line may continue
    code: _Block | None = None         # an open code block, waiting for its closing fence
    header: _Block | None = None       # the header block of the table we are inside, if any
    tables = 0
    for line in lines:
        is_row = not line.in_code and bool(_TABLE_ROW.match(line.text))
        if not is_row:
            header = None
        if line.in_code:
            text_block = None
            if code is None:                       # the opening fence
                code = _Block(line.start, line.end, "code")
                blocks.append(code)
            else:
                code.end = line.end
                if _FENCE.match(line.text):        # the closing fence
                    code = None
        elif is_row:
            text_block = None
            if header is None:                     # a new table: its first row is the header
                tables += 1
                header = _Block(line.start, line.end, "table_header", table=tables)
                blocks.append(header)
            elif header is blocks[-1] and "\n" not in doc.text[header.start:header.end] \
                    and _TABLE_SEPARATOR.match(line.text):
                header.end = line.end              # the |---|---| row stays with the header
            else:
                blocks.append(_Block(line.start, line.end, "row", table=header.table))
        elif not line.text.strip():
            text_block = None
        elif _heading(line):
            blocks.append(_Block(line.start, line.end, "heading"))
            text_block = None
        elif _LIST_ITEM.match(line.text) or text_block is None:
            kind = "item" if _LIST_ITEM.match(line.text) else "paragraph"
            text_block = _Block(line.start, line.end, kind)
            blocks.append(text_block)
        else:
            text_block.end = line.end              # the paragraph or list item carries on
    blocks = _glue_headings(blocks)
    for block in blocks:
        block.words = len(_WORD.findall(doc.text[block.start:block.end]))
    return blocks


def _glue_headings(blocks: list[_Block]) -> list[_Block]:
    """Fold each heading block into the start of the block that follows it."""
    glued: list[_Block] = []
    pending: _Block | None = None      # headings waiting for a block to attach to
    for block in blocks:
        if block.kind == "heading":
            if pending is None:
                pending = block
            else:
                pending.end = block.end
            continue
        if pending is not None:
            block.start, pending = pending.start, None
        glued.append(block)
    if pending is not None:            # a heading with nothing after it stays on its own
        glued.append(pending)
    return glued


def _pack(blocks: list[_Block], max_words: int, overlap: int) -> list[list[int]]:
    """Group consecutive blocks into pieces of at most `max_words` words (a single block that
    is bigger stays whole). Each new piece starts with the previous piece's trailing blocks,
    as many as fit in `overlap` words. Returns the block positions of each piece."""
    pieces: list[list[int]] = []
    current: list[int] = []
    for i, block in enumerate(blocks):
        if current and _words(blocks, current) + block.words > max_words:
            pieces.append(current)
            current = _tail(blocks, current, overlap)
            if _words(blocks, current) + block.words > max_words:
                current = []           # no room for overlap next to this block
        current.append(i)
    if current:
        pieces.append(current)
    return pieces


def _words(blocks: list[_Block], piece: list[int]) -> int:
    """Total words in the blocks of a piece."""
    return sum(blocks[i].words for i in piece)


def _tail(blocks: list[_Block], piece: list[int], overlap: int) -> list[int]:
    """The last whole blocks of a piece whose words add up to no more than `overlap`."""
    tail: list[int] = []
    for i in reversed(piece):
        if _words(blocks, tail) + blocks[i].words > overlap:
            break
        tail.insert(0, i)
    return tail


def _prefix(doc: Doc, section: _Section, blocks: list[_Block], piece: list[int]) -> str:
    """The context put before a chunk's text: its heading path (the doc title before the
    first heading) and, when the piece starts partway through a table, that table's header
    and separator rows."""
    path = " > ".join(section.path) if section.path else doc.title
    prefix = f"{path}\n" if path else ""
    first = blocks[piece[0]]
    if first.kind == "row":            # a table continuation: its header is in an earlier piece
        header = next(b for b in blocks if b.kind == "table_header" and b.table == first.table)
        prefix += doc.text[header.start:header.end] + "\n"
    return prefix


# --- The "perfect chunking" swap ----------------------------------------------------------
#
# Diagnosis asks: would the bot have answered right if chunking hadn't split the answer?
# To test it, the chunks the gold span was split across are merged into one chunk that holds
# the whole span, and everything else stays exactly as the bot had it.

def rechunk_around(chunks: list[Chunk], doc: Doc, gold: Gold) -> list[Chunk]:
    """A copy of `chunks` where the chunks of `doc` that overlap the gold span are merged into
    one chunk covering their union (and the whole span). Every other chunk is unchanged."""
    if doc.id != gold.doc_id:
        raise ValueError(f"gold span is in {gold.doc_id!r}, not {doc.id!r}")
    if is_intact(chunks, gold):
        return chunks
    overlapping = gold_chunks(chunks, gold)
    if not overlapping:
        # No chunk touches the span (it fell in a gap the chunker skipped): add one that holds it.
        extra = Chunk(f"{doc.id}#gold", doc.id, gold.start, gold.end, doc.text[gold.start:gold.end])
        return [*chunks, extra]
    first = overlapping[0]
    start = min(gold.start, *(c.start for c in overlapping))
    end = max(gold.end, *(c.end for c in overlapping))
    merged = Chunk(first.id, doc.id, start, end, _prefix_of(first) + doc.text[start:end])
    result: list[Chunk] = []
    for c in chunks:
        if c is first:
            result.append(merged)
        elif not any(c is o for o in overlapping):
            result.append(c)
    return result


def _prefix_of(chunk: Chunk) -> str:
    """The context prefix of a chunk: the text before its real slice of the doc."""
    return chunk.text[: len(chunk.text) - (chunk.end - chunk.start)]
