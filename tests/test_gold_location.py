from ragblame.chunking import gold_chunks, is_intact
from ragblame.types import Chunk, Gold

TEXT = "Personal accounts get 200 pulls per 6 hours. Teams get unlimited pulls."


def chunk(i, start, end, doc="hub/usage"):
    return Chunk(f"{doc}#{i}", doc, start, end, TEXT[start:end])


def test_span_inside_one_chunk_is_intact():
    chunks = [chunk(0, 0, 45), chunk(1, 45, len(TEXT))]
    gold = Gold("hub/usage", 0, 44, TEXT[0:44])
    assert is_intact(chunks, gold)
    assert [c.id for c in gold_chunks(chunks, gold)] == ["hub/usage#0"]


def test_span_across_a_boundary_is_split():
    chunks = [chunk(0, 0, 30), chunk(1, 30, len(TEXT))]
    gold = Gold("hub/usage", 0, 44, TEXT[0:44])
    assert not is_intact(chunks, gold)
    assert [c.id for c in gold_chunks(chunks, gold)] == ["hub/usage#0", "hub/usage#1"]


def test_other_docs_never_count_as_gold():
    chunks = [chunk(0, 0, len(TEXT), doc="desktop/faq")]
    gold = Gold("hub/usage", 0, 44, TEXT[0:44])
    assert gold_chunks(chunks, gold) == []
    assert not is_intact(chunks, gold)
