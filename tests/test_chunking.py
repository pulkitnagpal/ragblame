import re

from ragblame.chunking import fixed_chunks, gold_chunks, heading_chunks, is_intact, rechunk_around
from ragblame.types import Doc, Gold

HEADER = "| Plan | Pull limit | Private repos |"
SEPARATOR = "|------|------------|---------------|"
PLANS = ["Personal", "Pro", "Team", "Business", "Starter", "Hobby",
         "Student", "Open Source", "Verified", "Sponsored", "Legacy", "Enterprise"]
ROWS = [f"| {p} | {(i + 1) * 100} pulls per 6 hours for each authenticated user | {i + 1} private repos |"
        for i, p in enumerate(PLANS)]

PAGE = "\n".join([
    "Docker Hub has limits on how often you can pull images.",
    "",
    "# Usage and limits",
    "",
    "## Pull rate limits",
    "",
    "Docker Hub counts every pull of an image manifest. Limits depend on your plan and on",
    "whether you are signed in when you pull.",
    "",
    "The table below lists the limit for each plan.",
    "",
    HEADER,
    SEPARATOR,
    *ROWS,
    "",
    "## Checking your usage",
    "",
    "You can see your current usage in three ways:",
    "",
    "- Open the usage page in Docker Hub and pick the namespace you care about.",
    "- Read the `ratelimit-remaining` header on any manifest request you send.",
    "- Ask your organization owner for the monthly report.",
    "",
    "```bash",
    "# this is a shell comment, not a heading",
    "curl -I https://registry-1.docker.io/v2/library/alpine/manifests/latest",
    "```",
    "",
    "### Abuse rate limit",
    "",
    "Separately from pull limits, Docker Hub may block clients that send too many requests",
    "in a short time. Wait a few minutes and try again.",
])
HUB = Doc("hub/usage", "Usage and limits", "https://docs.docker.com/docker-hub/usage/", PAGE)


def numbered_doc(doc_id, n_words):
    """A doc whose words are all different (w0 w1 w2 ...), so overlaps can be counted exactly."""
    return Doc(doc_id, doc_id, "", " ".join(f"w{i}" for i in range(n_words)))


def words(text):
    return re.findall(r"\S+", text)


def prefix_of(chunk, doc):
    """The context prefix: whatever the chunk text has before the real slice of the doc."""
    body = doc.text[chunk.start:chunk.end]
    assert chunk.text.endswith(body)
    return chunk.text[: len(chunk.text) - len(body)]


def line_spans(text, predicate):
    """(start, end) of every line of text for which predicate(line) is true."""
    spans, pos = [], 0
    for line in text.split("\n"):
        if predicate(line):
            spans.append((pos, pos + len(line)))
        pos += len(line) + 1
    return spans


# --- fixed_chunks ---

def test_fixed_chunk_text_matches_offsets():
    docs = [numbered_doc("a", 1000), HUB, Doc("empty", "", "", "")]
    chunks = fixed_chunks(docs)
    by_id = {d.id: d for d in docs}
    assert chunks
    for c in chunks:
        assert c.text == by_id[c.doc_id].text[c.start:c.end]
    assert [c.id for c in chunks if c.doc_id == "a"] == [f"a#{n}" for n in range(5)]
    assert not [c for c in chunks if c.doc_id == "empty"]


def test_fixed_chunks_overlap_exactly():
    chunks = fixed_chunks([numbered_doc("a", 1000)], words=220, overlap=20)
    assert len(chunks) > 2
    for a, b in zip(chunks, chunks[1:]):
        shared = set(words(a.text)) & set(words(b.text))
        assert len(shared) == 20
        assert words(a.text)[-20:] == words(b.text)[:20]
    assert all(len(words(c.text)) == 220 for c in chunks[:-1])
    assert words(chunks[-1].text)[-1] == "w999"


def test_short_doc_is_one_chunk():
    doc = Doc("short", "Short", "", "  Only a few words here.\n")
    chunks = fixed_chunks([doc])
    assert len(chunks) == 1
    assert chunks[0].id == "short#0"
    assert chunks[0].text == "Only a few words here."


# --- heading_chunks ---

def test_heading_chunks_prefix_heading_path():
    chunks = heading_chunks([HUB])
    prefixes = [prefix_of(c, HUB) for c in chunks]
    assert "Usage and limits > Pull rate limits\n" in prefixes
    assert "Usage and limits > Checking your usage > Abuse rate limit\n" in prefixes
    # The "# this is a shell comment" line sits in a code block, so it is not a heading.
    assert not any("shell comment" in p for p in prefixes)
    # Every chunk starts with its heading path.
    pull = next(c for c in chunks if "Personal |" in c.text)
    assert pull.text.startswith("Usage and limits > Pull rate limits\n")


def test_heading_chunks_never_split_table_rows():
    for max_words in (25, 40, 80, 300):
        chunks = heading_chunks([HUB], max_words=max_words, overlap=8)
        rows = line_spans(PAGE, lambda line: line.startswith("|"))
        for c in chunks:
            for start, end in rows:
                assert not start < c.start < end, (max_words, c.id)
                assert not start < c.end < end, (max_words, c.id)
        if max_words < 300:
            table_chunks = [c for c in chunks if any(r in c.text for r in ROWS)]
            assert len(table_chunks) > 1  # the table really was split


def test_heading_table_continuation_repeats_header():
    chunks = heading_chunks([HUB], max_words=40, overlap=20)
    table_chunks = [c for c in chunks if any(r in HUB.text[c.start:c.end] for r in ROWS)]
    assert len(table_chunks) >= 3
    for c in table_chunks:
        # Each piece of the table says what its columns mean, from the slice or from the prefix.
        assert c.text.startswith("Usage and limits > Pull rate limits\n")
        assert f"{HEADER}\n{SEPARATOR}\n" in c.text
    continuations = [c for c in table_chunks if HEADER not in HUB.text[c.start:c.end]]
    assert continuations
    for c in continuations:
        assert prefix_of(c, HUB) == f"Usage and limits > Pull rate limits\n{HEADER}\n{SEPARATOR}\n"
    # The overlap between table pieces is whole rows.
    first, second = table_chunks[0], table_chunks[1]
    assert second.start < first.end


def test_heading_chunks_text_endswith_slice():
    docs = [HUB, numbered_doc("plain", 700), Doc("empty", "", "", "\n\n")]
    by_id = {d.id: d for d in docs}
    for max_words, overlap in ((300, 30), (40, 8), (12, 4)):
        chunks = heading_chunks(docs, max_words=max_words, overlap=overlap)
        assert not [c for c in chunks if c.doc_id == "empty"]
        for c in chunks:
            doc = by_id[c.doc_id]
            assert 0 <= c.start < c.end <= len(doc.text)
            assert c.text.endswith(doc.text[c.start:c.end])
        # Together the chunks cover every word of the doc.
        hub = [c for c in chunks if c.doc_id == "hub/usage"]
        covered = set()
        for c in hub:
            covered.update(range(c.start, c.end))
        assert all(i in covered for i, ch in enumerate(PAGE) if not ch.isspace())


# --- rechunk_around ---

def test_rechunk_around_makes_span_intact():
    doc = numbered_doc("a", 1000)
    chunks = fixed_chunks([doc])
    # A span that starts before chunk 1's overlap ends and runs past chunk 2's start:
    # chunks 0, 1 and 2 all overlap it.
    start = doc.text.index("w190")
    end = doc.text.index("w410") + len("w410")
    gold = Gold("a", start, end, doc.text[start:end])
    assert not is_intact(chunks, gold)
    assert len(gold_chunks(chunks, gold)) == 3
    new = rechunk_around(chunks, doc, gold)
    assert is_intact(new, gold)
    merged = gold_chunks(new, gold)
    assert len(merged) == 1
    m = merged[0]
    assert m.id == "a#0"
    assert (m.start, m.end) == (chunks[0].start, chunks[2].end)
    assert m.text == doc.text[m.start:m.end]
    # The same doc's chunks away from the span are unchanged.
    assert new[1:] == chunks[3:]
    # An intact span leaves the list as it was.
    inside = Gold("a", chunks[4].start, chunks[4].start + 10, "")
    assert rechunk_around(chunks, doc, inside) == chunks


def test_rechunk_around_keeps_prefix_of_first_chunk():
    chunks = heading_chunks([HUB], max_words=40, overlap=8)
    start = PAGE.index(ROWS[0])
    end = PAGE.index(ROWS[-1]) + len(ROWS[-1])
    gold = Gold("hub/usage", start, end, PAGE[start:end])
    assert not is_intact(chunks, gold)
    first = gold_chunks(chunks, gold)[0]
    new = rechunk_around(chunks, HUB, gold)
    assert is_intact(new, gold)
    m = gold_chunks(new, gold)[0]
    assert m.id == first.id
    assert m.text == prefix_of(first, HUB) + PAGE[m.start:m.end]


def test_rechunk_around_leaves_other_docs_untouched():
    a, b = numbered_doc("a", 600), numbered_doc("b", 600)
    chunks = fixed_chunks([a, b])
    start = a.text.index("w195")
    end = a.text.index("w230") + len("w230")
    gold = Gold("a", start, end, a.text[start:end])
    new = rechunk_around(chunks, a, gold)
    assert [c for c in new if c.doc_id == "b"] == [c for c in chunks if c.doc_id == "b"]
    assert is_intact(new, gold)


# --- gold location and ids ---

def test_gold_in_two_docs_found_in_both():
    sentence = "Personal accounts get 200 pulls per 6 hours."
    one = Doc("hub/usage", "Usage", "", "Intro words here. " * 30 + sentence + " Trailing text." * 10)
    two = Doc("faq/limits", "FAQ", "", sentence + " Then more text follows here." * 50)
    for chunker in (fixed_chunks, heading_chunks):
        chunks = chunker([one, two])
        for doc in (one, two):
            start = doc.text.index(sentence)
            gold = Gold(doc.id, start, start + len(sentence), sentence)
            found = gold_chunks(chunks, gold)
            assert found
            assert all(c.doc_id == doc.id for c in found)
            assert any(sentence in c.text for c in found)


def test_ids_unique():
    docs = [HUB, numbered_doc("a", 900), numbered_doc("b", 50)]
    for chunks in (fixed_chunks(docs), heading_chunks(docs), heading_chunks(docs, max_words=30, overlap=5)):
        ids = [c.id for c in chunks]
        assert len(ids) == len(set(ids))
        for doc in docs:
            mine = [c.id for c in chunks if c.doc_id == doc.id]
            assert mine == [f"{doc.id}#{n}" for n in range(len(mine))]
