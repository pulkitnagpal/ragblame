from ragblame.gold import find_gold
from ragblame.types import Chunk, Doc, Hit

LICENSE = ("Docker Desktop is free for small businesses (fewer than 250 employees AND less than $10 million "
           "in annual revenue), personal use, education, and non-commercial open source projects.")
DOCS = [Doc("subscription-billing/desktop-license", "Docker Desktop license", "https://x.test/l", "# License\n\n" + LICENSE),
        Doc("desktop/faq", "FAQ", "https://x.test/f", "# FAQ\n\nDocker Desktop runs on Mac, Windows and Linux.")]


class FakeBot:
    """Just enough of a pipeline for the finder: docs, chunks and an exact search."""
    def __init__(self):
        self.docs = DOCS
        self._chunks = [Chunk(f"{d.id}#0", d.id, 0, len(d.text), d.text) for d in DOCS]

    def chunks(self):
        return list(self._chunks)

    def search(self, query, k, *, exact=False):
        return [Hit(c, 1.0) for c in self._chunks][:k]


def test_finds_quote_and_returns_offsets_into_the_doc():
    reply = {"passage": 1, "quote": "fewer than 250 employees AND less than $10 million in annual revenue",
             "answer": "Only if you have fewer than 250 employees and under $10M revenue."}
    gold, answer = find_gold("Is Docker Desktop free for my 300-person company?", FakeBot(), ask_json=lambda p: reply)
    doc = DOCS[0]
    assert gold.doc_id == doc.id
    assert doc.text[gold.start:gold.end] == "fewer than 250 employees AND less than $10 million in annual revenue"
    assert "250" in answer


def test_none_when_claude_finds_no_answer():
    gold, answer = find_gold("What is Docker's stock price?", FakeBot(),
                             ask_json=lambda p: {"passage": None, "quote": "", "answer": ""})
    assert gold is None


def test_quote_found_in_another_doc_when_passage_number_is_off():
    reply = {"passage": 2, "quote": "less than $10 million in annual revenue", "answer": "under $10M"}
    gold, _ = find_gold("revenue limit?", FakeBot(), ask_json=lambda p: reply)
    assert gold.doc_id == "subscription-billing/desktop-license"


def test_invented_quote_is_rejected():
    reply = {"passage": 1, "quote": "Docker Desktop is free for everyone forever", "answer": "yes"}
    gold, _ = find_gold("is it free?", FakeBot(), ask_json=lambda p: reply)
    assert gold is None


def test_prompt_shows_candidate_passages_numbered():
    seen = {}
    def ask_json(prompt):
        seen["p"] = prompt
        return {"passage": None, "quote": "", "answer": ""}
    find_gold("is it free?", FakeBot(), ask_json=ask_json)
    assert "[1] (subscription-billing/desktop-license)" in seen["p"] and "[2] (desktop/faq)" in seen["p"]
