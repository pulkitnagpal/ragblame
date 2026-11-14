import threading

from ragblame import questions as qmod
from ragblame.questions import find_quote, load_questions, save_questions, write_questions
from ragblame.types import Doc, Gold, Question

TABLE_PAGE = """# Usage and limits

Docker Hub limits the number of pulls.

| Plan     | Pulls per hour |  Notes   |
|----------|---------------:|----------|
| Personal | 100            | per IP   |
| Pro      | **Unlimited**  | fair use |

After the limit, pulls fail with a `429` error.
"""


def span(text, quote):
    r = find_quote(text, quote)
    return None if r is None else text[r[0]:r[1]]


def test_find_quote_exact():
    text = "Intro. Personal accounts get 200 pulls per 6 hours. Teams get more."
    s, e = find_quote(text, "Personal accounts get 200 pulls per 6 hours.")
    assert (s, e) == (7, 51)
    assert text[s:e] == "Personal accounts get 200 pulls per 6 hours."


def test_find_quote_tolerates_whitespace_and_pipes():
    # extra whitespace and line breaks in the page
    text = "The daemon   listens on\n  port 2375\tby default."
    assert span(text, "The daemon listens on port 2375 by default.") == text
    # a table row quoted without pipes, or with different padding
    assert span(TABLE_PAGE, "Personal 100 per IP") == "Personal | 100            | per IP"
    assert span(TABLE_PAGE, "|Personal|100|per IP|") == "Personal | 100            | per IP"
    # header row plus data row, with or without the separator line
    got = span(TABLE_PAGE, "| Plan | Pulls per hour | Notes |\n| Personal | 100 | per IP |")
    assert got.startswith("Plan") and got.endswith("per IP")
    got = span(TABLE_PAGE, "| Plan | Pulls per hour | Notes |\n|---|---|---|\n| Personal | 100 | per IP |")
    assert got.startswith("Plan") and got.endswith("per IP")
    # header row plus a row that is not directly under it: the span is that row
    assert span(TABLE_PAGE, "| Plan | Pulls per hour | Notes |\n| Pro | Unlimited | fair use |") == \
        "Pro      | **Unlimited**  | fair use"


def test_find_quote_ignores_bold_and_code_markers():
    assert span(TABLE_PAGE, "Pro | Unlimited | fair use") == "Pro      | **Unlimited**  | fair use"
    assert span(TABLE_PAGE, "pulls fail with a 429 error") == "pulls fail with a `429` error"


def test_find_quote_smart_quotes():
    text = "Set “max-size” to 10m — the daemon’s default is unlimited."
    got = span(text, 'Set "max-size" to 10m - the daemon\'s default is unlimited.')
    assert got == text
    # and the other way round
    assert span('It\'s "on" by default -- see below', "It’s “on” by default") == 'It\'s "on" by default'


def test_find_quote_returns_original_offsets():
    text = "Intro  **line**\n\n| A | B |\n|---|---|\n| x  |  **y** |\nTail: it’s   done."
    s, e = find_quote(text, "it's done.")
    assert text[s:e] == "it’s   done."
    s, e = find_quote(text, "x y")
    assert text[s:e] == "x  |  **y"
    assert find_quote(text, "not in the page") is None
    assert find_quote(text, "   ") is None


def test_find_quote_ignores_link_markup():
    text = "Read the [install guide](https://docs.docker.com/install/) first."
    assert span(text, "Read the install guide first.") == text


def doc(i, words, extra=""):
    body = " ".join(f"w{i}x{k}" for k in range(words))
    return Doc(f"page{i}", f"Page {i}", f"https://docs.example/page{i}/", f"{extra} {body}".strip())


def test_unmatched_quote_rejected_and_counted():
    d = Doc("hub/usage", "Usage", "https://u/1", "Personal accounts get 200 pulls per 6 hours. " + "filler " * 100)
    other = Doc("hub/other", "Other", "https://u/2", "Other page. " + "padding " * 100)

    def fake(prompt):
        if "https://u/2" in prompt:
            return {"questions": [{"question": "bad", "answer": "no quote field"}, "not a dict"]}
        return {"questions": [
            {"question": "How many pulls do I get?", "answer": "200 per 6 hours",
             "quote": "Personal accounts get 200 pulls per 6 hours."},
            {"question": "Is it free?", "answer": "yes", "quote": "Docker Hub is free forever."},
        ]}

    qs, rejected = write_questions([d, other], n=10, ask_json=fake)
    assert rejected == 3
    assert len(qs) == 1
    q = qs[0]
    assert q.id == "q001"
    assert q.text == "How many pulls do I get?" and q.answer == "200 per 6 hours"
    assert q.gold == Gold("hub/usage", 0, 44, d.text[0:44])


def test_one_failed_call_does_not_stop_the_run():
    from ragblame.llm import LLMError
    docs = [doc(i, 120) for i in range(4)]

    def fake(prompt):
        if "page2/" in prompt:
            raise LLMError("claude failed")
        page = next(d for d in docs if d.url in prompt)
        return {"questions": [{"question": "q", "answer": "a", "quote": page.text.split()[3]}]}

    qs, rejected = write_questions(docs, n=10, ask_json=fake)
    assert len(qs) == 3 and rejected == 0


def test_write_questions_stops_at_n():
    docs = [doc(i, 100 + 10 * i) for i in range(30)] + [doc(99, 20)]  # page99 is too short
    seen = []
    lock = threading.Lock()

    def fake(prompt):
        page = next(d for d in docs if d.url in prompt)
        with lock:
            seen.append(page.id)
        words = page.text.split()
        return {"questions": [
            {"question": f"q about {page.id} 1", "answer": "a", "quote": " ".join(words[0:5])},
            {"question": f"q about {page.id} 2", "answer": "b", "quote": " ".join(words[10:15])},
        ]}

    qs, rejected = write_questions(docs, n=7, seed=7, ask_json=fake, workers=2)
    assert len(qs) == 7
    assert [q.id for q in qs] == [f"q{i:03d}" for i in range(1, 8)]
    assert rejected == 0
    assert "page99" not in seen
    assert len(seen) < 30                          # stopped early instead of asking about every page
    assert len({q.gold.doc_id for q in qs}) == 4   # two per page, without replacement
    for q in qs:
        d = next(x for x in docs if x.id == q.gold.doc_id)
        assert d.text[q.gold.start:q.gold.end] == q.gold.quote
    # deterministic for a given seed
    again, _ = write_questions(docs, n=7, seed=7, ask_json=fake, workers=2)
    assert [q.text for q in again] == [q.text for q in qs]


def test_prompt_truncates_long_pages():
    long = Doc("big", "Big", "https://docs.example/big", " ".join(f"word{k}" for k in range(5000)))
    p = qmod.build_prompt(long)
    assert "word2999" in p and "word3000 " not in p and "word4999" not in p
    assert "https://docs.example/big" in p


def test_jsonl_round_trip(tmp_path):
    qs = [Question("q001", "How do I “prune”?", "docker system prune", Gold("a/b", 3, 9, "prune")),
          Question("q002", "Unanswerable?", "n/a", None)]
    path = tmp_path / "sub" / "questions.jsonl"
    save_questions(qs, path)
    assert len(path.read_text().splitlines()) == 2
    assert load_questions(path) == qs
