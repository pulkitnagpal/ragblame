from pathlib import Path

from ragblame.corpus import load_docs

FIXTURES = Path(__file__).parent / "fixtures" / "docs"


def test_load_docs_reads_title_url_text():
    docs = {d.title: d for d in load_docs(FIXTURES)}
    page = docs["Usage and limits"]
    assert page.url.startswith("https://")
    assert page.url == "https://docs.docker.com/docker-hub/usage/"
    assert "Source:" not in page.text
    assert page.text.startswith("# Usage and limits")
    assert page.text == page.text.strip()


def test_load_docs_skips_non_md():
    docs = load_docs(FIXTURES)
    assert len(docs) == 2
    assert all("not a docs page" not in d.text for d in docs)


def test_ids_are_relative_paths_without_extension():
    ids = [d.id for d in load_docs(FIXTURES)]
    assert ids == ["desktop/faq", "docker-hub/usage"]  # forward slashes, sorted, no ".md"


def test_title_falls_back_to_filename(tmp_path):
    (tmp_path / "no-heading.md").write_text("Source: https://example.com/x/\n\nJust text.\n")
    [doc] = load_docs(tmp_path)
    assert doc.title == "no-heading"
    assert doc.text == "Just text."
