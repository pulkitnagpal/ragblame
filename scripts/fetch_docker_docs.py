"""Fetch the support-style subset of the Docker docs into data/docker/.

This is the *corpus* for the demo bot: the pile of documents a RAG system
searches before it answers. A support bot is only as good as the pages it can
find, so we pick the pages a Docker help desk would lean on (Desktop, Hub,
accounts, billing, security, FAQs) and leave out noise like API references
and release notes.

Usage:
    python scripts/fetch_docker_docs.py            # writes data/docker/
    python scripts/fetch_docker_docs.py --dry-run  # only lists the chosen pages

Only the standard library is used, so it runs with any Python 3.12+.
"""
from __future__ import annotations

import argparse
import re
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlparse

BASE = "https://docs.docker.com"
USER_AGENT = "ragblame/0.1 (+https://github.com/pulkitnagpal/ragblame)"
DELAY_SECONDS = 0.35  # be polite: at least 0.3 s between requests
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "docker"

# Top-level sections a support bot needs. The first seven are the help-centre core;
# get-started, scout, compose and the user-facing extensions pages bring the corpus
# to a realistic size (a few hundred pages, a few thousand chunks).
SECTIONS = (
    "desktop", "docker-hub", "accounts", "subscription-billing", "security", "support", "faqs",
    "get-started", "scout", "compose", "extensions",
)

# Pages that are noise for a support bot: long changelogs, API/SDK references and
# security bulletins. They would flood retrieval with near-duplicate version text.
EXCLUDE = (
    re.compile(r"/release-notes/"),
    re.compile(r"/security/security-announcements/"),
    re.compile(r"/extensions/extensions-sdk/"),  # developer SDK, not end-user help
    re.compile(r"/reference/api/"),
)

# Hugo shortcodes such as {{< tabs >}} or {{% include "x.md" %}}. The rendered .md pages
# resolve most of them, but any that slip through are template junk, not content.
SHORTCODE = re.compile(r"\{\{[<%].*?[%>]\}\}", re.DOTALL)


def fetch_text(url: str) -> str:
    """GET a URL and return its body as text. Raises on HTTP errors or HTML error pages."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=30) as response:
        body = response.read().decode("utf-8")
    if body.lstrip().lower().startswith(("<!doctype html", "<html")):
        raise ValueError("got an HTML page, not markdown")
    return body


def urls_from_llms_full(text: str) -> list[str]:
    """Page URLs from llms-full.txt, which lists each page as a 'URL: ...' line."""
    return re.findall(r"^URL:\s*(\S+)", text, flags=re.MULTILINE)


def urls_from_sitemap(xml: str) -> list[str]:
    """Page URLs from sitemap.xml's <loc> tags."""
    return re.findall(r"<loc>\s*([^<\s]+)\s*</loc>", xml)


def page_index() -> list[str]:
    """Every docs page URL we know about.

    llms.txt is now a short discovery guide that points at llms-full.txt for the full
    page index. llms-full.txt skips section landing pages (e.g. /docker-hub/usage/,
    which holds the pull-limit table), so we also merge in sitemap.xml.
    """
    guide = fetch_text(f"{BASE}/llms.txt")
    full_link = re.search(r"https://docs\.docker\.com/llms-full\.txt", guide)
    urls = set(re.findall(r"\((https://docs\.docker\.com/[^)\s]*)\)", guide))
    if full_link:
        time.sleep(DELAY_SECONDS)
        urls.update(urls_from_llms_full(fetch_text(full_link.group(0))))
    time.sleep(DELAY_SECONDS)
    urls.update(urls_from_sitemap(fetch_text(f"{BASE}/sitemap.xml")))
    return sorted(u for u in urls if u.startswith(BASE + "/") and u.endswith("/"))


def section_of(url: str) -> str:
    """The first path segment, e.g. 'docker-hub' for /docker-hub/usage/."""
    return urlparse(url).path.strip("/").split("/")[0]


def keep(url: str) -> bool:
    """True if the page belongs in the support-bot subset."""
    path = urlparse(url).path
    if not path.strip("/") or section_of(url) not in SECTIONS:
        return False
    return not any(pattern.search(path) for pattern in EXCLUDE)


def markdown_url(url: str) -> str:
    """docs.docker.com serves rendered markdown at <page-url>.md (shortcodes resolved)."""
    return url.rstrip("/") + ".md"


def local_path(url: str, out_dir: Path) -> Path:
    """data/docker/<url-path>.md, e.g. data/docker/docker-hub/usage.md."""
    return out_dir / (urlparse(url).path.strip("/") + ".md")


def clean(markdown: str) -> str:
    """Remove leftover Hugo shortcodes and trailing whitespace; collapse big blank gaps."""
    text = SHORTCODE.sub("", markdown)
    text = "\n".join(line.rstrip() for line in text.splitlines())
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


def save_page(url: str, markdown: str, out_dir: Path) -> Path:
    """Write one page with a 'Source: <url>' first line, then a blank line, then the page."""
    path = local_path(url, out_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"Source: {url}\n\n{clean(markdown)}", encoding="utf-8")
    return path


def print_stats(pages: dict[str, list[int]]) -> None:
    """Page and word counts per section and in total."""
    print(f"\n{'section':<22}{'pages':>7}{'words':>10}")
    for section in sorted(pages):
        counts = pages[section]
        print(f"{section:<22}{len(counts):>7}{sum(counts):>10}")
    total_pages = sum(len(c) for c in pages.values())
    total_words = sum(sum(c) for c in pages.values())
    print(f"{'TOTAL':<22}{total_pages:>7}{total_words:>10}")


def main(argv: list[str] | None = None) -> int:
    """Fetch every kept page, skipping (and logging) any that fail."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument("--dry-run", action="store_true", help="list chosen pages, fetch nothing")
    args = parser.parse_args(argv)

    chosen = [u for u in page_index() if keep(u)]
    print(f"{len(chosen)} pages chosen from the index")
    if args.dry_run:
        print("\n".join(chosen))
        return 0

    words: dict[str, list[int]] = defaultdict(list)
    failures = []
    for i, url in enumerate(chosen, 1):
        time.sleep(DELAY_SECONDS)
        try:
            markdown = fetch_text(markdown_url(url))
        except (urllib.error.URLError, ValueError, TimeoutError, UnicodeDecodeError) as err:
            print(f"[{i}/{len(chosen)}] SKIP {url}: {err}", file=sys.stderr)
            failures.append(url)
            continue
        save_page(url, markdown, args.out)
        words[section_of(url)].append(len(clean(markdown).split()))
        print(f"[{i}/{len(chosen)}] ok {url}")

    print_stats(words)
    if failures:
        print(f"\n{len(failures)} pages failed and were skipped:", *failures, sep="\n  ")
    return 0


if __name__ == "__main__":
    sys.exit(main())
