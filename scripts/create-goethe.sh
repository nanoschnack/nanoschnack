#!/usr/bin/env bash
set -euo pipefail

out_path="${1:-data/goethe.txt}"

python - <<PY
from pathlib import Path
from html import unescape
from html.parser import HTMLParser
import re
import time
from urllib.parse import urljoin, urlparse
import requests

AUTHOR_PAGE = "https://www.projekt-gutenberg.org/autoren/namen/goethe.html"
HEADERS = {"User-Agent": "nanoschnack-goethe/1.0 (+https://nanoschnack.de)"}
REQUEST_DELAY_SECS = 0.3


def _fetch_html(url, retries=4, backoff=1.5):
    session = requests.Session()
    for attempt in range(retries):
        resp = session.get(url, headers=HEADERS, timeout=30)
        if resp.status_code != 200:
            time.sleep(backoff * (attempt + 1))
            continue
        content = resp.content
        meta = re.search(
            r"<meta[^>]+charset=['\"]?([a-zA-Z0-9_-]+)",
            content.decode("latin-1", errors="ignore"),
            re.IGNORECASE,
        )
        encoding = meta.group(1) if meta else resp.encoding
        if not encoding:
            encoding = resp.apparent_encoding or "utf-8"
        return content.decode(encoding, errors="replace"), encoding
    raise RuntimeError(f"Failed to fetch HTML: {url}")


class _LinkParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag != "a":
            return
        href = dict(attrs).get("href")
        if href:
            self.links.append(href)


class _ParagraphParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.paragraphs = []
        self._capture = False
        self._buffer = []
        self._class_stack = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "div":
            self._class_stack.append(attrs.get("class", ""))
        elif tag == "p":
            if not self._is_excluded():
                self._capture = True
                self._buffer = []

    def handle_endtag(self, tag):
        if tag == "div" and self._class_stack:
            self._class_stack.pop()
        elif tag == "p" and self._capture:
            text = "".join(self._buffer).strip()
            if text:
                self.paragraphs.append(text)
            self._capture = False
            self._buffer = []

    def handle_data(self, data):
        if self._capture:
            self._buffer.append(data)

    def _is_excluded(self):
        excluded = {"navi-gb", "bottomnavi-gb", "dropdown", "anzeige-chap"}
        return any(cls in excluded for cls in self._class_stack)



def _work_links_from_author_page(html):
    parser = _LinkParser()
    parser.feed(html)
    links = []
    for href in parser.links:
        if "/goethe/" not in href:
            continue
        if not href.endswith(".html"):
            continue
        links.append(urljoin(AUTHOR_PAGE, href))
    return sorted(set(links))



def _work_pages_from_seed(seed_url):
    seed_parsed = urlparse(seed_url)
    base_dir = seed_parsed.path.rsplit("/", 1)[0] + "/"
    base_prefix = f"{seed_parsed.scheme}://{seed_parsed.netloc}{base_dir}"

    to_visit = [seed_url]
    seen = set()
    pages = []
    while to_visit:
        url = to_visit.pop()
        if url in seen:
            continue
        seen.add(url)
        html, encoding = _fetch_html(url)
        pages.append((url, html, encoding))
        parser = _LinkParser()
        parser.feed(html)
        for href in parser.links:
            resolved = urljoin(url, href)
            if not resolved.startswith(base_prefix):
                continue
            if not resolved.endswith(".html"):
                continue
            if resolved not in seen:
                to_visit.append(resolved)
        time.sleep(REQUEST_DELAY_SECS)
    return pages



def _extract_paragraphs(html):
    parser = _ParagraphParser()
    parser.feed(html)
    paragraphs = []
    for paragraph in parser.paragraphs:
        paragraph = unescape(" ".join(paragraph.split()))
        if paragraph.startswith("Anzeige"):
            continue
        if len(paragraph) < 40:
            continue
        paragraphs.append(paragraph)
    return paragraphs



def main():
    author_html, author_encoding = _fetch_html(AUTHOR_PAGE)
    seeds = _work_links_from_author_page(author_html)
    if not seeds:
        raise RuntimeError("No work links found on the author page.")
    print(f"Found {len(seeds)} work entry points. (charset: {author_encoding})", flush=True)

    out_file = Path("${out_path}")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    total_paragraphs = 0
    with out_file.open("w", encoding="utf-8") as handle:
        for idx, seed in enumerate(seeds, start=1):
            pages = _work_pages_from_seed(seed)
            print(f"[{idx}/{len(seeds)}] {seed} (charset: {pages[0][2]})", flush=True)
            for page_url, html, encoding in pages:
                paragraphs = _extract_paragraphs(html)
                for paragraph in paragraphs:
                    handle.write(paragraph + "\n")
                total_paragraphs += len(paragraphs)
    if total_paragraphs == 0:
        raise RuntimeError("No paragraphs extracted; the output file is empty.")
    print(f"Done. Wrote {total_paragraphs} paragraphs to ${out_path}.")


if __name__ == "__main__":
    main()
PY
