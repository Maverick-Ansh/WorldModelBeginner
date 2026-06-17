#!/usr/bin/env python3
"""Fetch recent arXiv papers for a set of categories and write a JSON feed.

This runs inside GitHub Actions (where outbound internet is available) and uses
only the Python standard library, so nothing needs to be pip-installed.

Output: a single JSON file consumed by the static site in ``webapp/site``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

ARXIV_API = "https://export.arxiv.org/api/query"

# The categories requested, mapped to their human-readable names.
CATEGORIES = {
    "cs.CV": "Computer Vision & Pattern Recognition",
    "cs.AI": "Artificial Intelligence",
    "cs.LG": "Machine Learning",
    "cs.CL": "Computation & Language (NLP)",
    "cs.NE": "Neural & Evolutionary Computing",
}

NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}

USER_AGENT = (
    "arxiv-research-feed/1.0 "
    "(+https://github.com/Maverick-Ansh/WorldModelBeginner static site)"
)

VERSION_RE = re.compile(r"v\d+$")


def _text(node, default=""):
    """Return collapsed text of an XML node, or a default if missing."""
    if node is None or node.text is None:
        return default
    return " ".join(node.text.split())


def fetch_category(category, max_results, retries=4):
    """Return raw Atom XML for one category, retrying on transient errors."""
    params = urllib.parse.urlencode(
        {
            "search_query": f"cat:{category}",
            "start": 0,
            "max_results": max_results,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
    )
    url = f"{ARXIV_API}?{params}"
    delay = 3
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=60) as resp:
                return resp.read()
        except (urllib.error.URLError, TimeoutError) as err:
            last_err = err
            print(f"  attempt {attempt} for {category} failed: {err}", file=sys.stderr)
            time.sleep(delay)
            delay *= 2
    raise RuntimeError(f"could not fetch {category}: {last_err}")


def parse_entries(xml_bytes):
    """Parse Atom XML bytes into a list of paper dicts."""
    root = ET.fromstring(xml_bytes)
    papers = []
    for entry in root.findall("atom:entry", NS):
        raw_id = _text(entry.find("atom:id", NS))
        if "/abs/" in raw_id:
            arxiv_id = raw_id.rsplit("/abs/", 1)[-1]
        else:
            arxiv_id = raw_id

        authors = [_text(a.find("atom:name", NS)) for a in entry.findall("atom:author", NS)]
        authors = [a for a in authors if a]

        categories = [c.get("term") for c in entry.findall("atom:category", NS) if c.get("term")]

        primary = entry.find("arxiv:primary_category", NS)
        if primary is not None and primary.get("term"):
            primary_cat = primary.get("term")
        elif categories:
            primary_cat = categories[0]
        else:
            primary_cat = ""

        abs_url = raw_id
        pdf_url = ""
        for link in entry.findall("atom:link", NS):
            if link.get("type") == "application/pdf":
                pdf_url = link.get("href", "")
            elif link.get("rel") == "alternate":
                abs_url = link.get("href", abs_url)
        if not pdf_url and arxiv_id:
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"

        papers.append(
            {
                "id": arxiv_id,
                "title": _text(entry.find("atom:title", NS)),
                "authors": authors,
                "abstract": _text(entry.find("atom:summary", NS)),
                "published": _text(entry.find("atom:published", NS)),
                "updated": _text(entry.find("atom:updated", NS)),
                "abs_url": abs_url,
                "pdf_url": pdf_url,
                "primary_category": primary_cat,
                "categories": categories,
            }
        )
    return papers


def main():
    parser = argparse.ArgumentParser(description="Build the arXiv paper feed JSON.")
    parser.add_argument("--out", default="webapp/site/papers.json")
    parser.add_argument("--max-results", type=int, default=60)
    args = parser.parse_args()

    by_id = {}
    for category in CATEGORIES:
        print(f"Fetching {category} ...", file=sys.stderr)
        try:
            xml_bytes = fetch_category(category, args.max_results)
            entries = parse_entries(xml_bytes)
        except (RuntimeError, ET.ParseError) as err:
            print(f"  skipping {category}: {err}", file=sys.stderr)
            entries = []
        print(f"  got {len(entries)} entries", file=sys.stderr)
        for paper in entries:
            key = VERSION_RE.sub("", paper["id"])
            if key not in by_id:
                by_id[key] = paper
        # arXiv asks callers to wait a few seconds between requests.
        time.sleep(3)

    papers = sorted(by_id.values(), key=lambda p: p.get("published", ""), reverse=True)

    if not papers:
        print("ERROR: no papers fetched from any category", file=sys.stderr)
        return 1

    payload = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "categories": list(CATEGORIES.keys()),
        "category_names": CATEGORIES,
        "count": len(papers),
        "papers": papers,
    }

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, separators=(",", ":"))
    print(f"Wrote {len(papers)} papers to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
