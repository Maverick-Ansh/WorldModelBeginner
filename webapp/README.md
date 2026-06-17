# arXiv Research Feed 📚

A free, auto-updating website that lists recent research papers from arXiv in:

| Category | Topic |
|----------|-------|
| `cs.CV`  | Computer Vision & Pattern Recognition |
| `cs.AI`  | Artificial Intelligence |
| `cs.LG`  | Machine Learning |
| `cs.CL`  | Computation & Language (NLP) |
| `cs.NE`  | Neural & Evolutionary Computing |

**Live site:** https://maverick-ansh.github.io/WorldModelBeginner/

## How it works

There is no server and no cost. Everything runs on free GitHub infrastructure:

1. **`fetch_papers.py`** queries the public [arXiv API](https://info.arxiv.org/help/api/index.html)
   for the newest papers in each category and writes `site/papers.json`.
   It uses only the Python standard library (no dependencies).
2. **`site/index.html`** is a single self-contained page (vanilla HTML/CSS/JS,
   no frameworks, no CDNs) that loads `papers.json` and renders a searchable,
   filterable list of papers.
3. **`.github/workflows/arxiv-pages.yml`** runs the fetcher and publishes the
   `site/` folder to **GitHub Pages** on every push and on a daily schedule.

## Updating manually

Go to the repo's **Actions** tab → **Build & Deploy arXiv Feed** → **Run workflow**.

## Daily auto-updates

GitHub only runs *scheduled* workflows from the repository's **default branch**.
To enable the daily refresh, merge this branch into the default branch.
Until then, the site still updates on every push and whenever you run the
workflow manually.

## Run locally

```bash
python webapp/fetch_papers.py --out webapp/site/papers.json
cd webapp/site && python -m http.server 8000
# open http://localhost:8000
```

---

Thank you to arXiv for use of its open access interoperability. This project
links to papers hosted on arXiv.org and is not affiliated with or endorsed by arXiv.
