"""The PAPER version of the project page, as HTML and PDF, for review and for arXiv preparation.

It is built FROM the generated page (`site/index.html`, written by `tools/make_site.py`), so every number still comes from a result
file; nothing is retyped. Differences from the website: the website-only "Use it" panel and the navigation are dropped, the paper's
descriptive title is the heading, the three-regimes summary follows the abstract, a disclosure paragraph precedes the references, and
a print stylesheet sets A4 pages with page numbers and keeps figures and tables from breaking across pages.

uv run python tools/make_site.py && uv run python tools/make_paper.py        # needs a local Chrome for the PDF
"""
import datetime
import pathlib
import re
import subprocess
import tempfile
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
OUT = ROOT / "paper"
src = (ROOT / "site/index.html").read_text()
version = next(line.split('"')[1] for line in (ROOT / "pyproject.toml").read_text().splitlines() if line.startswith("version"))
snapshot = re.search(r"snapshot ([0-9a-f]{7})", src).group(1)
today = datetime.date.today()
date = f"{today.day} {today.strftime('%B %Y')}"
TITLE = re.search(r'<p class="papertitle">(.*?)</p>', src, re.S).group(1).strip()

style = re.search(r"<style>.*?</style>", src, re.S).group(0)
fonts = re.search(r'<link rel="stylesheet"[^>]*fonts.googleapis[^>]*>', src).group(0)
body = src[src.index("<main>") + len("<main>"):src.index("</main>")]


def cut(pattern, text, flags=re.S):
    found = re.search(pattern, text, flags)
    assert found, pattern[:60]
    return found.group(0), text[:found.start()] + text[found.end():]


_, body = cut(r'<aside class="useit site-only".*?</aside>\s*', body)  # the website-only panel
regimes, body = cut(r'<section aria-labelledby="regimes".*?</section>\s*', body)
_, body = cut(r"<header>.*?</header>\s*", body)
header = (f'<header class="paper">\n  <h1>{TITLE}</h1>\n  <p class="byline">Yohei Nakajima <span class="aff">· independent</span></p>\n'
          f'  <p class="running">Working paper · {date} · glance-vlm v{version}, snapshot {snapshot} · code, data and the registration notebook: github.com/yoheinakajima/glance · '
          "current version: glance.yohei.me</p>\n</header>\n")
regimes = regimes.replace("The result in three regimes", "Summary: the result in three regimes")
abstract_end = body.index("</section>", body.index('aria-labelledby="abstract"')) + len("</section>")
body = body[:abstract_end] + "\n\n" + regimes + body[abstract_end:]
disclosure = ('<section aria-labelledby="disclosure"><h2 id="disclosure" class="plain">Disclosure</h2>\n'
              "  <p>This work was carried out with AI assistance throughout: an AI assistant wrote most of the code, ran the experiments and drafted the text under the author’s direction, "
              "and the project’s notebook records each registration, result and correction with its time. The author reviewed the claims and is responsible for them. "
              "Hosted models’ answers were used for evaluation only and are not stored; only whether each answer was right is kept.</p>\n</section>\n\n")
body = body.replace('<section aria-labelledby="refs">', disclosure + '<section aria-labelledby="refs">', 1)

PRINT = """
<style>
@page{size:A4;margin:19mm 19mm 21mm;@bottom-center{content:counter(page);font:8.5pt "IBM Plex Sans",Arial,sans-serif;color:#56616b}}
:root{--paper:#fff;--ink:#14171a;--muted:#56616b;--rule:#d5dbe1;--link:#14171a;--miss:#8f2d2d;--own:#14171a;--tint:#f3f5f7}
html{font-size:14.4px}body{padding:0;font-size:10.3pt;line-height:1.5}main{max-width:none;gap:1.5rem}
header.paper h1{font-size:19pt;line-height:1.22;margin:0 0 .55rem;max-width:none}header.paper .byline{margin:.2rem 0}header.paper .running{margin:.5rem 0 0;border:0;padding:0}
.abstract{font-size:10.3pt;line-height:1.5}h2{break-after:avoid;font-size:12.5pt}h3{break-after:avoid}p{orphans:3;widows:3}
figure,.table-scroll,table,pre,.protocol,.regimes tr,ul.misses li,.refs li{break-inside:avoid}
.table-scroll{overflow:visible}td,thead th{white-space:normal}td.n{white-space:nowrap}table{font-size:8.5pt}td{padding:.3rem .7rem .3rem 0}td:first-child{min-width:7.5rem}
.regimes td{font-size:8.9pt}.ci{font-size:7.6pt}.caption,figcaption{font-size:8.6pt}pre{font-size:7.6pt;white-space:pre-wrap;word-break:break-word}
a{color:inherit;text-decoration:none}.refs{font-size:8.8pt}.refs a{word-break:break-all;color:var(--muted)}
.only-narrow{display:none!important}.only-wide{display:block!important}footer{font-size:8.6pt}
</style>
"""
html_out = ('<!doctype html>\n<html lang="en" data-theme="light">\n<head>\n<meta charset="utf-8">\n<meta name="viewport" content="width=device-width, initial-scale=1">\n'
            f"<title>{TITLE}</title>\n<meta name=\"author\" content=\"Yohei Nakajima\">\n{fonts}\n{style}\n{PRINT}</head>\n<body>\n<main>\n{header}{body}</main>\n</body>\n</html>\n")
OUT.mkdir(exist_ok=True)
page = OUT / "glance-vlm-paper.html"
page.write_text(html_out)
pdf = OUT / f"glance-vlm-paper-v{version}.pdf"
pdf.unlink(missing_ok=True)
with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as profile:
    proc = subprocess.Popen([CHROME, "--headless=new", "--disable-gpu", "--no-first-run", f"--user-data-dir={profile}", "--no-pdf-header-footer", "--virtual-time-budget=6000",
                             f"--print-to-pdf={pdf}", page.as_uri()], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(60):
        if pdf.exists() and pdf.stat().st_size > 0:
            break
        time.sleep(1)
    time.sleep(2)
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
print("wrote", page.relative_to(ROOT), "and", pdf.relative_to(ROOT), pdf.stat().st_size // 1024, "KB")
