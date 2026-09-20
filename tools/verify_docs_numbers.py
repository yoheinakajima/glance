"""Check that every 3-decimal number in the docs appears in a committed source (results/**/metrics.json, report.md,
summary.txt, the prefix-cache acceptance JSON, lab reports, STATUS.md). Prints the ones that do not.

uv run python tools/verify_docs_numbers.py
"""
import json, pathlib, re, sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
known: set[str] = set()


def collect(o):
    if isinstance(o, dict):
        for v in o.values(): collect(v)
    elif isinstance(o, list):
        for v in o: collect(v)
    elif isinstance(o, float):
        known.update({f"{o:.3f}", f"{o:.4f}"})


for path in list((ROOT / "results").rglob("*.json")) + list((ROOT / "lab").rglob("*.json")):
    try:
        collect(json.loads(path.read_text()))
    except json.JSONDecodeError:
        pass
text_sources = [ROOT / "STATUS.md", ROOT / "lab" / "NOTES.md"] + list((ROOT / "results").rglob("*.md")) + list((ROOT / "results").rglob("*.txt")) + list((ROOT / "lab").rglob("*.md"))
source_text = "\n".join(p.read_text() for p in text_sources if p.exists())

# Numbers quoted from third-party pages (not our measurements). They are cited with their source in the document and
# are excluded from the check on purpose; everything else must trace to a committed file.
EXTERNAL = {"docs/paper/RELATED_WORK.md": {"8.566", "0.114", "0.061", "0.025", "0.012", "0.004"}}

bad = 0
for doc in sorted((ROOT / "docs").rglob("*.md")):
    text = doc.read_text()
    nums = set(re.findall(r"(?<![\w.])\d\.\d{3}(?![\d])", text))
    external = EXTERNAL.get(str(doc.relative_to(ROOT)), set()) & nums
    missing = sorted(n for n in nums - external if n not in known and n not in source_text)
    print(f"{doc.relative_to(ROOT)}: {len(nums)} numbers, {len(missing)} unverified" + (f", {len(external)} third-party (not checked)" if external else ""))
    for n in missing:
        line = next(l for l in text.splitlines() if n in l)
        print(f"    {n} | {line.strip()[:160]}")
    bad += len(missing)
sys.exit(1 if bad else 0)
