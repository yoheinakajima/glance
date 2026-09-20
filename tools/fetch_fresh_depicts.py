"""A fresh, real-photo benchmark with labels nobody on this project made: Wikimedia Commons files whose structured data
says what they depict (property P180, added by uploaders and editors), taken AFTER the evaluated models were released,
own work, CC0 / CC BY / CC BY-SA. Zero labeling by us; the label noise that exists (a "dog" tag on a dog statue) is
the same for every system.

Images: .cache/datasets/fresh_commons/images/ (not committed, not redistributed). Metadata, labels, authors, licenses and
URLs: glance/evals/manifests/fresh_commons_source.jsonl (committed, so anyone can re-fetch the same files).

uv run python tools/fetch_fresh_depicts.py --taken-after 2026-08-15 --per-class 12
"""
import argparse
import json
import pathlib
import re
import time
import urllib.error
import urllib.parse
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
API = "https://commons.wikimedia.org/w/api.php"
UA = "glance-research/0.2 (local model evaluation; a few hundred freely licensed thumbnails, fetched once)"
OK_LICENSES = ("cc0", "cc by 4.0", "cc by-sa 4.0", "cc by 3.0", "cc by-sa 3.0", "public domain", "cc by 2.0", "cc by-sa 2.0")
CLASSES = {  # class key -> (Wikidata item the file must depict, description shown to the models)
    "dog": ("Q144", "A dog"), "cat": ("Q146", "A cat"), "bird": ("Q5113", "A bird"), "horse": ("Q726", "A horse"),
    "bicycle": ("Q11442", "A bicycle"), "car": ("Q1420", "A car"), "train": ("Q870", "A train"), "boat": ("Q35872", "A boat"),
    "bridge": ("Q12280", "A bridge"), "church": ("Q16970", "A church building"), "mountain": ("Q8502", "A mountain"),
    "beach": ("Q40080", "A beach"), "flower": ("Q506", "A flower"),
}


def get(url, tries=6):
    wait = 60
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as exc:
            if exc.code not in (429, 500, 502, 503) or attempt == tries - 1:
                raise
            pause = int(exc.headers.get("Retry-After") or wait)
            print(f"  HTTP {exc.code}: waiting {pause} s", flush=True)
            time.sleep(pause)
            wait *= 2


def candidates(qid, taken_after):
    data = json.loads(get(API + "?" + urllib.parse.urlencode({
        "action": "query", "format": "json", "generator": "search", "gsrnamespace": 6, "gsrlimit": 50, "gsrsort": "create_timestamp_desc",
        "gsrsearch": f"haswbstatement:P180={qid} filetype:bitmap filemime:image/jpeg",
        "prop": "imageinfo", "iiprop": "url|timestamp|extmetadata|size", "iiurlwidth": 1280})))
    out = []
    for page in sorted((data.get("query") or {}).get("pages", {}).values(), key=lambda p: p.get("index", 0)):
        info = (page.get("imageinfo") or [{}])[0]
        meta = {k: re.sub(r"<[^>]+>", "", str(v.get("value", ""))).strip() for k, v in (info.get("extmetadata") or {}).items()}
        taken, lic = (meta.get("DateTimeOriginal") or "")[:10], (meta.get("LicenseShortName") or "").lower()
        if (re.match(r"\d{4}-\d{2}-\d{2}", taken) and taken >= taken_after and info.get("width", 0) >= 1000
                and any(lic.startswith(ok) for ok in OK_LICENSES) and "own work" in (meta.get("Credit") or "").lower()):
            out.append({"title": page["title"], "page": info.get("descriptionurl"), "thumb": info.get("thumburl"), "author": meta.get("Artist"),
                        "license": meta.get("LicenseShortName"), "taken": taken, "uploaded": info.get("timestamp")})
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--taken-after", default="2026-08-15")
    parser.add_argument("--per-class", type=int, default=12)
    parser.add_argument("--per-author", type=int, default=2, help="at most this many images of one class from one author (near-duplicates)")
    args = parser.parse_args()
    images = ROOT / ".cache" / "datasets" / "fresh_commons" / "images"
    images.mkdir(parents=True, exist_ok=True)
    manifest = ROOT / "glance" / "evals" / "manifests" / "fresh_commons_source.jsonl"
    pools = {}
    for key, (qid, _) in CLASSES.items():
        pools[key] = candidates(qid, args.taken_after)
        print(f"{key}: {len(pools[key])} fresh own-work candidates", flush=True)
        time.sleep(4)
    seen_in = {}
    for key, pool in pools.items():
        for c in pool:
            seen_in.setdefault(c["title"], set()).add(key)
    rows = []
    for key, pool in pools.items():
        per_author, kept = {}, 0
        for c in pool:
            if len(seen_in[c["title"]]) > 1 or per_author.get(c["author"], 0) >= args.per_author:
                continue  # tagged with two of our classes, or a near-duplicate series
            name = f"{key}_{kept + 1:02d}.jpg"
            path = images / name
            if not path.exists():
                path.write_bytes(get(c["thumb"]))
                time.sleep(3)
            rows.append({"item_id": name[:-4], "label": key, "file": name, **{k: c[k] for k in ("title", "page", "author", "license", "taken", "uploaded")}})
            per_author[c["author"]] = per_author.get(c["author"], 0) + 1
            kept += 1
            if kept >= args.per_class:
                break
        print(f"{key}: kept {kept}", flush=True)
    manifest.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"{len(rows)} images -> {images.relative_to(ROOT)}/ ; labels and attributions -> {manifest.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
