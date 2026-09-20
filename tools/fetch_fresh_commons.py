"""A clean holdout of REAL photos: Wikimedia Commons files taken and uploaded after the evaluated models were released, so
they cannot be in any of their training sets. Own-work uploads under CC0 / CC BY / CC BY-SA / public domain only.

Images go to gold/photos/ (gitignored; nothing is redistributed). gold/commons_manifest.jsonl keeps title, page URL,
author, license and dates for attribution and re-fetching. Search terms are spread so that the five labeling questions
(tools/label_gold.py) get balanced answers. Search relevance is loose, so NO labels are proposed from the terms: a
person labels every image from scratch.

uv run python tools/fetch_fresh_commons.py --taken-after 2026-08-20 --per-term 6
"""
import argparse
import json
import pathlib
import re
import time
import urllib.parse
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
API = "https://commons.wikimedia.org/w/api.php"
UA = "glance-research/0.2 (local model evaluation; fetches a few dozen freely licensed thumbnails once)"
OK_LICENSES = ("cc0", "cc by 4.0", "cc by-sa 4.0", "cc by 3.0", "cc by-sa 3.0", "public domain", "cc by 2.0", "cc by-sa 2.0")
# search term -> proposed labels (only what the term itself implies; everything is confirmed by a person)
TERMS = {
    "dog": {"subject": "animal"}, "cat": {"subject": "animal"}, "bird": {"subject": "animal"}, "insect macro": {"subject": "animal"},
    "dish food": {"subject": "food"}, "street food": {"subject": "food"}, "fruit market": {"subject": "food"},
    "bicycle": {"subject": "object"}, "tool": {"subject": "object"}, "teapot": {"subject": "object"}, "chair": {"subject": "object"},
    "street festival": {}, "market people": {}, "hikers": {}, "cyclist race": {},
    "church interior": {"indoors": True}, "museum interior": {"indoors": True}, "kitchen": {"indoors": True},
    "night street": {"light": "low_light"}, "blue hour city": {"light": "low_light"},
    "information board": {"text": True}, "street sign": {"text": True}, "menu board": {"text": True}, "book page": {"text": True, "subject": "document_or_screen"},
    "landscape mountain": {"subject": "scene", "indoors": False}, "beach": {"subject": "scene", "indoors": False},
}


def get(url, tries=6):
    """One polite request: on HTTP 429 / 5xx wait (Retry-After if given, else 60 s, doubling) and try again."""
    import urllib.error

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


def api(params):
    return json.loads(get(API + "?" + urllib.parse.urlencode({**params, "format": "json"})))


def plain(html):
    return re.sub(r"<[^>]+>", "", str(html or "")).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--taken-after", default="2026-08-20", help="keep photos whose capture date (EXIF) is on or after this day")
    parser.add_argument("--per-term", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true", help="list candidates, download nothing")
    args = parser.parse_args()
    photos = ROOT / "gold" / "photos"
    photos.mkdir(parents=True, exist_ok=True)
    manifest_path = ROOT / "gold" / "commons_manifest.jsonl"
    seen = {json.loads(l)["title"] for l in manifest_path.read_text().splitlines()} if manifest_path.exists() else set()
    count = len(seen)
    for term, proposal in TERMS.items():
        data = api({"action": "query", "generator": "search", "gsrsearch": f"{term} filetype:bitmap filemime:image/jpeg", "gsrnamespace": 6,
                    "gsrsort": "create_timestamp_desc", "gsrlimit": 50, "prop": "imageinfo", "iiprop": "url|timestamp|extmetadata|size|mime", "iiurlwidth": 1280})
        kept = 0
        for page in sorted((data.get("query") or {}).get("pages", {}).values(), key=lambda p: p.get("index", 0)):
            info = (page.get("imageinfo") or [{}])[0]
            meta = {k: plain(v.get("value", "")) for k, v in (info.get("extmetadata") or {}).items()}
            taken = (meta.get("DateTimeOriginal") or "")[:10]
            license_name = (meta.get("LicenseShortName") or "").lower()
            if page["title"] in seen or info.get("width", 0) < 1000 or not re.match(r"\d{4}-\d{2}-\d{2}", taken) or taken < args.taken_after:
                continue
            if not any(license_name.startswith(ok) for ok in OK_LICENSES) or "own work" not in (meta.get("Credit") or "").lower():
                continue
            count += 1
            name = f"commons_{count:04d}.jpg"
            row = {"image": f"gold/photos/{name}", "title": page["title"], "page": info.get("descriptionurl"), "author": meta.get("Artist"),
                   "license": meta.get("LicenseShortName"), "taken": taken, "uploaded": info.get("timestamp"), "search_term": term}
            print(("would fetch " if args.dry_run else "fetch ") + f"{name}  taken {taken}  {row['license']:<14} [{term}] {page['title'][:60]}")
            if not args.dry_run:
                (photos / name).write_bytes(get(info["thumburl"]))
                with open(manifest_path, "a") as f:
                    f.write(json.dumps(row) + "\n")
                time.sleep(3.0)
            seen.add(page["title"])
            kept += 1
            if kept >= args.per_term:
                break
        time.sleep(4.0)
    print(f"{count} images in total" + (" (dry run: nothing downloaded)" if args.dry_run else f" in {photos.relative_to(ROOT)}/"))


if __name__ == "__main__":
    main()
