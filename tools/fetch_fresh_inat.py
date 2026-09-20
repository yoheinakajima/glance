"""A fresh, real-photo benchmark with CLEANER labels than fetch_fresh_depicts.py: iNaturalist research-grade
observations, where at least two people agreed on the identification and the photo IS of that organism (not just
"depicts", which can mean "appears somewhere in the frame"). Taken AFTER the evaluated models were released, CC0 /
CC BY / CC BY-SA. Zero labeling by us; the label is the community identification's iconic taxon.

Images: .cache/datasets/fresh_inat/images/ (not committed, not redistributed). Metadata, labels, attributions and
URLs: glance/evals/manifests/fresh_inat_source.jsonl (committed, so anyone can re-fetch the same files).

uv run python tools/fetch_fresh_inat.py --observed-after 2026-08-15 --per-class 20
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import time
import urllib.error
import urllib.parse
import urllib.request

from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parent.parent
API = "https://api.inaturalist.org/v1/observations"
UA = "glance-research/0.2 (local model evaluation; about 200 freely licensed photos, fetched once)"
OK_LICENSES = ("cc0", "cc-by", "cc-by-sa")
MAX_PAGES = 4  # per class, per the registered design (lab/NOTES.md entry 35)
CLASSES = {  # our class key -> iNaturalist iconic taxon name
    "bird": "Aves", "insect": "Insecta", "plant": "Plantae", "mammal": "Mammalia", "reptile": "Reptilia",
    "amphibian": "Amphibia", "fish": "Actinopterygii", "fungus": "Fungi", "arachnid": "Arachnida", "mollusc": "Mollusca",
}


def get(url, tries=6):
    """GET raw bytes, retrying HTTP 429/5xx with exponential backoff honoring Retry-After."""
    wait = 2
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as exc:
            if exc.code not in (429, 500, 502, 503, 504) or attempt == tries - 1:
                raise
            pause = int(exc.headers.get("Retry-After") or wait)
            print(f"  HTTP {exc.code}: waiting {pause} s (attempt {attempt + 1}/{tries})", flush=True)
            time.sleep(pause)
            wait *= 2


def square_to_medium(url: str) -> str | None:
    """Rewrite an iNaturalist photo URL's `square.<ext>` suffix to `medium.<ext>` (~500 px); None if it doesn't match."""
    m = re.search(r"square\.(jpg|jpeg|png)$", url, re.IGNORECASE)
    if not m:
        return None
    return url[: m.start()] + "medium." + m.group(1)


def first_ok_photo(photos: list[dict]) -> dict | None:
    """The first photo (in order) whose license_code is cc0/cc-by/cc-by-sa, or None."""
    for photo in photos:
        if (photo.get("license_code") or "").lower() in OK_LICENSES:
            return photo
    return None


def keep_observation(obs: dict, iconic_name: str, cutoff: str, seen_observers: set[str]) -> dict | None:
    """Pure filter: the photo to keep for `obs` under class `iconic_name`, or None if it should be skipped.

    Does not mutate `seen_observers` -- the caller adds the observer only once a photo is actually kept, so this
    stays a pure function of its inputs and is safe to unit-test directly.
    """
    if obs.get("quality_grade") != "research":
        return None
    taxon = obs.get("taxon") or {}
    if not taxon.get("rank"):
        return None
    if taxon.get("iconic_taxon_name") != iconic_name:
        return None
    observed_on = obs.get("observed_on") or ""
    if not observed_on or observed_on < cutoff:
        return None
    login = (obs.get("user") or {}).get("login")
    if not login or login in seen_observers:
        return None
    return first_ok_photo(obs.get("photos") or [])


def fetch_class(key: str, iconic_name: str, cutoff: str, per_class: int, images_dir: pathlib.Path) -> tuple[int, list[dict]]:
    seen_observers: set[str] = set()
    rows: list[dict] = []
    candidates = 0
    for page in range(1, MAX_PAGES + 1):
        if len(rows) >= per_class:
            break
        params = {
            "quality_grade": "research", "photos": "true", "d1": cutoff, "photo_license": "cc0,cc-by,cc-by-sa",
            "iconic_taxa": iconic_name, "order_by": "created_at", "order": "desc", "per_page": 100, "page": page,
        }
        data = json.loads(get(API + "?" + urllib.parse.urlencode(params)))
        time.sleep(1.1)  # at most 1 request/second to api.inaturalist.org
        results = data.get("results") or []
        candidates += len(results)
        for obs in results:
            if len(rows) >= per_class:
                break
            photo = keep_observation(obs, iconic_name, cutoff, seen_observers)
            if photo is None:
                continue
            img_url = square_to_medium(photo.get("url") or "")
            if img_url is None or not img_url.lower().endswith((".jpg", ".jpeg")):
                continue  # not a jpeg (e.g. .png); skip per the registered design
            obs_id = obs["id"]
            name = f"{key}_{obs_id}.jpg"
            path = images_dir / name
            if not path.exists():
                try:
                    raw = get(img_url)
                except urllib.error.HTTPError as exc:
                    print(f"  {key}/{obs_id}: download failed ({exc}), skipping", flush=True)
                    continue
                time.sleep(0.5)  # be polite between image downloads
                path.write_bytes(raw)
                if not _verify_image(path):
                    path.unlink(missing_ok=True)
                    continue
            taxon = obs["taxon"]
            seen_observers.add(obs["user"]["login"])
            rows.append({
                "item_id": name[:-4], "label": key, "file": name, "observation_id": obs_id,
                "page": f"https://www.inaturalist.org/observations/{obs_id}",
                "photo_id": photo.get("id"), "photo_license": (photo.get("license_code") or "").lower(),
                "attribution": photo.get("attribution"), "observer": obs["user"]["login"],
                "observed_on": obs.get("observed_on"), "created_at": obs.get("created_at"),
                "taxon_id": taxon.get("id"), "taxon_name": taxon.get("name"),
                "taxon_common_name": taxon.get("preferred_common_name"), "taxon_rank": taxon.get("rank"),
                "iconic_taxon_name": taxon.get("iconic_taxon_name"), "captive": obs.get("captive"),
                "num_identification_agreements": obs.get("num_identification_agreements"),
            })
        if len(results) < 100:
            break  # fewer than a full page: no more results to page through
    return candidates, rows


def _verify_image(path: pathlib.Path) -> bool:
    """PIL-verify the download, and require min(width, height) >= 200; caller deletes the file on failure."""
    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            width, height = img.size
        return min(width, height) >= 200
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--observed-after", default="2026-08-15")
    parser.add_argument("--per-class", type=int, default=20)
    args = parser.parse_args()

    images_dir = ROOT / ".cache" / "datasets" / "fresh_inat" / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    manifest = ROOT / "glance" / "evals" / "manifests" / "fresh_inat_source.jsonl"

    all_rows: list[dict] = []
    for key, iconic_name in CLASSES.items():
        candidates, rows = fetch_class(key, iconic_name, args.observed_after, args.per_class, images_dir)
        print(f"{key}: {candidates} candidates seen, {len(rows)} kept", flush=True)
        if len(rows) < args.per_class:
            print(f"  {key}: only reached {len(rows)}/{args.per_class} within {MAX_PAGES} pages", flush=True)
        all_rows += rows

    manifest.write_text("".join(json.dumps(r) + "\n" for r in all_rows))
    licenses: dict[str, int] = {}
    for r in all_rows:
        licenses[r["photo_license"]] = licenses.get(r["photo_license"], 0) + 1
    print(f"\n{len(all_rows)} images total -> {images_dir.relative_to(ROOT)}/ ; manifest -> {manifest.relative_to(ROOT)}")
    print(f"license breakdown: {licenses}")


if __name__ == "__main__":
    main()
