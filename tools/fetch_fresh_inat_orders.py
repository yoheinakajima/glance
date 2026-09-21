"""A HARDER fresh photo test than fetch_fresh_inat.py (lab/NOTES.md entry 47): same iNaturalist research-grade
rules, but classes are INSECT ORDERS instead of iconic taxa, so the classes look alike (a beetle vs a true bug vs
a fly). Observations are selected by the `taxon_id` filter (the order's id, matching the observation's taxon or
any of its ancestors) instead of `iconic_taxa`. Order taxon ids are resolved live from the API at start, never
hard-coded. Zero labeling by us; the label is the community identification's order.

Images: .cache/datasets/fresh_inat_orders/images/ (not committed, not redistributed). Metadata, labels,
attributions and URLs: glance/evals/manifests/fresh_inat_orders_source.jsonl (committed, so anyone can re-fetch
the same files).

uv run python tools/fetch_fresh_inat_orders.py --observed-after 2026-08-15 --per-class 30
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
import urllib.error
import urllib.parse

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from fetch_fresh_inat import MAX_PAGES, first_ok_photo, get, square_to_medium, _verify_image  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
OBSERVATIONS_API = "https://api.inaturalist.org/v1/observations"
TAXA_API = "https://api.inaturalist.org/v1/taxa"
CLASSES = {  # our class key -> scientific order name
    "beetle": "Coleoptera",
    "butterfly_or_moth": "Lepidoptera",
    "bee_wasp_or_ant": "Hymenoptera",
    "fly": "Diptera",
    "dragonfly_or_damselfly": "Odonata",
    "true_bug": "Hemiptera",
    "grasshopper_or_cricket": "Orthoptera",
}


def resolve_order_taxon_id(payload: dict, order_name: str) -> int:
    """Pure: pick the taxon id from a parsed `/v1/taxa` response whose `name` == order_name exactly and whose
    `rank` == "order". Raises ValueError if there is no such exact match."""
    for result in payload.get("results") or []:
        if result.get("name") == order_name and result.get("rank") == "order":
            return result["id"]
    raise ValueError(f"no exact order-rank taxon named {order_name!r} in taxa search results")


def lookup_order_taxon_id(order_name: str) -> int:
    params = {"q": order_name, "rank": "order"}
    payload = json.loads(get(TAXA_API + "?" + urllib.parse.urlencode(params)))
    return resolve_order_taxon_id(payload, order_name)


def keep_observation(obs: dict, order_id: int, cutoff: str, seen_observers: set[str]) -> dict | None:
    """Pure filter: the photo to keep for `obs` under order `order_id`, or None if it should be skipped.

    Mirrors fetch_fresh_inat.keep_observation but matches by ORDER (the observation's taxon id or any of its
    `ancestor_ids`) instead of iconic taxon name equality. Does not mutate `seen_observers` -- the caller adds the
    observer only once a photo is actually kept, so this stays a pure function of its inputs and is safe to
    unit-test directly.
    """
    if obs.get("quality_grade") != "research":
        return None
    taxon = obs.get("taxon") or {}
    ancestor_ids = taxon.get("ancestor_ids") or []
    if taxon.get("id") != order_id and order_id not in ancestor_ids:
        return None
    observed_on = obs.get("observed_on") or ""
    if not observed_on or observed_on < cutoff:
        return None
    login = (obs.get("user") or {}).get("login")
    if not login or login in seen_observers:
        return None
    return first_ok_photo(obs.get("photos") or [])


def fetch_class(key: str, order_name: str, order_id: int, cutoff: str, per_class: int, images_dir: pathlib.Path) -> tuple[int, list[dict]]:
    seen_observers: set[str] = set()
    rows: list[dict] = []
    candidates = 0
    for page in range(1, MAX_PAGES + 1):
        if len(rows) >= per_class:
            break
        params = {
            "quality_grade": "research", "photos": "true", "d1": cutoff, "photo_license": "cc0,cc-by,cc-by-sa",
            "taxon_id": order_id, "order_by": "created_at", "order": "desc", "per_page": 100, "page": page,
        }
        data = json.loads(get(OBSERVATIONS_API + "?" + urllib.parse.urlencode(params)))
        time.sleep(1.1)  # at most 1 request/second to api.inaturalist.org
        results = data.get("results") or []
        candidates += len(results)
        for obs in results:
            if len(rows) >= per_class:
                break
            photo = keep_observation(obs, order_id, cutoff, seen_observers)
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
                "order_name": order_name,
            })
        if len(results) < 100:
            break  # fewer than a full page: no more results to page through
    return candidates, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--observed-after", default="2026-08-15")
    parser.add_argument("--per-class", type=int, default=30)
    args = parser.parse_args()

    images_dir = ROOT / ".cache" / "datasets" / "fresh_inat_orders" / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    manifest = ROOT / "glance" / "evals" / "manifests" / "fresh_inat_orders_source.jsonl"

    order_ids: dict[str, int] = {}
    for key, order_name in CLASSES.items():
        order_ids[key] = lookup_order_taxon_id(order_name)
        time.sleep(1.1)  # at most 1 request/second to api.inaturalist.org
    print("resolved order taxon ids (live from the API, not hard-coded):")
    for key, order_name in CLASSES.items():
        print(f"  {key} ({order_name}): {order_ids[key]}")

    all_rows: list[dict] = []
    for key, order_name in CLASSES.items():
        candidates, rows = fetch_class(key, order_name, order_ids[key], args.observed_after, args.per_class, images_dir)
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
