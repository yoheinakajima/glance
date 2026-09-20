# Datasets

Every license below was read from the dataset's own card or record before anything was downloaded
(HANDOFF sections 8 and 11). Expectations in the hand-off were from memory; where the card says something
different, the card is what is recorded. Unclear means the suite is skipped. All use here is evaluation only:
nothing is trained, and no dataset content is redistributed by this repo (manifests hold item ids and hashes).

| Suite | Dataset | Where it comes from | License as verified | Checked | Pin | Used |
| --- | --- | --- | --- | --- | --- | --- |
| `pope` | POPE questions (random, popular, adversarial) | https://github.com/RUCAIBox/POPE, `output/coco/coco_pope_*.json` | MIT (GitHub repo license, SPDX `MIT`) | 2026-09-19 | commit `08d957b917e5a378a2f99d35b6293c536a66298b` | yes |
| `pope` | COCO val2014 images (only the ~500 the selected items need) | http://images.cocodataset.org/val2014/ | COCO Terms of Use (cocodataset.org/#termsofuse): annotations CC BY 4.0; the consortium does not own the images, and their use "must abide by the Flickr Terms of Use" | 2026-09-19 | file names from the POPE question files | yes, local evaluation only |
| `gqa_yesno` | GQA testdev-balanced, yes/no questions | https://huggingface.co/datasets/lmms-lab/GQA (`testdev_balanced_images`, `testdev_balanced_instructions`) | MIT on the Hub card; the upstream GQA download page carries a CC BY 4.0 badge | 2026-09-19 | revision `a6e72d6e1b912da88af8b2f9eba05d5ea8ec2dd8` | yes |
| `pets37` | Oxford-IIIT Pet, test split | https://huggingface.co/datasets/timm/oxford-iiit-pet | CC BY-SA 4.0 (Hub card) | 2026-09-19 | revision `089695c834a7deb60505b7cc506672db1c31a6aa` | yes |
| `caltech101` | Caltech-101 | https://data.caltech.edu/records/mzrjq-6wc02 (doi:10.22002/D1.20086) | CC BY 4.0 (CaltechDATA record, rights `cc-by-4.0`) | 2026-09-19 | `caltech-101.zip`, md5 `3138e1922a9193bfa496528edbbc45d0` | yes |
| `blur_ladder` | Synthetic Gaussian blur on Caltech-101 images that `caltech101` does not use | derived locally | CC BY 4.0 (derived from Caltech-101) | 2026-09-19 | blur radii frozen in `glance/evals/suites/blur_ladder.py` | yes |
| `doctype16` | RVL-CDIP subset (optional suite) | https://huggingface.co/datasets/aharley/rvl_cdip | **Unclear**: the Hub cards (`aharley/rvl_cdip`, `chainyo/rvl-cdip`) list `other`; the source collection (IIT-CDIP) has no clear reuse terms | 2026-09-19 | not downloaded | **no, skipped** |
| (score lab, evaluation only) | KADID-10k: 81 reference images x 25 distortions x 5 levels with DMOS | https://database.mmsp-kn.de/kadid-10k-database.html (`kadid10k.zip`, 3.07 GB) | **No formal license.** The page says "KADID-10k is freely available to the research community" and asks for citation (Lin, Hosu, Saupe, QoMEX 2019). Unclear under HANDOFF section 11. **Exception approved by the project owner on 2026-09-20: evaluation only**, never used to fit anything that ships, never redistributed; only item ids, levels, DMOS-derived metrics and model logits are committed | 2026-09-20 | zip downloaded 2026-09-20; sha256 recorded in `lab/NOTES.md` | yes, branch `score-lab` only |
| `human_gold` | Your own hand-labeled images | local `gold/human_gold.jsonl` | Private. Never leaves the machine; `gold/` is gitignored, and the frontier baseline refuses it without `--allow-upload-gold` | n/a | n/a | empty in this run |

Notes:

- The Hub mirror `lmms-lab/POPE` carries no license tag, so it was not used. POPE is rebuilt from the official
  MIT-licensed question files plus COCO's own image host.
- The Hub mirror `flwrlabs/caltech101` lists its license as `unknown`, so it was not used. The official CaltechDATA
  record states CC BY 4.0.
- ImageNet and other research-only sets are excluded by the hand-off. The one exception, approved explicitly by the
  project owner for evaluation only on the research branch, is KADID-10k (row above). It is not part of the v0 harness
  or of anything meant for commercial reuse.
- Public suites are likely in every model's training data. `human_gold` is the only uncontaminated check.
