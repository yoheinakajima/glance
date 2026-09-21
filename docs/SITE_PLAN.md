# Project page: plan and design notes

Owner's brief (2026-09-20): "a well designed beautiful minimalist non AI looking academic looking site". First draft is
built; private preview: https://claude.ai/artifact/HjAYTcWLY348UVL4rA447s (only the owner can open it until shared).

## How it is built

`uv run python tools/make_site.py` writes `site/index.html` (standalone, for static hosting such as GitHub Pages) and
`site/page.html` (the same content without the document wrapper, used for the private preview). EVERY number on the page
is read from a result file (`results/lab/*.json`, `lab/*.json`); the only literals are the three paired differences of
notebook entry 32b, marked in the code. Re-run it after any job lands; nothing is edited by hand in `site/`. No
JavaScript, no build step, no chart library: the two figures are inline SVG drawn to one scale by the generator.
Publishing anywhere public needs the owner's explicit go-ahead (there is no git remote yet).

## Design (so later edits stay in one voice)

- Register: a preprint, not a landing page. One left-aligned column (41rem measure), no hero, no cards, no gradients, no
  icons or emoji, no rounded boxes, nothing centered, no animation.
- Type: STIX Two Text (the face of scientific journals) for running text and headings; IBM Plex Sans for the apparatus
  (captions, table heads, intervals, chart labels); IBM Plex Mono for commands. One scale; headings carry a top rule and a
  section number because the page IS a numbered paper.
- Color: white paper, blue-black ink `#14171a`, blue-grey apparatus `#56616b`, rules `#d5dbe1`, link blue `#1a4480`, and one
  oxblood `#8f2d2d` used ONLY for the words "not supported". A designed dark theme mirrors it.
- Tables: booktabs (heavy top and bottom rule, thin mid rule, no vertical lines, no zebra), tabular figures, the open
  model's rows in semibold.
- The subject's own conventions, as content: every accuracy carries its 95% interval in brackets; figures are
  dot-and-whisker plots with filled marks for the open model and hollow marks for hosted models (reads in greyscale and
  for colour-blind readers); the "Limits and misses" section prints the registered verdicts, misses first.
- Wording follows `docs/paper/OUTLINE.md` ("Positioning, revised"): the finding is about the open model; Glance is the
  calibration and measurement harness; the readout is shared with Simple Jev, jev-visual and LitJev and not claimed as new;
  "level with", never "beats"; no "parity on ratings"; the ten-point loss of our zero-shot rating read to the same model's
  written answer is stated on the page.

## Sections now

Header (running head, title, one-sentence subtitle, author, in-page links) · Abstract · 1 Evidence (1.1 fresh photographs:
Table 1 + Figure 1; 1.2 ratings: Figure 2 + the exact-versus-within-one paragraph; 1.3 reading versus writing: Table 2) ·
2 What is, and is not, new · 3 Limits and misses · 4 Reproduce · footer (licenses, attribution, generator).

## To add as results land (each is a few lines in the generator)

1. iNaturalist frontier rows in Table 1 (after the owner's one paste: `uv run python tools/frontier_batch.py`).
2. E16, the one-pass read at the JSON answer position: replaces the "trails by ten points" sentence if it holds.
3. E15, model size (2B / 4B / 8B): a third figure, same dot-plot function, from `results/lab/scaling.json`.
4. E3 second model family and E13 outside open systems: one table, "portability", only when measured.
5. E4 creative-QA rubrics (crop, occlusion, tilt, caption legibility, watermark) zero-shot.
6. A method figure (image + typed question -> one forward pass -> logits at the answer position -> probabilities), drawn
   as plain SVG in the same line weight as the plots.
7. Links: paper PDF, code, BibTeX block, once they exist.

## Decisions that are the owner's

- Author line and affiliation (now: "Yohei Nakajima"); whether and how to credit AI assistance.
- Where it is hosted and under what address; when it becomes public.
- Whether the page should carry the project name alone or a paper title once the paper has one.
