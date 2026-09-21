# Publishing checklist

**State on 2026-09-21, about 09:00: PUBLISHED.** The owner made `github.com/yoheinakajima/glance` public, `glance-vlm` 0.3.0 is on
PyPI (published with Trusted Publishing by the owner's browser-capable assistant, `docs/PYPI.md`; checked: it installs into a clean
Python 3.11 environment and `glance doctor` runs), and the page is live at https://glance.yohei.me/ with HTTPS. Everything pushed to
this repository is now public at once. The page still deploys ONLY on a manual run of the `pages` workflow (section 3b), and only the
owner decides when. The sections below are the record of how it was set up.

Going public and turning on the page are the owner's actions. The assistant pushed to a private repository only.

## 1. Look before you publish (5 minutes)

- Project page as it will appear: open `site/index.html` locally, or the private preview linked in `docs/SITE_PLAN.md`.
- `README.md` (what visitors and their agents read first), `AGENTS.md`, `docs/CLAIMS.md` (every claim with its caveat).
- Author line and affiliation on the page (`tools/make_site.py`, search for `byline`) and in `CITATION.cff`.
- `git log --oneline | head -30` on `main` (the work branch `score-lab` is fast-forwarded into `main`).

## 2. The repository exists and is PRIVATE

At the owner's request (2026-09-20, late evening) the assistant created `github.com/yoheinakajima/glance` as a PRIVATE
repository and pushed `main` (fast-forwarded from `score-lab`) and `score-lab`. Nothing is public. To publish the code:

```bash
gh repo edit yoheinakajima/glance --visibility public --accept-visibility-change-consequences
```

## 3. Turn on the project page at glance.yohei.me

1. DNS at your registrar: a `CNAME` record, host `glance`, value `yoheinakajima.github.io`.
2. One paste (sets Pages to deploy from GitHub Actions, then runs the workflow that uploads `site/`):

```bash
cd ~/code/glance && gh api -X POST repos/yoheinakajima/glance/pages -f build_type=workflow >/dev/null; gh workflow run pages && gh run watch
```

3. In the repository's Settings -> Pages, confirm the custom domain `glance.yohei.me` (the file `site/CNAME` already asks
   for it) and tick "Enforce HTTPS" once the certificate is issued (can take up to an hour after DNS resolves).

## 3b. The page is already live once; to update it (state on 2026-09-21)

The owner set up Pages, DNS and the custom domain on 2026-09-20 and deployed once by hand (about 00:13). Pushes do NOT
redeploy on their own (the workflow deploys on a manual run, or on push only if the repository variable `PAGES_AUTO` is
`true`). Everything committed since then (the insect-order test with six hosted models, the second model family on
photographs, table numbering, the overnight results) goes live with one paste:

```bash
cd ~/code/glance && gh workflow run pages --repo yoheinakajima/glance && sleep 5 && gh run watch --repo yoheinakajima/glance
```

Then, in Settings -> Pages, tick "Enforce HTTPS" if the certificate has been issued since.

## 4. Afterwards

- Optional next steps, prepared but not done: PyPI package under the free name `glance-vlm`; a Hugging Face Space demo.
- What is NOT in the repository on purpose: model weights, the photographs (re-fetched from the committed manifests with
  attribution), KADID-10k data, hosted models' answers (only right/wrong flags), your API keys (never stored anywhere).
