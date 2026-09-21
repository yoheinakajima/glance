# Publishing checklist (nothing here has been run; the repository has no remote)

Everything below is the owner's action. The assistant prepared the repository but did not create, push or publish anything.

## 1. Look before you publish (5 minutes)

- Project page as it will appear: open `site/index.html` locally, or the private preview linked in `docs/SITE_PLAN.md`.
- `README.md` (what visitors and their agents read first), `AGENTS.md`, `docs/CLAIMS.md` (every claim with its caveat).
- Author line and affiliation on the page (`tools/make_site.py`, search for `byline`) and in `CITATION.cff`.
- `git log --oneline | head -30` on `main` (the work branch `score-lab` is fast-forwarded into `main`).

## 2. Create the repository and push (one paste; makes the code PUBLIC)

```bash
cd ~/code/glance && gh repo create yoheinakajima/glance --public --source . --remote origin --description "Ask an open vision-language model typed questions about an image and get probabilities back, on your own machine." && git push -u origin main && git push origin score-lab --tags
```

Use `--private` instead of `--public` to stage it first; Pages on a private repository needs a paid plan.

## 3. Turn on the project page at glance.yohei.me

1. DNS at your registrar: a `CNAME` record, host `glance`, value `yoheinakajima.github.io`.
2. One paste (sets Pages to deploy from GitHub Actions, then runs the workflow that uploads `site/`):

```bash
cd ~/code/glance && gh api -X POST repos/yoheinakajima/glance/pages -f build_type=workflow >/dev/null; gh workflow run pages && gh run watch
```

3. In the repository's Settings -> Pages, confirm the custom domain `glance.yohei.me` (the file `site/CNAME` already asks
   for it) and tick "Enforce HTTPS" once the certificate is issued (can take up to an hour after DNS resolves).

## 4. Afterwards

- The README's clone line says `<this repository>`: replace it with the real URL (the assistant can do this once the
  repository exists).
- Optional next steps, prepared but not done: PyPI package under the free name `glance-vlm`; a Hugging Face Space demo.
- What is NOT in the repository on purpose: model weights, the photographs (re-fetched from the committed manifests with
  attribution), KADID-10k data, hosted models' answers (only right/wrong flags), your API keys (never stored anywhere).
