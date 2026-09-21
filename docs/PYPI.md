# Publishing `glance-vlm` to PyPI (instructions for an assistant with a browser, written 2026-09-21)

**DONE on 2026-09-21: `glance-vlm` 0.3.0 is on PyPI.** For the next release: raise `version` in `pyproject.toml`, commit, push, and run Steps 4 and 5 again
(the trusted publishers and the two environments already exist; Steps 1 to 3 are history). The repository is public now; the sentence in Rule 3 about it being private is history too.

You are publishing a Python package for Yohei Nakajima. Read all of this before doing anything.

**What is already done** (do not redo): the package builds as `glance-vlm` version 0.3.0 (the name `glance` is taken on PyPI;
`glance-vlm` was free on 2026-09-21; the import name and the command stay `glance`). The wheel was installed into a clean
environment and answered real questions with the real model. A GitHub Actions workflow, `.github/workflows/publish.yml`,
builds and uploads with Trusted Publishing, so **no API token is ever created, copied or stored**.

## Rules

1. Never create, view, copy or paste an API token or a password. If any page asks you to log in, confirm a password, do
   two-factor authentication, or accept terms, STOP and ask Yohei to do that step himself in the browser, then continue.
2. Do not create accounts. If Yohei has no account on pypi.org or test.pypi.org, stop and tell him (they are separate sites
   with separate accounts; both need two-factor authentication turned on before they allow publishing).
3. Do not change the GitHub repository's visibility, its Pages settings, or anything else outside the steps below. The
   repository is private on purpose; Trusted Publishing works from a private repository.
4. A version can be uploaded to PyPI exactly once and can never be replaced. Publish to TestPyPI first, check it, and ask
   Yohei for an explicit "go" before the real PyPI upload.
5. If a step fails in a way this file does not cover, stop and report the exact error text. Do not improvise around it.

## Step 0. Check the starting point (terminal)

```bash
cd ~/code/glance && git status --short | grep -v "^??" ; git log --oneline -1
curl -s -o /dev/null -w "glance-vlm on PyPI: %{http_code} (404 means the name is still free)\n" https://pypi.org/pypi/glance-vlm/json
grep -n '^name\|^version' pyproject.toml
```

Expected: no modified tracked files, `name = "glance-vlm"`, `version = "0.3.0"`, and 404. If the name is no longer free, stop.
The workflow file must be on GitHub's default branch: `gh api repos/yoheinakajima/glance/contents/.github/workflows/publish.yml --jq .name`
should print `publish.yml`.

## Step 1. Tell TestPyPI to trust the workflow (browser)

Open https://test.pypi.org/manage/account/publishing/ . Under "Add a new pending publisher", choose the GitHub tab and enter
exactly:

| Field | Value |
| --- | --- |
| PyPI Project Name | `glance-vlm` |
| Owner | `yoheinakajima` |
| Repository name | `glance` |
| Workflow name | `publish.yml` |
| Environment name | `testpypi` |

Press Add. The page should now list a pending publisher for `glance-vlm`.

## Step 2. The same on the real PyPI (browser)

Open https://pypi.org/manage/account/publishing/ and add the same pending publisher, with Environment name `pypi`.

## Step 3. Create the two GitHub environments (terminal)

```bash
gh api -X PUT repos/yoheinakajima/glance/environments/testpypi > /dev/null && gh api -X PUT repos/yoheinakajima/glance/environments/pypi > /dev/null && gh api repos/yoheinakajima/glance/environments --jq '.environments[].name'
```

Optional, recommended: in the repository's Settings -> Environments -> `pypi`, add Yohei as a required reviewer, so the
real upload waits for his click.

## Step 4. Publish to TestPyPI and check it (terminal)

```bash
gh workflow run publish.yml --repo yoheinakajima/glance -f target=testpypi && sleep 8 && gh run watch --repo yoheinakajima/glance $(gh run list --repo yoheinakajima/glance --workflow publish.yml --limit 1 --json databaseId --jq '.[0].databaseId')
```

Then install it from TestPyPI into a throwaway environment and run it (dependencies come from the real PyPI):

```bash
rm -rf /tmp/glance-testpypi && uv venv -q --python 3.11 /tmp/glance-testpypi && uv pip install -q --python /tmp/glance-testpypi/bin/python --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ --index-strategy unsafe-best-match "glance-vlm==0.3.0" && /tmp/glance-testpypi/bin/glance doctor | tail -4
```

Expected: `glance doctor` prints the selected model tier and no error. Also open https://test.pypi.org/project/glance-vlm/ and
check that the README renders (relative links to files in the repository will not resolve there; that is expected).

## Step 5. Ask Yohei for a go, then publish to PyPI (terminal)

Show him the TestPyPI page and the output of Step 4. Only after he says go:

```bash
gh workflow run publish.yml --repo yoheinakajima/glance -f target=pypi && sleep 8 && gh run watch --repo yoheinakajima/glance $(gh run list --repo yoheinakajima/glance --workflow publish.yml --limit 1 --json databaseId --jq '.[0].databaseId')
```

If the `pypi` environment has a required reviewer, the run waits until Yohei approves it on the run's page. Then verify:

```bash
rm -rf /tmp/glance-pypi && uv venv -q --python 3.11 /tmp/glance-pypi && uv pip install -q --python /tmp/glance-pypi/bin/python glance-vlm && /tmp/glance-pypi/bin/glance doctor | tail -4 && /tmp/glance-pypi/bin/glance --help | head -5
```

## Step 6. Report back

Report: the two project URLs (https://test.pypi.org/project/glance-vlm/ and https://pypi.org/project/glance-vlm/), the version
published, the output of the last verify command, and anything that looked wrong. Do not edit the repository; the install line
for the README (`pip install glance-vlm`, Python 3.11) is added by whoever maintains the repository once the upload is confirmed.

## If something fails

- `invalid-publisher` or "not a trusted publisher" in the publish step: one of the five fields in Step 1 or 2 does not match
  (most often the environment name or the workflow file name). Fix the pending publisher on the PyPI side; do not switch to tokens.
- "File already exists": that version is already uploaded. Stop and report; publishing again needs a new version number in
  `pyproject.toml`, which is the maintainer's change to make.
- The workflow is not found by `gh workflow run`: the file is not on the default branch yet. Stop and report.
- `uv pip install` from TestPyPI cannot resolve dependencies: keep the `--extra-index-url` and `--index-strategy` flags exactly as written.
- Python version: the package declares Python 3.11 only (the only version it was tested on). That is intended for this release.
