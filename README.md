# glance

Image decision harness v0. Takes image(s) plus typed questions (`noul`, `choice`, `score`) and returns
probability distributions read from single forward passes of an open VLM or a dual encoder. No text
generation, no training. `HANDOFF.md` is the spec; `STATUS.md` is the build log.

```bash
uv sync
uv run glance doctor --json
uv run glance decide samples/request.json --model siglip
uv run glance eval --suite pope --n 200 --model vlm
uv run glance serve
```
