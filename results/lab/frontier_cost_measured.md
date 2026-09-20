# Measured cost and speed of the frontier calls

| Model | Task | calls | median seconds per answer | measured $ per 1,000 answers | median output tokens |
| --- | --- | --- | --- | --- | --- |
| anthropic/claude-opus-5 | lab scales, one 4-level rating per call | 1000 | 2.4 | not logged (run predates cost logging) | - |
| openai/gpt-5.6 | lab scales, one 4-level rating per call | 1000 | 1.1 | not logged (run predates cost logging) | - |
| openrouter/google/gemini-3.1-pro-preview | lab scales, one 4-level rating per call | 999 | 4.0 | $6.15 | 192 |
| anthropic/claude-opus-5 | fresh photos, one yes/no or pick-one per call | 196 | 2.7 | not logged (run predates cost logging) | - |
| openai/gpt-5.6 | fresh photos, one yes/no or pick-one per call | 196 | 1.2 | $7.64 | 12 |
| openrouter/google/gemini-3.1-pro-preview | fresh photos, one yes/no or pick-one per call | 196 | 2.9 | $3.03 | 12 |
