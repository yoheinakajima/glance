# Measured cost and speed of the frontier calls

| Model | Task | calls | median seconds per answer | measured $ per 1,000 answers | median output tokens |
| --- | --- | --- | --- | --- | --- |
| anthropic/claude-opus-5 | lab scales, one 4-level rating per call | 1000 | 2.4 | not logged (run predates cost logging) | - |
| openai/gpt-5.6 | lab scales, one 4-level rating per call | 1000 | 1.1 | not logged (run predates cost logging) | - |
| openrouter/google/gemini-3.1-pro-preview | lab scales, one 4-level rating per call | 999 | 4.0 | $6.15 | 192 |
| anthropic/claude-opus-5 | fresh photos, one yes/no or pick-one per call | 196 | 2.7 | not logged (run predates cost logging) | - |
| openai/gpt-5.6 | fresh photos, one yes/no or pick-one per call | 196 | 1.2 | $7.64 | 12 |
| openrouter/google/gemini-3.1-pro-preview | fresh photos, one yes/no or pick-one per call | 196 | 2.9 | $3.03 | 12 |
| anthropic/claude-opus-5 | iNaturalist photos (about 500 px), one yes/no or pick-one per call | 300 | 2.3 | $3.74 | 13 |
| openai/gpt-5.6 | iNaturalist photos (about 500 px), one yes/no or pick-one per call | 300 | 1.2 | $1.86 | 12 |
| openrouter/google/gemini-3.1-pro-preview | iNaturalist photos (about 500 px), one yes/no or pick-one per call | 300 | 2.8 | $3.11 | 49 |
| anthropic/claude-haiku-4-5 | iNaturalist photos (about 500 px), one yes/no or pick-one per call | 300 | 0.7 | $0.60 | 10 |
| openai/gpt-5.6-luna | iNaturalist photos (about 500 px), one yes/no or pick-one per call | 300 | 0.9 | $0.10 | 12 |
| openrouter/google/gemini-3.1-flash-lite | iNaturalist photos (about 500 px), one yes/no or pick-one per call | 300 | 1.5 | $0.33 | 12 |
| anthropic/claude-haiku-4-5 | fresh photos, one yes/no or pick-one per call | 196 | 0.9 | $1.79 | 10 |
| openai/gpt-5.6-luna | fresh photos, one yes/no or pick-one per call | 196 | 1.2 | $0.38 | 12 |
| openrouter/google/gemini-3.1-flash-lite | fresh photos, one yes/no or pick-one per call | 196 | 1.8 | $0.32 | 12 |
| anthropic/claude-haiku-4-5 | lab scales, one 4-level rating per call | 1000 | 0.7 | $0.55 | 10 |
| openai/gpt-5.6-luna | lab scales, one 4-level rating per call | 1000 | 1.1 | $0.12 | 44 |
| openrouter/google/gemini-3.1-flash-lite | lab scales, one 4-level rating per call | 1000 | 1.6 | $0.33 | 7 |
