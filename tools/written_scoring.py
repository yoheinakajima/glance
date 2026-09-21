"""Strict and lenient scoring of an open model's WRITTEN answers (E25, `lab/NOTES.md` entry 59; both rules fixed before any output).

Strict is the rule every written row follows: an unparsable or invalid answer is wrong. Lenient exists so that a comparison cannot be
blamed on formatting: when strict parsing fails, the answer is the first allowed answer that appears in the raw text as a whole
word, case-insensitive; none found is wrong. Rows come from `glance.lab.gen_accuracy` (fields `valid`, `correct`, `text`, `allowed`,
`label`).
"""
import re


def strict(row) -> bool:
    return bool(row["correct"])


def lenient(row) -> bool:
    if row["valid"] or "text" not in row:
        return bool(row["correct"])
    text = re.sub(r"```[a-z]*", " ", row["text"].lower())
    first = None
    for answer in row["allowed"]:
        m = re.search(r"(?<![a-z0-9_])" + re.escape(str(answer).lower()) + r"(?![a-z0-9_])", text)
        if m and (first is None or m.start() < first[0]):
            first = (m.start(), str(answer))
    return first is not None and first[1] == str(row["label"])
