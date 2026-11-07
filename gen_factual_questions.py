"""Draft the factual eval questions (rows with TODO:/ISI: placeholders) from YOUR documents.
Run after ingest.py (needs ./data_clean/*.txt):   python gen_factual_questions.py
For each placeholder row it picks a random passage from the row's expected_source, asks the
LLM for ONE specific question + short answer grounded in that passage, and writes them into
eval_questions.csv (backup: eval_questions.csv.bak). Costs 1 LLM call per placeholder row.
Rows whose expected_source is not found (e.g. book_5) fall back to the first non-speech
file in data_clean. READ THE RESULT and edit anything vague before running eval.py."""
import json, random, re, shutil
from pathlib import Path
import pandas as pd
from llm import chat, MAIN_MODEL

random.seed(7)
CSV = Path("eval_questions.csv")
CLEAN = Path("data_clean")
WINDOW = 1400

PROMPT = """Below is a passage from a document in a Lee Kuan Yew corpus.
Write ONE factual question that can be answered from this passage, plus a short reference answer.
Rules:
- The question must be specific (a name, number, place, date, decision or stated reason).
- It must make sense on its own: never say "the passage", "this speech", "the text" or "the document".
- Do not copy a sentence from the passage word for word; paraphrase.
- The answer must be 1-2 sentences, fully supported by the passage.
- English only.
Return ONLY JSON: {{"question": "...", "answer": "..."}}

PASSAGE:
{p}"""


def pick_passage(text):
    text = re.sub(r"\s+", " ", text).strip()
    best = None
    for _ in range(30):                      # prefer dense, prose-like windows
        start = random.randint(0, max(0, len(text) - WINDOW))
        w = text[start:start + WINDOW]
        letters = sum(c.isalpha() for c in w) / max(1, len(w))
        if letters > 0.75:
            return w
        best = best or w
    return best


def source_text(stem):
    f = CLEAN / f"{stem}.txt"
    if not f.exists():
        books = sorted(p for p in CLEAN.glob("*.txt") if not p.stem.startswith("lky"))
        if not books:
            return None, None
        f = books[0]
        print(f"  '{stem}' not found, using {f.stem} instead")
    return f.stem, f.read_text(encoding="utf-8")


qs = pd.read_csv(CSV, dtype=str, keep_default_na=False)
todo = qs["question"].str.match(r"(?:ISI|TODO):")
if not todo.any():
    raise SystemExit("No TODO:/ISI: rows left in eval_questions.csv.")
shutil.copy(CSV, str(CSV) + ".bak")
print(f"Model: {MAIN_MODEL} | drafting {int(todo.sum())} question(s)")

for i in qs.index[todo]:
    stem, text = source_text(qs.at[i, "expected_source"])
    if text is None:
        print(f"[{qs.at[i, 'id']}] no text available in data_clean, skipped")
        continue
    out = chat([{"role": "user", "content": PROMPT.format(p=pick_passage(text))}],
               model=MAIN_MODEL, max_tokens=400)
    try:
        d = json.loads(re.sub(r"```json|```", "", out).strip())
        q, a = d["question"].strip(), d["answer"].strip()
    except Exception:
        print(f"[{qs.at[i, 'id']}] could not parse LLM output, left as is: {out[:100]!r}")
        continue
    qs.at[i, "question"], qs.at[i, "reference"], qs.at[i, "expected_source"] = q, a, stem
    print(f"[{qs.at[i, 'id']}] {stem}\n   Q: {q}\n   A: {a}")

qs.to_csv(CSV, index=False)
print(f"\nWritten to {CSV}. Review the questions before running eval.py.")
