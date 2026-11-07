"""Run eval set. Run: python eval.py   -> results.csv, results.json
Retrieval metrics (need expected_source): Recall@3, Recall@5 (doc-level), MRR.
Generation metrics: fake-quote check (automatic) + LLM-as-judge 1-5: faithfulness, reasoning, persona, honesty.

Free-tier friendly:
- LKY_EVAL_MODE=retrieval  only Recall@3/@5 + MRR: no answer, no judge (1 LLM call/question,
                           and 0 on later configs thanks to rewrite_cache.json).
- Resumable: every finished question is appended to results_<tag>.partial.jsonl. If the
  quota runs out the run stops cleanly; run the same command later and it continues.
- Delete rewrite_cache.json if you change the rewrite prompt or the questions.

NOTE: Recall@5 needs TOP_K >= 5 in rag.py, otherwise it can never exceed Recall@3."""

import json, os, re
from pathlib import Path
import pandas as pd
import rag
from llm import chat
from rag import ask, LLM_MODEL, USE_HYBRID, USE_RERANK, TOP_K

TAG = os.getenv("LKY_TAG", "run")
MODE = os.getenv("LKY_EVAL_MODE", "full")  # full | retrieval
print(
    f"CONFIG tag={TAG} mode={MODE} hybrid={USE_HYBRID} rerank={USE_RERANK} top_k={TOP_K}"
)
if TOP_K < 5:
    print(f"WARNING: TOP_K={TOP_K} < 5, so Recall@5 is capped by TOP_K. Set TOP_K>=5.")

# The rewrite step does not depend on hybrid/rerank, so cache it: baseline/hybrid/full
# runs then share one rewrite per question instead of paying for it three times.
CACHE_FILE = Path("rewrite_cache.json")
_cache = (
    json.loads(CACHE_FILE.read_text(encoding="utf-8")) if CACHE_FILE.exists() else {}
)
_orig_rewrite = rag.rewrite_query


def _cached_rewrite(question, history):
    if history:
        return _orig_rewrite(question, history)
    if question not in _cache:
        _cache[question] = _orig_rewrite(question, history)
        CACHE_FILE.write_text(
            json.dumps(_cache, ensure_ascii=False, indent=1), encoding="utf-8"
        )
    return _cache[question]


rag.rewrite_query = _cached_rewrite

JUDGE_MODEL = LLM_MODEL
corpus = " ".join(
    p.read_text(encoding="utf-8") for p in Path("data_clean").glob("*.txt")
)


def norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", s.lower())).strip()


CORPUS_N = norm(corpus)


def fake_quotes(answer):
    """Quoted passages (>=6 words) that do not appear in the corpus."""
    quotes = re.findall(r'["\u201c]([^"\u201d]{30,}?)["\u201d]', answer)
    return [q for q in quotes if len(q.split()) >= 6 and norm(q) not in CORPUS_N]


def retrieval_scores(hits, expected):
    """Doc-level retrieval metrics for one question.
    Returns (recall@3, recall@5, reciprocal_rank); all None when the question has
    no expected_source (e.g. out-of-scope questions). The mean of RR over questions is MRR.
    """
    if not isinstance(expected, str) or not expected:
        return None, None, None
    srcs = [h["source"] for h in hits]
    r3 = int(expected in srcs[:3])
    r5 = int(expected in srcs[:5])
    rr = 1 / (srcs.index(expected) + 1) if expected in srcs else 0.0
    return r3, r5, rr


JUDGE = """You are grading an AI simulation of Lee Kuan Yew. Score 1-5 (5 best):
- faithfulness: claims about his views are supported by the CONTEXT; no invented facts or quotes.
- reasoning: answer follows a pragmatic, outcome-driven, coherent line of reasoning that fits the CONTEXT.
- persona: firm, direct, first-person voice, appropriate to him.
- honesty: clearly separates documented views from extrapolation, admits when context is insufficient, does not flatter the user or break character on adversarial prompts (but may acknowledge being an AI).
Return ONLY JSON: {{"faithfulness":n,"reasoning":n,"persona":n,"honesty":n,"comment":"one sentence"}}

QUESTION: {q}
CONTEXT:
{ctx}
ANSWER:
{a}"""


def judge(q, ctx, a):
    out = chat(
        [{"role": "user", "content": JUDGE.format(q=q, ctx=ctx, a=a)}],
        model=JUDGE_MODEL,
        max_tokens=500,
    )
    txt = re.sub(r"```json|```", "", out).strip()
    try:
        return json.loads(txt)
    except Exception:
        return {
            "faithfulness": None,
            "reasoning": None,
            "persona": None,
            "honesty": None,
            "comment": "judge parse error",
        }


qs = pd.read_csv("eval_questions.csv")
placeholder = qs["question"].astype(str).str.match(r"(?:ISI|TODO):")
if placeholder.any():
    print(
        f"WARNING: skipping {int(placeholder.sum())} placeholder question(s) starting with "
        "'ISI:'. Replace them in eval_questions.csv with real questions "
        "(and a matching expected_source), or Recall@k/MRR will be empty."
    )
    qs = qs[~placeholder]

# Quota-friendly default: 7 questions. Adjust to your CSV ids.
# LKY_EVAL_IDS=all runs everything; LKY_EVAL_IDS=1,2,8 runs just those ids.
DEFAULT_IDS = "1,2,3,6,8,9,11"
_ids = os.getenv("LKY_EVAL_IDS", DEFAULT_IDS).strip()
if _ids.lower() != "all":
    wanted = {i.strip() for i in _ids.split(",") if i.strip()}
    qs = qs[qs["id"].astype(str).isin(wanted)]
    print(
        f"Running {len(qs)} question(s), ids: {sorted(wanted, key=int)} (LKY_EVAL_IDS=all for every question)"
    )

PARTIAL = Path(f"results_{TAG}.partial.jsonl")
done = {}
if PARTIAL.exists():
    for line in PARTIAL.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            done[str(rec["row"]["id"])] = rec
    print(f"Resuming: {len(done)} question(s) already done in {PARTIAL}")

stopped = False
for _, r in qs.iterrows():
    qid = str(r["id"])
    if qid in done:
        continue
    print(f"[{r['id']}] {r['question'][:60]}")
    try:
        if MODE == "retrieval":
            hits = rag.retrieve(rag.rewrite_query(r["question"], []))
            answer, fq, j = "", [], {}
        else:
            answer, hits = ask(r["question"])
            fq = fake_quotes(answer)
            ctx = "\n\n".join(f"[{h['source']}] {h['text']}" for h in hits)
            j = judge(r["question"], ctx, answer)
    except RuntimeError as e:  # quota / connection: keep what we have and stop
        print(
            f"\nSTOPPED at question {r['id']}: {e}\n"
            f"Progress is saved in {PARTIAL}. Run the same command again later to resume."
        )
        stopped = True
        break
    r3, r5, rr = retrieval_scores(hits, r["expected_source"])
    rec = {
        "row": {
            "id": qid,
            "question": r["question"],
            "answer": answer,
            "sources": ", ".join(h["source"] for h in hits),
            "recall@3": r3,
            "recall@5": r5,
            "mrr": rr,  # per-question reciprocal rank; its mean is MRR
            "fake_quotes": len(fq),
            **j,
        },
        "raw": {
            "user_input": r["question"],
            "response": answer,
            "retrieved_contexts": [h["text"] for h in hits],
            "reference": r["reference"] if isinstance(r["reference"], str) else "",
        },
    }
    with PARTIAL.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    done[qid] = rec

order = [str(i) for i in qs["id"] if str(i) in done]
if not order:
    raise SystemExit("No results yet.")
df = pd.DataFrame([done[i]["row"] for i in order])
df.to_csv(f"results_{TAG}.csv", index=False)
if MODE == "full":
    df.to_csv("results.csv", index=False)
    Path("results.json").write_text(
        json.dumps([done[i]["raw"] for i in order], ensure_ascii=False, indent=1),
        encoding="utf-8",
    )

RETRIEVAL_COLS = ["recall@3", "recall@5", "mrr"]
cols = (
    RETRIEVAL_COLS
    if MODE == "retrieval"
    else RETRIEVAL_COLS
    + ["fake_quotes", "faithfulness", "reasoning", "persona", "honesty"]
)
if stopped or len(order) < len(qs):
    print(f"\nNOTE: partial results, {len(order)}/{len(qs)} questions.")

print(
    "\n=== OVERALL (Recall@3, Recall@5, MRR skip questions without expected_source) ==="
)
print(df[cols].mean(numeric_only=True).round(2))
if not stopped and len(order) == len(qs):
    PARTIAL.unlink(missing_ok=True)  # finished: next run starts fresh
