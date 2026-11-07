"""Core RAG: hybrid retrieval (vector + BM25) -> rerank -> grounded generation.
Env flags for A/B eval: LKY_HYBRID=0/1, LKY_RERANK=0/1, LKY_TOPK, LKY_CANDIDATES,
LKY_LOW_CONF, LKY_TEMPERATURE. LLM settings live in llm.py."""

import os
import re
import math
import time
import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from llm import (
    chat,
    chat_stream,
    log,
    MAIN_MODEL as LLM_MODEL,
    FAST_MODEL as REWRITE_MODEL,
)

DB_DIR = "chroma_db"
COLLECTION = "lky"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
RERANK_MODEL = os.getenv("LKY_RERANK_MODEL", "BAAI/bge-reranker-base")
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

USE_HYBRID = os.getenv("LKY_HYBRID", "1") == "1"
USE_RERANK = os.getenv("LKY_RERANK", "1") == "1"
TOP_K = int(os.getenv("LKY_TOPK", "6"))
CANDIDATES = int(os.getenv("LKY_CANDIDATES", "20"))
LOW_CONF = float(
    os.getenv("LKY_LOW_CONF", "0.05")
)  # rerank score (0-1) below which we warn the LLM
TEMPERATURE = float(os.getenv("LKY_TEMPERATURE", "0.3"))

_emb = SentenceTransformer(EMBED_MODEL)
_db = chromadb.PersistentClient(DB_DIR).get_or_create_collection(
    COLLECTION, metadata={"hnsw:space": "cosine"}
)
SYSTEM = open("persona_prompt.txt", encoding="utf-8").read()

_all = _db.get(include=["documents", "metadatas"])
_ids, _docs, _metas = _all["ids"], _all["documents"], _all["metadatas"]
_pos = {i: n for n, i in enumerate(_ids)}

_STOP = set(
    "a an and are as at be but by for from had has have he i in is it its my of on or our that the their this to was we were what which who will with you your how do does".split()
)


def _tok(text):
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOP]


_bm25 = BM25Okapi([_tok(d) for d in _docs]) if (USE_HYBRID and _docs) else None
_reranker = CrossEncoder(RERANK_MODEL) if USE_RERANK else None

# Warm up once at import so the first user question doesn't pay for lazy initialisation.
_emb.encode(["warm up"], normalize_embeddings=True)
if _reranker:
    _reranker.predict([("warm up", "warm up")])


def rewrite_query(question, history):
    """Standalone English query: resolves follow-ups and translates."""
    convo = "\n".join(f"{m['role']}: {m['content'][:500]}" for m in history[-4:])
    prompt = (
        "Rewrite the user's last question as a standalone English search query "
        "for a corpus of Lee Kuan Yew documents (1964-1968). Resolve references "
        "using the conversation. For modern topics, add related themes that "
        "existed in 1964-1968 (e.g. technology -> industrialisation, labour, "
        "foreign investment). Output only the query.\n\n"
        f"Conversation:\n{convo}\n\nLast question: {question}"
    )
    out = chat(
        [{"role": "user", "content": prompt}], model=REWRITE_MODEL, max_tokens=200
    )
    return out.strip() or question


def _vector_ids(query, n):
    vec = _emb.encode([QUERY_PREFIX + query], normalize_embeddings=True).tolist()
    res = _db.query(query_embeddings=vec, n_results=min(n, len(_ids)))
    return res["ids"][0]


def _bm25_ids(query, n):
    scores = _bm25.get_scores(_tok(query))
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]
    return [_ids[i] for i in order if scores[i] > 0]


def _rrf(rankings, k=60):
    """Reciprocal Rank Fusion: merge rankings without comparing raw scores."""
    score = {}
    for ranking in rankings:
        for rank, id_ in enumerate(ranking):
            score[id_] = score.get(id_, 0) + 1 / (k + rank + 1)
    return sorted(score, key=score.get, reverse=True)


def _chunk(id_):
    n = _pos[id_]
    return {"id": id_, "text": _docs[n], **_metas[n]}


def retrieve(query, k=TOP_K):
    n = CANDIDATES if (USE_HYBRID or USE_RERANK) else k
    rankings = [_vector_ids(query, n)]
    if USE_HYBRID and _bm25 is not None:
        rankings.append(_bm25_ids(query, n))
    fused = _rrf(rankings)[:CANDIDATES] if len(rankings) > 1 else rankings[0]
    cands = [_chunk(i) for i in fused]
    if USE_RERANK and cands:
        scores = _reranker.predict([(query, c["text"]) for c in cands])
        for c, s in zip(cands, scores):
            c["score"] = 1 / (1 + math.exp(-float(s)))
        cands.sort(key=lambda c: c["score"], reverse=True)
    return cands[:k]


def build_context(hits):
    return "\n\n".join(
        f"[source: {h['source']} | type: {h['type']} | year: {h['year']} | page: {h['page']}]\n{h['text']}"
        for h in hits
    )


def _prepare(question, history):
    """Rewrite + retrieve + build the LLM messages. Returns (msgs, hits)."""
    history = history or []
    t0 = time.time()
    query = rewrite_query(question, history)
    t1 = time.time()
    hits = retrieve(query)
    t2 = time.time()
    log(
        f"rewrite {t1 - t0:.1f}s | retrieve+rerank {t2 - t1:.1f}s | query: {query[:80]!r}"
    )

    note = ""
    if USE_RERANK and (not hits or hits[0].get("score", 1) < LOW_CONF):
        note = (
            "\n<retrieval_note>Confidence is low: the documents probably do not "
            "cover this topic.</retrieval_note>"
        )

    msgs = [{"role": m["role"], "content": m["content"]} for m in history]
    msgs.append(
        {
            "role": "user",
            "content": f"<context>\n{build_context(hits)}\n</context>{note}\n\nQuestion: {question}",
        }
    )
    return msgs, hits


def ask(question, history=None):
    """Returns (answer, hits). history = [{'role','content'}, ...]."""
    msgs, hits = _prepare(question, history)
    answer = chat(
        msgs, system=SYSTEM, model=LLM_MODEL, max_tokens=1500, temperature=TEMPERATURE
    )
    return answer, hits


def ask_stream(question, history=None):
    """Returns (text_generator, hits): the answer arrives token by token."""
    msgs, hits = _prepare(question, history)
    return (
        chat_stream(
            msgs,
            system=SYSTEM,
            model=LLM_MODEL,
            max_tokens=1500,
            temperature=TEMPERATURE,
        ),
        hits,
    )
