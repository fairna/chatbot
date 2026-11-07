# Think Like Lee Kuan Yew (RAG + Eval)

    pip install -r requirements.txt
    export GEMINI_API_KEY=...                # Windows cmd: set GEMINI_API_KEY=...  (key from aistudio.google.com)
    # put the PDFs into ./data
    python ingest.py          # re-run after any change to documents or chunking
    streamlit run app.py

Pipeline
- Ingest: clean text -> chunk across pages (1000/150) -> embed -> Chroma (+ metadata)
- Query: rewrite (standalone, English) -> vector top-20 + BM25 top-20 -> RRF fusion
         -> cross-encoder rerank -> top-6 -> strict grounded prompt -> answer + citations

Compare configurations (edit eval_questions.csv first):
    LKY_HYBRID=0 LKY_RERANK=0 LKY_TAG=baseline python eval.py
    LKY_HYBRID=1 LKY_RERANK=0 LKY_TAG=hybrid   python eval.py
    LKY_HYBRID=1 LKY_RERANK=1 LKY_TAG=full     python eval.py
Windows (cmd): set LKY_HYBRID=0 && set LKY_RERANK=0 && set LKY_TAG=baseline && python eval.py
Results: results_<tag>.csv per run.

LLM: Google Gemini through its OpenAI-compatible endpoint (llm.py).
Change models with LLM_MODEL / LLM_FAST_MODEL. Any OpenAI-compatible provider works
(Groq, OpenRouter, Ollama) by setting LLM_BASE_URL, LLM_API_KEY, LLM_MODEL.
Free tier rate limits: set LLM_MIN_INTERVAL=7 (seconds between calls) before running eval.py.
