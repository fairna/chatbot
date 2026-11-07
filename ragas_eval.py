"""RAGAS metrics on results.json (run eval.py first). Run: python ragas_eval.py
RAGAS API changes between versions; written for ragas 0.2+/0.3. Check the docs if it errors.
"""

import json
from ragas import evaluate, EvaluationDataset
from ragas.metrics import Faithfulness, ResponseRelevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from llm import API_KEY, BASE_URL, MAIN_MODEL

data = json.load(open("results.json", encoding="utf-8"))
ds = EvaluationDataset.from_list(
    [
        {
            "user_input": d["user_input"],
            "response": d["response"],
            "retrieved_contexts": d["retrieved_contexts"],
        }
        for d in data
    ]
)

llm = LangchainLLMWrapper(
    ChatOpenAI(model=MAIN_MODEL, api_key=API_KEY, base_url=BASE_URL, temperature=0)
)
emb = LangchainEmbeddingsWrapper(
    HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
)

res = evaluate(
    ds, metrics=[Faithfulness(), ResponseRelevancy()], llm=llm, embeddings=emb
)
print(res)
res.to_pandas().to_csv("ragas_results.csv", index=False)
