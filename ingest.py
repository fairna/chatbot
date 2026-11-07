"""Read PDFs in ./data, clean, chunk across pages, embed, store in Chroma.
Run: python ingest.py   (re-run whenever documents or chunk settings change)"""

import re
import bisect
from pathlib import Path
import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

DB_DIR = "chroma_db"
COLLECTION = "lky"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
CHUNK_SIZE, CHUNK_OVERLAP = 1000, 150

DATA = Path("data")
CLEAN = Path("data_clean")  # cleaned text copies, used by eval to verify quotes
CLEAN.mkdir(exist_ok=True)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
)
emb = SentenceTransformer(EMBED_MODEL)

client = chromadb.PersistentClient(DB_DIR)
try:
    client.delete_collection(COLLECTION)
except Exception:
    pass
col = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})


def meta_from_name(stem):
    m = re.match(r"lky(\d{4})(\d{2})(\d{2})", stem)
    return ("answer", m.group(1)) if m else ("book", "")


def clean(text):
    text = re.sub(r"-\n(?=[a-z])", "", text)  # hyphenated line breaks
    text = "\n".join(
        l for l in text.splitlines() if not re.fullmatch(r"\s*\d{1,4}\s*", l)
    )  # bare page numbers
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


ids, docs, metas = [], [], []
for pdf in sorted(DATA.glob("*.pdf")):
    stem = pdf.stem
    dtype, year = meta_from_name(stem)
    pages = [clean(p.extract_text() or "") for p in PdfReader(pdf).pages]
    if sum(len(t) for t in pages) < 200:
        print(f"WARNING {pdf.name}: almost no text. Probably a scan, needs OCR.")
        continue

    full, starts = "", []  # join pages so chunks can cross page boundaries
    for t in pages:
        starts.append(len(full))
        full += t + "\n\n"
    (CLEAN / f"{stem}.txt").write_text(full, encoding="utf-8")

    pos = 0
    for n, chunk in enumerate(splitter.split_text(full)):
        found = full.find(chunk, pos)
        if found == -1:
            found = pos
        pos = found + 1
        page = max(1, bisect.bisect_right(starts, found))  # page where the chunk starts
        ids.append(f"{stem}-{n}")
        docs.append(chunk)
        metas.append({"source": stem, "type": dtype, "year": year, "page": page})
    print(f"{pdf.name}: {n + 1} chunks")

for i in range(0, len(docs), 64):
    batch = docs[i : i + 64]
    col.add(
        ids=ids[i : i + 64],
        documents=batch,
        embeddings=emb.encode(batch, normalize_embeddings=True).tolist(),
        metadatas=metas[i : i + 64],
    )
print(f"Done. {len(docs)} chunks stored.")
