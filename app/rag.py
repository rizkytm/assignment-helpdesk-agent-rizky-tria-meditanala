"""
Modul RAG — mengubah knowledge base menjadi embedding, menyimpannya ke FAISS,
melakukan semantic search, lalu menghasilkan jawaban menggunakan LLM.

Alur ini serupa dengan yang sudah dibangun di Assignment 3 (Semantic Search &
LLM Application) dan Sesi 11 (Building a Simple RAG System) — di assignment ini,
alur tersebut dibungkus menjadi fungsi yang dipanggil oleh router.py.
"""

import os
from pathlib import Path

import pandas as pd
import numpy as np
import faiss
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

EMBEDDING_MODEL = "gemini-embedding-001"
GENERATION_MODEL = "gemini-3.1-flash-lite"
TEMPERATURE = 0.3

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
KNOWLEDGE_BASE_PATH = DATA_DIR / "it_helpdesk_knowledge_base.csv"

_index = None
_documents = None
_df = None
_client = None


def _get_client() -> genai.Client:
    """Client Gemini dibuat lazy supaya aplikasi tetap bisa start saat API key
    belum dikonfigurasi — kegagalannya baru muncul sebagai response error di /ask."""
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY belum diatur di environment variable.")
        _client = genai.Client(api_key=api_key)
    return _client


def _get_embedding(text: str):
    result = _get_client().models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
    )
    return result.embeddings[0].values


def build_index():
    """
    Membangun index FAISS dari knowledge base CSV.
    Referensi: starter_notebook_semantic_search.ipynb Assignment 3, Section 1-4
    (load data -> unit retrievable -> embedding -> simpan ke FAISS).
    Dipanggil satu kali saat aplikasi startup, bukan di setiap request.
    """
    global _index, _documents, _df

    _df = pd.read_csv(KNOWLEDGE_BASE_PATH)
    _df["content"] = "Q: " + _df["question"] + " A: " + _df["answer"]
    _documents = _df["content"].tolist()

    embeddings = np.array(
        [_get_embedding(doc) for doc in _documents], dtype="float32"
    )

    _index = faiss.IndexFlatL2(embeddings.shape[1])
    _index.add(embeddings)

    if _index.ntotal != len(_df):
        raise RuntimeError(
            f"Jumlah vektor di index ({_index.ntotal}) tidak sama dengan "
            f"jumlah entri knowledge base ({len(_df)})."
        )

    print(f"Index FAISS siap: {_index.ntotal} vektor, dimensi {embeddings.shape[1]}")
    return _index


def semantic_search(query: str, top_k: int = 3):
    """
    Semantic search terhadap index yang sudah dibangun.
    Referensi: starter_notebook_semantic_search.ipynb, Section 5.
    """
    if _index is None:
        raise RuntimeError("Index FAISS belum dibangun — build_index() belum dijalankan.")

    query_embedding = np.array([_get_embedding(query)], dtype="float32")
    distances, indices = _index.search(query_embedding, top_k)

    results = []
    for rank, idx in enumerate(indices[0]):
        if idx == -1:
            continue
        results.append(
            {
                "rank": rank + 1,
                "question": _df.iloc[idx]["question"],
                "answer": _df.iloc[idx]["answer"],
                "category": _df.iloc[idx]["category"],
                "distance": float(distances[0][rank]),
            }
        )
    return results


def _build_prompt(query: str, search_results: list) -> str:
    context = "\n\n".join(
        f"FAQ {r['rank']}: {r['question']}\nJawaban: {r['answer']}"
        for r in search_results
    )
    return f"""Persona: Kamu adalah asisten IT Helpdesk internal yang ramah dan profesional.

Task: Jawab pertanyaan karyawan HANYA berdasarkan Context di bawah. Jika informasi tidak ditemukan dalam Context, katakan dengan jujur bahwa kamu belum memiliki informasi tersebut dan sarankan pengguna menghubungi tim IT Helpdesk — jangan mengarang jawaban dari pengetahuan umum.

Context:
{context}

Format: Jawab dalam bahasa Indonesia yang natural dan ringkas (2-4 kalimat atau langkah singkat), langsung menjawab pertanyaan. Jangan menambahkan informasi di luar Context.

Pertanyaan karyawan: {query}

Jawaban:"""


async def answer_from_knowledge_base(query: str) -> str:
    """
    Fungsi yang dipanggil router.py ketika intent = 'faq'.
    Menjalankan semantic search, menyusun prompt dengan context hasil retrieval,
    lalu memanggil LLM untuk menghasilkan jawaban.
    """
    search_results = semantic_search(query)
    prompt = _build_prompt(query, search_results)

    response = await _get_client().aio.models.generate_content(
        model=GENERATION_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=TEMPERATURE),
    )
    return response.text
