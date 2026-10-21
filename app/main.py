"""
Entry point aplikasi FastAPI — IT Helpdesk Assistant.

Alur endpoint utama:
  1. Menerima pertanyaan/permintaan dari user
  2. Meneruskan ke router (router.py) untuk menentukan intent
  3. Router memanggil RAG (rag.py) atau tool (tools.py) sesuai intent
  4. Mengembalikan response terstruktur ke client
"""

import logging
from contextlib import asynccontextmanager

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI
from pydantic import BaseModel

from app.rag import build_index
from app.router import handle_query

logger = logging.getLogger("helpdesk-assistant")
logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Bangun index FAISS sekali saat startup (bukan di setiap request)."""
    try:
        build_index()
    except Exception as exc:
        # Aplikasi tetap start; request /ask akan mengembalikan response error.
        logger.error("Gagal membangun index FAISS: %s", exc)
    yield


app = FastAPI(title="IT Helpdesk Assistant API", lifespan=lifespan)


class AskRequest(BaseModel):
    query: str


class AskResponse(BaseModel):
    status: str
    intent: str
    data: str


@app.get("/health")
async def health():
    """Endpoint sederhana untuk memastikan service berjalan."""
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    """
    Endpoint utama. Menerima pertanyaan bebas dari user (bisa berupa
    pertanyaan informasi/kebijakan ATAU permintaan cek status tiket),
    lalu meneruskannya ke router untuk diproses.

    Error handling: LLM gagal merespons / API key belum dikonfigurasi /
    intent tidak dikenali dikembalikan sebagai response terstruktur
    (status: error), bukan crash.
    """
    try:
        result = await handle_query(request.query)
    except Exception as exc:
        logger.exception("Gagal memproses query: %r", request.query)
        return AskResponse(
            status="error",
            intent="unknown",
            data=(
                "Layanan AI sedang tidak dapat memproses permintaan Anda "
                "(kesalahan konfigurasi API key atau gangguan dari LLM). "
                "Silakan coba lagi nanti."
            ),
        )

    return AskResponse(**result)
