from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from qdrant_helper import ensure_collection, add_sentences, search_similar
from embedder import embed_sentences
from pdf_reader import extract_sentences_from_pdf_url
import uuid
import os
from dotenv import load_dotenv
import httpx
from starlette.concurrency import run_in_threadpool

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    await run_in_threadpool(ensure_collection)

class QARequest(BaseModel):
    documents: str
    questions: List[str]

@app.post("/ask_pdf_url")
async def ask_pdf_url(request: QARequest):
    try:
        # Offload all blocking operations to threadpool
        sentences = await run_in_threadpool(extract_sentences_from_pdf_url, request.documents)
        if not sentences:
            return {"error": "No content found in the PDF."}

        doc_id = str(uuid.uuid4())

        sentence_embeddings = await run_in_threadpool(embed_sentences, sentences)
        await run_in_threadpool(add_sentences, doc_id, sentences, sentence_embeddings)

        all_question_embeddings = await run_in_threadpool(embed_sentences, request.questions)
        all_contexts = []

        for question_embedding in all_question_embeddings:
            results = await run_in_threadpool(search_similar, question_embedding, doc_id, 3)
            context = "\n".join([r['text'] for r in results])[:700]
            all_contexts.append(context)

        # Build the prompt
        combined_prompt = "Use the provided context to answer each question. Keep answers concise.\n\n"
        for i, (q, ctx) in enumerate(zip(request.questions, all_contexts)):
            combined_prompt += f"""### Question {i+1}:
{q}

### Context:
{ctx}

### Answer:\n\n"""

        async with httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=60.0, read=600.0, write=600.0)) as client:
            response = await client.post(
                GROQ_URL,
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "llama3-8b-8192",
                    "messages": [{"role": "user", "content": combined_prompt}],
                    "temperature": 0.2
                }
            )
            response.raise_for_status()
            data = response.json()
            full_reply = data["choices"][0]["message"]["content"].strip()

        return {"answers": full_reply}

    except Exception as e:
        return {"error": str(e)}
