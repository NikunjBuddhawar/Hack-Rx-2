import os
from uuid import uuid4
from qdrant_client import QdrantClient
from qdrant_client.models import (
    PointStruct, VectorParams, Distance,
    Filter, FieldCondition, MatchValue,
    PayloadSchemaType
)
from dotenv import load_dotenv

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "pdf_chunks"

def get_qdrant_client():
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

def ensure_collection():
    client = get_qdrant_client()
    existing = [col.name for col in client.get_collections().collections]
    if COLLECTION_NAME not in existing:
        client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="doc_id",
        field_schema=PayloadSchemaType.KEYWORD
    )

def add_sentences(doc_id: str, sentences: list, embeddings: list):
    client = get_qdrant_client()
    points = [
        PointStruct(
            id=str(uuid4()),
            vector=vec,
            payload={"text": text, "doc_id": doc_id}
        )
        for text, vec in zip(sentences, embeddings)
    ]
    client.upsert(collection_name=COLLECTION_NAME, points=points)

def search_similar(vector, doc_id: str, top_k=5):
    client = get_qdrant_client()
    hits = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=vector,
        limit=top_k,
        query_filter=Filter(
            must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
        )
    )
    return [{"score": h.score, "text": h.payload["text"]} for h in hits]
