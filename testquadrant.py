import os
from qdrant_client import QdrantClient
from dotenv import load_dotenv

load_dotenv()

qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)

print("✅ Connected successfully!")
print("Collections:", qdrant_client.get_collections())