import sys
sys.path.append('.')
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from config import Config

class QdrantStore:
    def __init__(self):
        self.client = QdrantClient(Config.QDRANT_URL)

    def store_book_chunks(self, chunks, embeddings):
        points = [
            PointStruct(
                id=idx,
                vector=embedding,
                payload={
                    "text": chunk,
                    "chunk_id": idx,
                    "length": len(chunk)
                }
            )
            for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings))
        ]
        
        self.client.upsert(
            collection_name="book_chunks",
            points=points,
            wait=True  # Ensure write confirmation
        )

