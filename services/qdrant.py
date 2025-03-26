import sys
sys.path.append('.')
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from config import Config
from services.embeddings import GeminiEmbedder
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class QdrantManager:
    def __init__(self):
        logging.info("Initializing QdrantManager")
        self.client = QdrantClient(url=Config.QDRANT_URL)
        self._ensure_collections()
    
    def _ensure_collections(self,collection=None):
        collections=[]
        if collection:
            collections=[collection]
        if not collections:
            collections = ["book_chunks", "conversations"]
        for name in collections:
            if not self.client.collection_exists(collection_name=name):
                logging.info(f"Creating missing collection: {name}")
                self.client.create_collection(
                    collection_name=name,
                    vectors_config=VectorParams(size=768, distance=Distance.COSINE)
                )
            else:
                logging.info(f"Collection {name} already exists")
    
    
    def store_chunks(self, chunks: list[str], collection: str = "book_chunks", similarity_threshold: float = 0.9):
        """
        Store chunks in the specified collection, updating similar memories if found.
        """
        if not chunks:
            logging.warning("No chunks provided for storage")
            return
        
        logging.info(f"Processing {len(chunks)} chunks for storage in collection {collection}")
        points_to_upsert = []

        self._ensure_collections(collection)
        
        for idx, chunk in enumerate(chunks):
            chunk=chunk["text"]
            vector = GeminiEmbedder.embed(text=chunk)
            search_results = self.search_memories(query=chunk, limit=1, collection=collection)
            
            if search_results:
                top_result = search_results[0]
                similarity_score = top_result.score
                existing_text = top_result.payload["text"]
                
                if chunk == existing_text:
                    logging.info(f"Exact match found for chunk {idx}, skipping storage")
                    continue
                
                if similarity_score >= similarity_threshold:
                    logging.info(f"Similar memory found (score: {similarity_score}) for chunk {idx}, updating existing")
                    points_to_upsert.append(
                        PointStruct(
                            id=top_result.id,  # Overwrite the existing memory
                            vector=vector,
                            payload={"text": chunk}
                        )
                    )
                    continue
            
            logging.info(f"No match found for chunk {idx}, adding as new memory")
            points_to_upsert.append(
                PointStruct(
                    id=idx + self._get_next_id(collection),
                    vector=vector,
                    payload={"text": chunk}
                )
            )
        
        if points_to_upsert:
            logging.info(f"Upserting {len(points_to_upsert)} points into collection {collection}")
            self.client.upsert(
                collection_name=collection,
                points=points_to_upsert
            )
        else:
            logging.info("No new or updated points to upsert")
    
    def _get_next_id(self, collection: str) -> int:
        """Get the next available ID by checking the current number of points in the collection."""
        try:
            count = self.client.count(collection_name=collection).count
            return count
        except Exception as e:
            logging.error(f"Failed to get count for collection {collection}: {str(e)}")
            return 0
    def search_memories(self, query: str, limit: int = 3, collection: str = "conversations"):
        logging.info(f"Searching memories in collection {collection} with query: {query}")
        vector = GeminiEmbedder.embed(query)
        results = self.client.search(
            collection_name=collection,
            query_vector=vector,
            limit=limit
        )
        logging.info(f"Search returned {len(results)} results")
        return results
    def retrieve_memory(self, query, similarity_threshold=0.8, limit=5, collection=None):
        """
        Retrieve memories (chunks) from the specified collection based on a query and similarity threshold.
        """
        collections=[]
        if collection:
            collections=[collection]

        if not collection:
            collections = ["conversations", "book_chunks"]
        matching_chunks = []
        for collection in collections:
            logging.info(f"Retrieving memories from collection {collection} with query: {query}")
            logging.info(f"Using similarity threshold: {similarity_threshold}, limit: {limit}")
            
            # Generate embedding for the query
            vector = GeminiEmbedder.embed(text=query)
            
            # Search the collection
            search_results = self.client.search(
                collection_name=collection,
                query_vector=vector,
                limit=limit
            )
            
            # Filter results based on similarity threshold
            
            for result in search_results:
                if result.score >= similarity_threshold:
                    chunk_info = {
                        "text": result.payload["text"],
                        "similarity_score": result.score,
                        "id": result.id
                    }
                    matching_chunks.append(chunk_info)
                    logging.info(f"Found matching chunk (score: {result.score}): {result.payload['text'][:50]}...")
        
        logging.info(f"Retrieved {len(matching_chunks)} chunks meeting similarity threshold")
        
        # Sort results by similarity score in descending order
        matching_chunks.sort(key=lambda x: x["similarity_score"], reverse=True)
        search_result=[]
        for chunk in matching_chunks:
            search_result.append(chunk["text"])
        
        return search_result


# Tester
if __name__ == "__main__":
    qm = QdrantManager()
    chunks = ["Abebe likes apples."]
    qm.store_chunks(chunks, collection="conversations")
    
    chunks = ["Abebe does not like apples."]
    qm.store_chunks(chunks, collection="conversations")
    
    results = qm.search_memories("Abebe apples", collection="conversations")
    for result in results:
        print(f"ID: {result.id}, Score: {result.score}, Text: {result.payload['text']}")