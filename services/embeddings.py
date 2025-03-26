import google.generativeai as genai
from config import Config
from tenacity import retry, stop_after_attempt, wait_exponential
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

genai.configure(api_key=Config.GEMINI_API_KEY)

class GeminiEmbedder:
    @staticmethod
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def embed(text: str) -> list[float]:
        if not isinstance(text, str) or not text.strip():
            logging.error("Input text must be a non-empty string")
            raise ValueError("Input text must be a non-empty string")
        
        logging.info("Generating embedding for input text")
        try:
            response = genai.embed_content(
                model="models/embedding-001",
                content=text,
                task_type="retrieval_document"
            )
            logging.info("Successfully generated embedding")
            return response["embedding"]
        except Exception as e:
            logging.error(f"Failed to generate embedding: {str(e)}")
            raise ValueError(f"Failed to generate embedding: {str(e)}")
