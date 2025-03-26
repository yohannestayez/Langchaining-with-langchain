import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class BookProcessor:
    def __init__(self):
        logging.info("Initializing BookProcessor with RecursiveCharacterTextSplitter")
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\nChapter ", "\n\nSection ", "\n\n", "\n", ".", " "]
        )
    
    def process_book(self, text: str) -> list[dict]:
        logging.info("Processing book text...")
        
        if not isinstance(text, str) or not text.strip():
            logging.error("Invalid input: Book text must be a non-empty string")
            raise ValueError("Book text must be a non-empty string")
        
        logging.info("Splitting text into chunks")
        chunks = self.splitter.split_text(text)
        
        logging.info(f"Successfully split text into {len(chunks)} chunks")
        print(chunks)
        return [{"text": chunk, "metadata": {}} for chunk in chunks]  # Metadata placeholder
