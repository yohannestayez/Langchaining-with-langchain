
import sys
sys.path.append('.')
import google.generativeai as genai
import logging
from typing import List
from langchain.memory import ChatMessageHistory
from services.qdrant import QdrantManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MemoryManager:
    """Manages short-term and long-term memory for conversation history."""
    
    def __init__(self, max_summary_length: int = 500, model_name: str = 'gemini-2.0-flash'):
        """
        Initialize MemoryManager with configurable parameters.
        """
        self.memory = ChatMessageHistory()
        self.long_term = QdrantManager()
        self.max_summary_length = max_summary_length
        self.model = genai.GenerativeModel(model_name)
        logging.info("MemoryManager initialized with max_summary_length=%d, model=%s", 
                    max_summary_length, model_name)
    
    def add_message(self, user_message: str, bot_response: str) -> None:
        """Add a user message and bot response to short-term memory."""
        try:
            logging.info("Storing conversation messages")
            self.memory.add_user_message(user_message.strip())
            self.memory.add_ai_message(bot_response.strip())
        except Exception as e:
            logging.error("Failed to add messages to memory: %s", str(e))
            raise
    
    def archive_conversation(self) -> bool:
        """
        Archive the current conversation to long-term storage by summarizing core content
        with a single LLM prompt, then clear short-term memory.
        """
        try:
            if not self.memory.messages:
                logging.info("No messages to archive")
                return False
            
            logging.info("Archiving conversation")
            summary = self._extract_and_summarize_core_content(self.memory.messages)
            summary={"text":summary}
            logging.info(f"Storing summarized conversation in long-term memory: {summary}")
            self.long_term.store_chunks([summary], collection="conversations")

            # Cleaning up short term memory
            if len(self.memory.messages) > 10:
                logging.info("Keeping only the last 5 interactions in short-term memory")
                self.memory.messages=self.memory.messages[-10:]

                
            logging.info("Successfully archived and cleared conversation history")
            return True
        except Exception as e:
            logging.error("Failed to archive conversation: %s", str(e))
            raise
    
    def _extract_and_summarize_core_content(self, messages: List) -> str:
        """
        Use a single LLM prompt to extract and summarize the core parts of the conversation..
        """
        try:
            logging.info("Extracting and summarizing core conversation content")
            # Format messages as a conversation log
            conversation_log = []
            for msg in messages:
                prefix = "User: " if msg.type == "human" else "AI: "
                conversation_log.append(f"{prefix}{msg.content}")
            
            # Generate summary Prompt
            prompt = (
                "Given the following conversation, extract and summarize only the core factual content, "
                "ignoring greetings, farewells, trivial chatter, and redundant information. "
                "Focus on key discussion points, facts, decisions, and conclusions. "
                "Provide a concise summary in plain text.\n\n"
                "Conversation:\n"
                f"{chr(10).join(conversation_log)}\n\n"
                "Summary:"
            )
          
            # Generate summary
            response = self.model.generate_content(prompt)
            summary = response.text.strip()
            # Truncate if necessary
            if len(summary) > self.max_summary_length:
                summary = summary[:self.max_summary_length].rsplit(' ', 1)[0] + "..."

            
            return summary
        
        except Exception as e:
            logging.error("Failed to summarize conversation: %s", str(e))
            return f"Error summarizing conversation: {str(e)}"
    def memory_execute(self, user_message: str, bot_response: str) -> None:
        """Add a user message and bot response to short-term memory."""
        try:
            logging.info("Storing conversation messages")
            self.add_message(user_message.strip(), bot_response.strip())
            self.archive_conversation()
        except Exception as e:
            logging.error("Failed to add messages to memory: %s", str(e))
            raise

# Simple tester
if __name__ == "__main__":
    """Test the functionalities of MemoryManager."""
    # Initialize MemoryManager
    mm = MemoryManager(max_summary_length=200)
    
    mm.add_message("Hello, how are you?", "Hi! I'm doing well, thanks for asking. How about you?")
    mm.add_message("What's the capital of France?", "The capital of France is Paris, known for its rich history and culture.")
    # Test archiving
    archived = mm.archive_conversation()
    print(f"Conversation archived: {archived}")