import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    QDRANT_URL = os.getenv("QDRANT_URL")

    @classmethod
    def validate(cls):
        required = ["GEMINI_API_KEY", "QDRANT_URL"]
        for var in required:
            if not getattr(cls, var):
                raise ValueError(f"Missing required environment variable: {var}")

Config.validate()