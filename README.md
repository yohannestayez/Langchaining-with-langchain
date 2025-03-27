# Interactive Book charachter Chatbot

## Overview
This project is an AI-powered chatbot that enables users to engage in interactive conversations with characters extracted from books. It leverages advanced vector search to create dynamic, emotionally resonant dialogues. The system integrates **Flask** for the web server, **LangChain** for managing conversation memory, **Qdrant** for vector storage, and the **Google Gemini** for text generation and embeddings.


## Folder Structure

```
│── app.py                  # Main Flask application handling routing and API requests
│── config.py               # Centralized configuration (API keys, database URLs, etc.)
│
├── modules                 # Core processing modules
│   ├── book_processor.py   # Handles PDF parsing and text chunking using LangChain
│   ├── character.py        # Extracts and structures character details via Gemini API
│   ├── emotion.py          # Analyzes sentiment and simulates emotions based on Dorner’s Psi Theory
│   └── memory.py           # Manages conversation history and memory archiving using LangChain and Qdrant
│
└──services                # External service integrations
   ├── embeddings.py       # Generates text embeddings
   └── qdrant.py           # Interfaces with Qdrant for vector storage and similarity search

```

---

## Core Components

### Book Processing & Character Extraction
- **Book Processing:**  
  The system processes PDF files to extract raw text using robust PDF parsing techniques. It then employs a recursive text splitter from LangChain to segment the text into manageable chunks based on chapter, section, and other textual markers. This segmentation ensures efficient downstream processing.

- **Character Extraction:**  
  Analyzed the segmented text using LLM to identify key characters within the book. The extraction process generates structured data for each character, including:
  - **Name**
  - **Emotional Traits:** Quantified as arousal and valence values derived from textual evidence.
  - **Summary:** A concise description capturing the character’s role and behavior.
  
  The extraction process utilizes the Gemini API and leverages Pydantic models for data validation and structuring.

### Emotion Simulation & Sentiment Analysis
- **Emotion Modeling:**  
  The project implements Dorner’s Psi Theory as the foundation for simulating character emotions. This theory informs how emotional states are modeled and updated throughout interactions. Each character's emotional state is dynamically adjusted using a combination of sentiment analysis and a decay mechanism to mimic realistic emotional transitions.

- **Sentiment Analysis:**  
  Incoming text is analyzed to gauge sentiment polarity and intensity. The Gemini API is used to perform this analysis, which then adjusts the character’s emotional parameters. Techniques such as regular expressions are applied to ensure that the responses from the API are properly formatted and reliable.

### Memory Management & Contextual Recall
- **Short-term Memory:** : The system records conversation history using LangChain’s ChatMessageHistory, preserving context across multiple turns of dialogue. This ensures that interactions remain coherent and contextually aware.

- **Long-term Memory:** : Key details from conversations are archived in the Qdrant vector database. The system summarizes and distills the core factual content of dialogues, storing this information as embeddings. This long-term memory facilitates efficient retrieval and context enrichment in future interactions.

### Embedding Generation & Qdrant Integration
- **Embedding Generation:**: Text is converted into high-dimensional vector representations using the Gemini API. These embeddings capture semantic nuances and are crucial for performing effective similarity searches.

- **Qdrant Vector Database:**: The project utilizes Qdrant to store and retrieve these embeddings:
  - **Storage:** Text chunks and conversation summaries are embedded and stored along with relevant metadata.
  - **Retrieval:** When processing user input, the system performs similarity searches against stored vectors to fetch the most contextually relevant content, ensuring that responses are grounded in prior conversation data.

---

### Setup and Run
Install dependencies from `requirements.txt`, set your `Gemini API key` and set your `qdrant url` in a `.env` file , and run `python app.py` to start the server.

### Dependencies
- `Langchain`
- `google-generativeai`
- `qdrant-client`
- `flask`
