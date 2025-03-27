from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from modules.book_processor import BookProcessor
from modules.character import CharacterExtractor
from modules.emotion import PsiEmotionEngine
from modules.memory import MemoryManager
from services.qdrant import QdrantManager
from config import Config
import google.generativeai as genai
import PyPDF2
import re
import json
import ast

app = Flask(__name__)
CORS(app)
characters = []  # Preserve global state

# Initialization Functions
def initialize_components():
    genai.configure(api_key=Config.GEMINI_API_KEY)
    return {
        'book_processor': BookProcessor(),
        'character_extractor': CharacterExtractor(),
        'qdrant': QdrantManager(),
        'memory': MemoryManager()
    }

components = initialize_components()

# PDF Handling Functions
def handle_pdf_upload(pdf_file):
    book_text = extract_pdf_text(pdf_file)
    if not book_text.strip():
        return None
    
    components['qdrant'].store_chunks(components['book_processor'].process_book(book_text))
    extracted_chars = components['character_extractor'].extract(book_text)
    processed_chars = [char.model_dump() for char in extracted_chars]
    
    components['qdrant'].store_chunks(
        [{"text": f"{char}"} for char in processed_chars],
        collection="characters"
    )
    return processed_chars

def extract_pdf_text(pdf_file):
    text = ""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
    except Exception as e:
        print(f"Error extracting PDF text: {e}")
    return text

def parser(text: str):
        cleaned_raw = text.strip().removeprefix("```json").removesuffix("```").strip()
        data = json.loads(cleaned_raw)
        return data
            
# Character Matching Functions
def match_character(message, characters):
    character_list = "\n".join([f"- {c['name']}: {c['summary']}" for c in characters])
    prompt = f"""Analyze this message and identify which character it's addressing:
    Message: "{message}"
    Available characters:
    {character_list}
    Rules:
    1. Match based on name mentions OR contextual relevance
    2. Consider nicknames, titles, and role references
    3. If multiple matches, choose the most specific or relevant character
    Respond ONLY with JSON format: 
    {{ "match": "character_name", "confidence": 0.0-1.0 }}
    """
    
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(prompt)
        print(f"Auto matching prompt: {response.text}")

        match_data = parser(response.text)
        
        
        if match_data.get('match'):
            character = next((c for c in characters if c['name'].lower() == match_data['match'].lower()), None)
            confidence = match_data.get('confidence', 0.0)
            return character, confidence
    except Exception as e:
        print(f"Auto matching failed: {e}")
    
    # Fallback to simple name matching
    for char in characters:
        if char["name"].lower() in message.lower():
            return char, 1.0  # High confidence for direct name mention
    return None, 0.0

# Chat Processing Functions
def handle_character_retrieval(message):
    global characters
    try:
        print("Retrieving characters from memory....")
        character_memories = components['qdrant'].retrieve_memory(
            query=message, 
            similarity_threshold=0.7,
            collection="characters"
        )
        try:
            characters = [ast.literal_eval(cm) for cm in character_memories]
            print(f"Retrieved characters from memory: {characters}")
        except (ValueError, SyntaxError) as e:
            print(f"Error parsing characters: {e}")
        return True
    except Exception as e:
        print(f"No characters found in memory: {e}")
        return False

def generate_fallback_response(message):
    prompt = f"You are a helpful assistant. User: {message}"
    model = genai.GenerativeModel('gemini-2.0-flash')
    response = model.generate_content(prompt)
    return jsonify({"response": response.text})

def create_conversation_prompt(current_character, emotion_engine, knowledge, message):
    return f"""
    You are {current_character['name']}. 
    Personality: {current_character['summary']}
    Current emotion: {emotion_engine.state['emotion']}
    History and knowledge: {knowledge}
    Conversation history:
    {components['memory'].memory.messages}
    
    User: {message}
    {current_character['name']}:
    """

def handle_chat_interaction(message):
    global characters
    
    # Try to match character from message
    current_character, confidence = match_character(message, characters)
    if current_character and confidence >= 0.3:
        pass  # Use this character
    else:
        # Try to infer from conversation history
        character_list = "\n".join([f"- {c['name']}: {c['summary']}" for c in characters])
        prompt = f"""
            Identify which character, if any, is being addressed in the conversation history.

            Conversation history:
            {components['memory'].memory.messages}

            Available characters:
            {character_list}

            Instructions:
            - Analyze the entire conversation history to determine the character being addressed.
            - Look for direct name mentions, contextual clues, or references to traits or events associated with the characters based on their summaries.
            - If multiple characters could be matches, select the one that is most directly addressed or most relevant to the context.
            - If no character is being addressed, respond with {{ "match": null, "confidence": 0.0 }}.
            - Use the character summaries to inform your decision when the conversation lacks explicit names.

            Response format:
            Respond with a JSON object containing the matched character's name (or null) and a confidence score between 0.0 and 1.0.

            Example:
            Conversation history: "Hey Tessie, how are you?"
            Available characters:
            - Tessie Hutchinson: The lottery's winner.
            - Bill Hutchinson: Tessie's husband.
            Response: {{ "match": "Tessie Hutchinson", "confidence": 1.0 }}

            Respond only with the JSON object in your final output.
        """
        try:
            response = genai.GenerativeModel('gemini-2.0-flash').generate_content(prompt)
            print(f"Character inference prompt: {response.text}")
            match_data = parser(response.text)
            
            if match_data.get('match'):
                current_character = next((c for c in characters if c['name'].lower() == match_data['match'].lower()), None)
                confidence = match_data.get('confidence', 0.0)
                if current_character and confidence >= 0.3:
                    pass  # Use this character
                else:
                    current_character = None
            else:
                current_character = None
        except Exception as e:
            print(f"Error inferring character from history: {e}")
            current_character = None
    
    if not current_character:
        # Fallback to general AI assistant
        return generate_fallback_response(message)
    
    # Proceed with character-based response
    emotion_engine = PsiEmotionEngine(current_character["traits"])
    emotion_engine.update(message)
    knowledge = components['qdrant'].retrieve_memory(query=message)
    prompt = create_conversation_prompt(current_character, emotion_engine, knowledge, message)
    response = genai.GenerativeModel('gemini-2.0-flash').generate_content(prompt)
    components['memory'].memory_execute(
        user_message=message,
        responder=current_character["name"],
        bot_response=response.text
    )
    return jsonify({
        "response": response.text,
        "character": current_character["name"],
        "emotion": emotion_engine.state
    })

@app.route('/')
def index():
    return render_template('index.html')
# Main Route
@app.route('/chat', methods=['POST'])
def chat():
    global characters
    
    message = request.form.get('message')
    pdf_file = request.files.get('pdf_file')

    if not message and not pdf_file:
        return jsonify({"error": "No message or PDF file provided"}), 400

    if pdf_file:
        if not pdf_file.filename.endswith('.pdf'):
            return jsonify({"error": "File must be a PDF"}), 400
        
        characters = handle_pdf_upload(pdf_file)
        if not characters:
            return jsonify({"error": "No text extracted from PDF"}), 400
        
        return jsonify({
            "response": "PDF uploaded and characters extracted.",
            "characters": characters
        })

    # If no characters are loaded, attempt to retrieve from memory
    if not characters:
        handle_character_retrieval(message)

    
    # Handle the chat interaction, which will fall back to general AI if necessary
    return handle_chat_interaction(message)

if __name__ == '__main__':
    app.run(debug=True, port=5000)