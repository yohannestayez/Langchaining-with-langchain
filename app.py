from flask import Flask, request, jsonify
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
import pydantic

app = Flask(__name__)

# Configure Gemini API
genai.configure(api_key=Config.GEMINI_API_KEY)

# Initialize components
book_processor = BookProcessor()
character_extractor = CharacterExtractor()
qdrant = QdrantManager()
memory = MemoryManager()

# Store characters globally (for simplicity; in production, use a database)
characters = []

def extract_pdf_text(pdf_file):
    """
    Extracts text from a given PDF file object using PyPDF2.

    :param pdf_file: File object of the uploaded PDF.
    :return: Extracted text as a string.
    """
    text = ""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
    except Exception as e:
        print(f"Error extracting PDF text: {e}")
    return text

@app.route('/chat', methods=['POST'])
def chat():
    
    """Handle PDF upload or chat message, with character extraction and automatic switching."""

    global characters
    
    message = request.form.get('message')  # Get message from form data
    pdf_file = request.files.get('pdf_file')  # Get PDF file from form data

    if not message and not pdf_file:
        return jsonify({"error": "No message or PDF file provided"}), 400

    # Step 1: Handle PDF upload if provided
    if pdf_file:
        if not pdf_file.filename.endswith('.pdf'):
            return jsonify({"error": "File must be a PDF"}), 400
        
        # Extract text from the PDF
        book_text = extract_pdf_text(pdf_file)
        if not book_text.strip():
            return jsonify({"error": "No text extracted from PDF"}), 400
        
        # Process book into chunks and store in Qdrant
        chunks = book_processor.process_book(book_text)
        qdrant.store_chunks(chunks)
        
        # Extract characters and store globally
        

        print("Extracting characters from PDF...")
        print(character_extractor.extract(book_text)[0])
        print(type(character_extractor.extract(book_text)[0]))
        characters = [char.model_dump() for char in character_extractor.extract(book_text)]
        qdrant.store_chunks([{"text": f"{char}"} for char in characters], collection="characters")

        # Now characterss contains a list of dictionaries

        if not characters:
                return jsonify({"error": "No characters extracted from PDF"}), 400
        
        # Return initial response with extracted characters
        return jsonify({
            "response": "PDF uploaded and characters extracted.",
            "characters": characters
        })

    # Step 2: Handle chat if no PDF file (assumes PDF already processed)
    if not message:
        return jsonify({"error": "No message provided"}), 400

    if not characters:
        # Default response if no PDF has been processed yet
        try:
           
            print("Retrieving characters from memory....")
            characters = qdrant.retrieve_memory(query=message , similarity_threshold = 0.6 ,collection="characters")
            print(characters[0])
            print(f"Retrieved characters from memory: {characters}")
            characters = [json.loads(character_memory.replace("'", "\""))for character_memory in characters]
            
            
        except:
            print("No characters found in memory. Using default character.")
            prompt = f"You are a helpful assistant. User: {message}"
            model = genai.GenerativeModel('gemini-2.0-flash')
            response = model.generate_content(prompt)
            return jsonify({"response": response.text})

    # Step 3: Detect character in message and switch automatically
    current_character = None
    def match_character(message: str, characters: list) -> dict:
        """
        Uses prompting to match message context to the most appropriate character
        from a list of character dictionaries with 'name' and 'description' fields
        """
        character_list = "\n".join([f"- {c['name']}: {c['summary']}" for c in characters])

        prompt = f"""Analyze this message and identify which character it's addressing:
        
        Message: "{message}"
        
        Available characters:
        {character_list}
        
        Rules:
        1. Match based on name mentions OR contextual relevance
        2. Consider nicknames, titles, and role references
        3. If multiple matches, choose the most specific
        4. If no match, return null
        
        Respond ONLY with JSON format: 
        {{ "match": "character_name" | null, "confidence": 0.0-1.0 }}
        """
        
        try:
            model = genai.GenerativeModel('gemini-2.0-flash')
            response = model.generate_content(prompt)
            
            # Clean response and parse JSON
            response_text = re.sub(r'[^{}]+', '', response.text)
            match_data = json.loads(response_text)
            
            if match_data.get('match'):
                return next((c for c in characters if c['name'].lower() == match_data['match'].lower()), None)
                
        except Exception as e:
            print(f"Auto matching failed: {e}. Using fallback...")
        
        # Fallback to simple name match
        for char in characters:
            if char["name"].lower() in message.lower():
                return char
        return None
    current_character = match_character(message, characters)
    
    if not current_character:
        character_list = "\n".join([f"- {c['name']}: {c['summary']}" for c in characters])
        # Default to first character if none mentioned
        prompt = f"""Identify the character I am interacting with in the last parts of the following conversation:  
                    {memory.memory.messages}  

                    If a name is mentioned toward the end, it is more likely that the conversation is with that character, 
                    but consider the full context before making a decision.

                    Select the character from the list below:  
                    {character_list}  

                    Respond only with the character's name as a string if found, otherwise respond with this exact string: 'null'. 
                    Don't add any additional text.
                    """
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(prompt)
        if response.text == "null":
            current_character = characters[0]
        else:
            current_character = match_character(response.text, characters)

    # Initialize emotion and memory for this interaction
    emotion_engine = PsiEmotionEngine(current_character["traits"])

    # Update emotional state based on message
    emotion_engine.update(message)
    # Retrieve relevant knowledge or long term moemory from qdrant memory memory
    knowledge = qdrant.retrieve_memory(query = message)
    # Generate response
    prompt = f"""
    You are {current_character['name']}. 
    Personality: {current_character['summary']}
    Current emotion: {emotion_engine.state['emotion']}
    History and knowledge around the subject with there relevenance in order(from relevant to less relevant): {knowledge}
    Conversation history:
    {memory.memory.messages}
    
    User: {message}
    {current_character['name']}:
    """
    model = genai.GenerativeModel('gemini-2.0-flash')
    response = model.generate_content(prompt)

    # Store interaction in memory
    memory.memory_execute(user_message = message, responder= current_character["name"], bot_response = response.text)

    return jsonify({
        "response": response.text,
        "character": current_character["name"],
        "emotion": emotion_engine.state
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)