import json
import logging
from pydantic import BaseModel
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class CharacterSchema(BaseModel):
    name: str
    traits: dict[str, float] # "arousal" and "valence" values
    summary: str

class CharacterExtractor:
    def extract(self, text: str) -> list[CharacterSchema]:
        if not isinstance(text, str) or not text.strip():
            logging.error("Invalid input: Text must be a non-empty string")
            raise ValueError("Text must be a non-empty string")
        
        logging.info("Extracting characters from text using optimized prompt")
        prompt = f"""
        Extract characters from this text: "{text}"
        
        For each character, return:
        - "name": string, the character's name
        - "traits": object with only these two properties:
          - "arousal": float (0 to 1, emotional intensity)
          - "valence": float (0 to 1, emotional positivity, 0=negative, 1=positive)
        - "summary": string, brief and direct yet detailed character description (max 100 words)
        
        Rules:
        - Base traits on text evidence only
        - If no clear traits, use 0.5 for both arousal and valence
        - Return JSON array only, no extra text
        
        Example output:
        [
            {{"name": "John", "traits": {{"arousal": 0.8, "valence": 0.2}}, "summary": "An angry soldier seeking revenge"}},
            {{"name": "Mary", "traits": {{"arousal": 0.3, "valence": 0.7}}, "summary": "A calm healer helping others"}}
        ]
        """
        model = genai.GenerativeModel('gemini-1.5-pro')
        response = model.generate_content(prompt)
        logging.info(f"...Successfully generated character data in type {type(response.text)}...")

        return  response.text if isinstance(response.text[0], CharacterSchema) else  self._parse_response(response.text)
    
    def _parse_response(self, raw: str) -> list[CharacterSchema]:
        try:
            # Efficiently clean response
            cleaned_raw = raw.strip().removeprefix("```json").removesuffix("```").strip()
            data = json.loads(cleaned_raw)
            
            if not isinstance(data, list):
                raise ValueError("Response must be a JSON array")
            
            # Validate and normalize each character's traits
            characters = []
            for item in data:
                traits = item.get("traits", {"arousal": 0.5, "valence": 0.5})
                # Ensure only arousal and valence are present and within bounds
                normalized_traits = {
                    "arousal": max(0.0, min(1.0, float(traits.get("arousal", 0.5)))),
                    "valence": max(0.0, min(1.0, float(traits.get("valence", 0.5))))
                }
                characters.append(CharacterSchema(
                    name=item["name"],
                    traits=normalized_traits,
                    summary=item.get("summary", "")
                ))
            
            logging.info(f"Successfully parsed {len(characters)} characters")
            return characters
        except json.JSONDecodeError as e:
            logging.error(f"JSON parsing failed: {str(e)} - Raw: {raw}")
            raise ValueError("Failed to parse character data")
        except KeyError as e:
            logging.error(f"Missing required field: {str(e)} - Raw: {raw}")
            raise ValueError("Invalid character data structure")
        except Exception as e:
            logging.error(f"Unexpected error: {str(e)}")
            raise

# Example usage
if __name__ == "__main__":
    extractor = CharacterExtractor()
    text = "John shouted furiously at the enemy, while Mary calmly tended to the wounded."
    characters = extractor.extract(text)
    for char in characters:
        print(char)
