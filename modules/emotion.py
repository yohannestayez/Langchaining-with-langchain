from sys import path
path.append('.')
import google.generativeai as genai
import json
import logging
from config import Config
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

genai.configure(api_key=Config.GEMINI_API_KEY)

class PsiEmotionEngine:
    def __init__(self, base_params: dict):
        logging.info("Initializing PsiEmotionEngine with base parameters")
        self.arousal = base_params.get("arousal", 0.5)
        self.valence = base_params.get("valence", 0.5)
        self.decay_rate = 0.9 # Rate of decay for arousal and valence
    
    def update(self, text: str):
        if not isinstance(text, str) or not text.strip():
            logging.error("Invalid input: Text must be a non-empty string")
            raise ValueError("Text must be a non-empty string")
        
        logging.info("Analyzing sentiment of input text")
        sentiment = self._analyze_sentiment(text)
        
        # Significantly increased impact for noticeable changes
        self.arousal += sentiment["intensity"] * 0.4  
        self.valence += sentiment["polarity"] * 0.5
        
        self._apply_bounds()

    def _clean_json_response(self, text: str) -> str:
        # Remove markdown code blocks
        text = re.sub(r'^```json\s*', '', text, flags=re.MULTILINE)
        text = re.sub(r'^```\s*', '', text, flags=re.MULTILINE)
        text = text.strip()
        
        # Replace single quotes with double quotes
        text = text.replace("'", '"')
        
        # Remove trailing commas
        text = re.sub(r',\s*}(?=\s*$)', '}', text)
        text = re.sub(r',\s*](?=\s*$)', ']', text)
        
        return text
    
    def _analyze_sentiment(self, text: str) -> dict:
        logging.info("Generating sentiment analysis using Gemini model")
        prompt = f"""
        Analyze the sentiment of this text: "{text}"

        Provide a JSON object with:
        - "polarity": float from -1 (very negative) to 1 (very positive)
        - "intensity": float from 0 (neutral) to 1 (extreme emotion)

        Consider:
        - Emotional keywords and their strength
        - Context and connotations
        - Punctuation and emphasis (e.g., "!" increases intensity)

        Examples:
        "I love this!" → {{"polarity": 0.9, "intensity": 0.8}}
        "This is awful" → {{"polarity": -0.7, "intensity": 0.6}}

        Return only the JSON object, no additional text:
        """

        
        try:
            model = genai.GenerativeModel('gemini-2.0-flash')
            response = model.generate_content(prompt)
            response= self._clean_json_response(response.text)
            sentiment_data = json.loads(response.strip())
            polarity = float(sentiment_data.get("polarity", 0.0))
            intensity = float(sentiment_data.get("intensity", 0.5))
            return {
                "polarity": max(-1, min(1, polarity)),
                "intensity": max(0, min(1, intensity))
            }
        except Exception as e:
            logging.error(f"Sentiment analysis failed: {str(e)}. Using fallback values")
            return {"polarity": 0.0, "intensity": 0.5}
    
    def _apply_bounds(self):
        self.arousal = max(0.0, min(1.0, self.arousal)) * self.decay_rate
        self.valence = max(0.0, min(1.0, self.valence)) * self.decay_rate
    
    @property
    def state(self) -> dict:
        current_state = {
            "arousal": round(self.arousal, 3),
            "valence": round(self.valence, 3),
            "emotion": self._current_emotion_label()
        }
        logging.info(f"Current emotional state: {current_state}")
        return current_state
    
    def _current_emotion_label(self) -> str:
        # Adjusted thresholds for more distinct emotional shifts
        if self.valence < 0.35:
            if self.arousal > 0.75: 
                return "rage" 
            elif self.arousal < 0.25:  
                return "despair"
            return "irritation"
        elif self.valence > 0.65: 
            if self.arousal > 0.75:  
                return "ecstasy"  
            elif self.arousal < 0.25:  
                return "peace"
            return "thrill"
        else:
            if self.arousal > 0.7:
                return "panic"
            elif self.arousal < 0.3:
                return "boredom"
            return "neutral"



if __name__ == "__main__":
    engine = PsiEmotionEngine({"arousal": 0.5, "valence": 0.5})

    # Initial state
    print(engine.state)  
    #Strong negative input
    engine.update(text="I HATE THIS SO MUCH!!!")
    print(engine.state)  # Irritation

    # Strong positive input
    engine.update(text= "This is the BEST day EVER!!!!")
    print(engine.state) # Panic