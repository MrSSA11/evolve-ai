from google import genai
from backend.config import GOOGLE_API_KEY

client = genai.Client(api_key=GOOGLE_API_KEY)

MODEL = "gemini-2.5-flash"

def ask_llm(prompt):
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
    )
    return response.text

def get_active_model():
    return MODEL
