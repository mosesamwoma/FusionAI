from google import genai
from config.settings import GEMINI_API_KEY


def generate(model_name, prompt):
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
        )
        return response.text
    except Exception as e:
        return f"Gemini Error: {str(e)}"
