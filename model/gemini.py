import time
from google import genai
from config.settings import GEMINI_API_KEY


def generate(model_name, conversation):
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)

        history = []
        for msg in conversation[:-1]:
            role = "user" if msg["role"] == "user" else "model"
            history.append({"role": role, "parts": [{"text": msg["content"]}]})

        latest = conversation[-1]["content"]

        for attempt in range(3):
            try:
                chat = client.chats.create(model=model_name, history=history)
                response = chat.send_message(latest)
                return response.text
            except Exception as e:
                if "429" in str(e):
                    time.sleep(5)
                    continue
                return f"Gemini Error: {str(e)}"

        return "Gemini Error: Rate limit exceeded, try again later"

    except Exception as e:
        return f"Gemini Error: {str(e)}"
