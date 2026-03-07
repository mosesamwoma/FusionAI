import requests
import aiohttp
from config.settings import GEMINI_API_KEY

URL = "https://generativelanguage.googleapis.com/v1beta/models"


def generate(model_name, conversation, image_data=None, image_mime=None):
    try:
        messages = []
        for m in conversation:
            parts = [{"text": m["content"]}]
            if image_data and m == conversation[-1]:
                parts.insert(0, {
                    "inline_data": {
                        "mime_type": image_mime or "image/jpeg",
                        "data": image_data
                    }
                })
            messages.append({"role": m["role"], "parts": parts})

        response = requests.post(
            f"{URL}/{model_name}:generateContent",
            headers={"Content-Type": "application/json"},
            params={"key": GEMINI_API_KEY},
            json={"contents": messages, "generationConfig": {
                "maxOutputTokens": 8000}},
            timeout=30,
        )
        data = response.json()
        if "error" in data:
            return f"Gemini Error: {data['error'].get('message', 'Unknown')}"
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        return f"Gemini Error: {str(e)}"


async def async_generate(session, model_name, conversation, image_data=None, image_mime=None):
    try:
        messages = []
        for m in conversation:
            parts = [{"text": m["content"]}]
            if image_data and m == conversation[-1]:
                parts.insert(0, {
                    "inline_data": {
                        "mime_type": image_mime or "image/jpeg",
                        "data": image_data
                    }
                })
            messages.append({"role": m["role"], "parts": parts})

        async with session.post(
            f"{URL}/{model_name}:generateContent",
            params={"key": GEMINI_API_KEY},
            json={"contents": messages, "generationConfig": {
                "maxOutputTokens": 8000}},
            timeout=aiohttp.ClientTimeout(total=30),
        ) as response:
            data = await response.json()
            if "error" in data:
                return None
            return data["candidates"][0]["content"]["parts"][0]["text"][:10000]
    except Exception:
        return None
