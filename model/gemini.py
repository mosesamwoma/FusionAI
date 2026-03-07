import requests
import aiohttp
from config.settings import GEMINI_API_KEY

URL = "https://generativelanguage.googleapis.com/v1beta/models"


def generate(model_name, conversation):
    try:
        messages = [
            {"role": m["role"], "parts": [{"text": m["content"]}]}
            for m in conversation
        ]
        response = requests.post(
            f"{URL}/{model_name}:generateContent",
            headers={"Content-Type": "application/json"},
            params={"key": GEMINI_API_KEY},
            json={"contents": messages},
            timeout=10,
        )
        data = response.json()
        if "error" in data:
            return f"Gemini Error: {data['error'].get('message', 'Unknown error')}"
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        return f"Gemini Error: {str(e)}"


async def async_generate(session, model_name, conversation):
    try:
        messages = [
            {"role": m["role"], "parts": [{"text": m["content"]}]}
            for m in conversation
        ]
        async with session.post(
            f"{URL}/{model_name}:generateContent",
            params={"key": GEMINI_API_KEY},
            json={"contents": messages},
            timeout=aiohttp.ClientTimeout(total=10),
        ) as response:
            data = await response.json()
            if "error" in data:
                return None
            return data["candidates"][0]["content"]["parts"][0]["text"][:600]
    except Exception:
        return None
