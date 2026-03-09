import requests
import aiohttp
import json
from config.settings import SAMBANOVA_API_KEY

URL = "https://api.sambanova.ai/v1/chat/completions"


def get_headers():
    return {
        "Authorization": f"Bearer {SAMBANOVA_API_KEY}",
        "Content-Type": "application/json",
    }


def generate(model_name, conversation, image_data=None, image_mime=None):
    try:
        response = requests.post(
            URL,
            headers=get_headers(),
            json={"model": model_name,
                  "messages": conversation, "max_tokens": 1000},
            timeout=30,
        )
        if not response.text or not response.text.strip():
            return None
        data = response.json()
        if not isinstance(data, dict):
            return None
        if "error" in data:
            return None
        if "choices" in data and data["choices"]:
            return data["choices"][0]["message"]["content"]
        return None
    except Exception as e:
        return f"SambaNova Error: {str(e)}"


async def async_generate(session, model_name, conversation, image_data=None, image_mime=None):
    try:
        async with session.post(
            URL,
            headers=get_headers(),
            json={"model": model_name,
                  "messages": conversation, "max_tokens": 1000},
            timeout=aiohttp.ClientTimeout(total=30),
        ) as response:
            text = await response.text()
            if not text or not text.strip():
                return None
            try:
                data = json.loads(text)
            except Exception:
                return None
            if not isinstance(data, dict):
                return None
            if "error" in data:
                return None
            if "choices" in data and data["choices"]:
                return data["choices"][0]["message"]["content"][:10000]
            return None
    except Exception:
        return None
