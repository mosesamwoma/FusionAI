import requests
import aiohttp
from config.settings import CEREBRAS_API_KEY

URL = "https://api.cerebras.ai/v1/chat/completions"


def get_headers():
    return {
        "Authorization": f"Bearer {CEREBRAS_API_KEY}",
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
        data = response.json()
        if "error" in data:
            return f"Cerebras Error: {data['error']}"
        # Handle both response formats
        if "choices" in data and data["choices"]:
            return data["choices"][0]["message"]["content"]
        if "message" in data:
            return data["message"].get("content", "")
        return None
    except Exception as e:
        return f"Cerebras Error: {str(e)}"


async def async_generate(session, model_name, conversation, image_data=None, image_mime=None):
    try:
        async with session.post(
            URL,
            headers=get_headers(),
            json={"model": model_name,
                  "messages": conversation, "max_tokens": 1000},
            timeout=aiohttp.ClientTimeout(total=30),
        ) as response:
            data = await response.json()
            if "error" in data:
                return None
            # Handle both response formats
            if "choices" in data and data["choices"]:
                return data["choices"][0]["message"]["content"][:10000]
            if "message" in data:
                return data["message"].get("content", "")[:10000]
            return None
    except Exception:
        return None
