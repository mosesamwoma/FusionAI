import requests
import aiohttp
from config.settings import SAMBANOVA_API_KEY

URL = "https://api.sambanova.ai/v1/chat/completions"


def get_headers():
    return {
        "Authorization": f"Bearer {SAMBANOVA_API_KEY}",
        "Content-Type": "application/json",
    }


def generate(model_name, conversation):
    try:
        response = requests.post(
            URL,
            headers=get_headers(),
            json={"model": model_name,
                  "messages": conversation, "max_tokens": 8000},
            timeout=30,
        )
        data = response.json()
        if "error" in data:
            return f"SambaNova Error: {data['error']}"
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"SambaNova Error: {str(e)}"


async def async_generate(session, model_name, conversation, image_data=None, image_mime=None):
    try:
        async with session.post(
            URL,
            headers=get_headers(),
            json={"model": model_name,
                  "messages": conversation, "max_tokens": 8000},
            timeout=aiohttp.ClientTimeout(total=30),
        ) as response:
            data = await response.json()
            if "error" in data:
                return None
            return data["choices"][0]["message"]["content"][:10000]
    except Exception:
        return None
