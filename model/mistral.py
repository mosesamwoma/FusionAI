import requests
import aiohttp
from config.settings import MISTRAL_API_KEY

URL = "https://api.mistral.ai/v1/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {MISTRAL_API_KEY}",
    "Content-Type": "application/json",
}


def generate(model_name, conversation):
    try:
        response = requests.post(
            URL,
            headers=HEADERS,
            json={"model": model_name,
                  "messages": conversation, "max_tokens": 800},
            timeout=10,
        )
        data = response.json()
        if "error" in data:
            return f"Mistral Error: {data['error']}"
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Mistral Error: {str(e)}"


async def async_generate(session, model_name, conversation):
    try:
        async with session.post(
            URL,
            headers=HEADERS,
            json={"model": model_name,
                  "messages": conversation, "max_tokens": 800},
            timeout=aiohttp.ClientTimeout(total=10),
        ) as response:
            data = await response.json()
            if "error" in data:
                return None
            return data["choices"][0]["message"]["content"][:600]
    except Exception:
        return None
