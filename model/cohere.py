import requests
import aiohttp
from config.settings import COHERE_API_KEY

URL = "https://api.cohere.com/v2/chat"
HEADERS = {
    "Authorization": f"Bearer {COHERE_API_KEY}",
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
            return f"Cohere Error: {data['error']}"
        return data["message"]["content"][0]["text"]
    except Exception as e:
        return f"Cohere Error: {str(e)}"


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
            return data["message"]["content"][0]["text"][:600]
    except Exception:
        return None
