import requests
import aiohttp
from config.settings import COHERE_API_KEY

URL = "https://api.cohere.com/v2/chat"


def get_headers():
    return {
        "Authorization": f"Bearer {COHERE_API_KEY}",
        "Content-Type": "application/json",
    }


def generate(model_name, conversation, image_data=None, image_mime=None):
    try:
        messages = [{"role": m["role"], "content": m["content"]}
                    for m in conversation]

        response = requests.post(
            URL,
            headers=get_headers(),
            json={
                "model": model_name,
                "messages": messages,
                "max_tokens": 8000,
            },
            timeout=30,
        )
        data = response.json()

        if "error" in data:
            return f"Cohere Error: {data['error']}"

        message = data.get("message", {})
        content = message.get("content", [])
        if content and isinstance(content, list):
            return content[0].get("text", "")

        if "text" in data:
            return data["text"]

        return f"Cohere Error: Unexpected response format: {data}"

    except Exception as e:
        return f"Cohere Error: {str(e)}"


async def async_generate(session, model_name, conversation, image_data=None, image_mime=None):
    try:
        messages = [{"role": m["role"], "content": m["content"]}
                    for m in conversation]

        async with session.post(
            URL,
            headers=get_headers(),
            json={
                "model": model_name,
                "messages": messages,
                "max_tokens": 8000,
            },
            timeout=aiohttp.ClientTimeout(total=30),
        ) as response:
            data = await response.json(content_type=None)

            if "error" in data:
                print(f"Cohere API error: {data['error']}")
                return None

            message = data.get("message", {})
            content = message.get("content", [])
            if content and isinstance(content, list):
                return content[0].get("text", "")[:10000]

            if "text" in data:
                return data["text"][:10000]

            print(f"Cohere unknown format: {data}")
            return None

    except Exception as e:
        print(f"Cohere async error: {e}")
        return None
