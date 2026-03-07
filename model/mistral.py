import requests
import aiohttp
from config.settings import MISTRAL_API_KEY

URL = "https://api.mistral.ai/v1/chat/completions"


def get_headers():
    return {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json",
    }


def generate(model_name, conversation, image_data=None, image_mime=None):
    try:
        messages = []
        for m in conversation:
            if image_data and m == conversation[-1] and model_name == "mistral-small-latest":
                content = [
                    {"type": "text", "text": m["content"]},
                    {"type": "image_url",
                        "image_url": f"data:{image_mime or 'image/jpeg'};base64,{image_data}"}
                ]
            else:
                content = m["content"]
            messages.append({"role": m["role"], "content": content})

        response = requests.post(
            URL,
            headers=get_headers(),
            json={"model": model_name, "messages": messages, "max_tokens": 8000},
            timeout=30,
        )
        data = response.json()
        if "error" in data:
            return f"Mistral Error: {data['error']}"
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Mistral Error: {str(e)}"


async def async_generate(session, model_name, conversation, image_data=None, image_mime=None):
    try:
        messages = []
        for m in conversation:
            if image_data and m == conversation[-1] and model_name == "mistral-small-latest":
                content = [
                    {"type": "text", "text": m["content"]},
                    {"type": "image_url",
                        "image_url": f"data:{image_mime or 'image/jpeg'};base64,{image_data}"}
                ]
            else:
                content = m["content"]
            messages.append({"role": m["role"], "content": content})

        async with session.post(
            URL,
            headers=get_headers(),
            json={"model": model_name, "messages": messages, "max_tokens": 8000},
            timeout=aiohttp.ClientTimeout(total=30),
        ) as response:
            data = await response.json()
            if "error" in data:
                return None
            return data["choices"][0]["message"]["content"][:10000]
    except Exception:
        return None
