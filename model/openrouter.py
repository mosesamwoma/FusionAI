import requests
import aiohttp
from config.settings import OPENROUTER_API_KEY

URL = "https://openrouter.ai/api/v1/chat/completions"


def get_headers():
    return {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://fusionai.app",
        "X-Title": "FusionAI",
    }


def generate(model_name, conversation, image_data=None, image_mime=None):
    try:
        messages = []
        for m in conversation:
            if image_data and m == conversation[-1] and "vision" in model_name:
                content = [
                    {"type": "text", "text": m["content"]},
                    {"type": "image_url", "image_url": {
                        "url": f"data:{image_mime or 'image/jpeg'};base64,{image_data}"}}
                ]
            else:
                content = m["content"]
            messages.append({"role": m["role"], "content": content})

        response = requests.post(
            URL,
            headers=get_headers(),
            json={"model": model_name, "messages": messages, "max_tokens": 800},
            timeout=15,
        )
        data = response.json()
        if "error" in data:
            return f"OpenRouter Error: {data['error']}"
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"OpenRouter Error: {str(e)}"


async def async_generate(session, model_name, conversation, image_data=None, image_mime=None):
    try:
        messages = []
        for m in conversation:
            if image_data and m == conversation[-1] and "vision" in model_name:
                content = [
                    {"type": "text", "text": m["content"]},
                    {"type": "image_url", "image_url": {
                        "url": f"data:{image_mime or 'image/jpeg'};base64,{image_data}"}}
                ]
            else:
                content = m["content"]
            messages.append({"role": m["role"], "content": content})

        async with session.post(
            URL,
            headers=get_headers(),
            json={"model": model_name, "messages": messages, "max_tokens": 800},
            timeout=aiohttp.ClientTimeout(total=15),
        ) as response:
            data = await response.json()
            if "error" in data:
                return None
            return data["choices"][0]["message"]["content"][:300]
    except Exception:
        return None
