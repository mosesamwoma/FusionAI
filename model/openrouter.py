import requests
from config.settings import OPENROUTER_API_KEY


def generate(model_name, conversation):
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": model_name,
                "messages": conversation,
                "max_tokens": 512,
            },
            timeout=30,
        )
        data = response.json()
        if "error" in data:
            return f"OpenRouter Error: {data['error']}"
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"OpenRouter Error: {str(e)}"
