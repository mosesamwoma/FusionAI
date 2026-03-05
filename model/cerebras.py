import requests
from config.settings import CEREBRAS_API_KEY


def generate(model_name, conversation):
    try:
        response = requests.post(
            "https://api.cerebras.ai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {CEREBRAS_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": model_name,
                "messages": conversation,
                "temperature": 0.7,
                "max_tokens": 300,
            },
            timeout=10,
        )
        data = response.json()
        if "error" in data:
            return f"Cerebras Error: {data['error']}"
        if "choices" in data:
            return data["choices"][0]["message"]["content"]
        return f"Cerebras Error: {data}"
    except requests.exceptions.Timeout:
        return "Cerebras Error: Request timed out"
    except Exception as e:
        return f"Cerebras Error: {str(e)}"
