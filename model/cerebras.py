import requests
from config.settings import CEREBRAS_API_KEY


def generate(model_name, prompt):
    response = requests.post(
        "https://api.cerebras.ai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {CEREBRAS_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 300,
        },
        timeout=30,
    )
    try:
        data = response.json()
    except Exception:
        return f"Cerebras Error: Invalid JSON -> {response.text}"
    if "error" in data:
        return f"Cerebras Error: {data['error']}"
    if "choices" in data:
        return data["choices"][0]["message"]["content"]
    return f"Cerebras Unexpected Response: {data}"
