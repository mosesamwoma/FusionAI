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
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 300,
        },
        timeout=30,
    )

    try:
        data = response.json()
    except Exception:
        return f"Cerebras Error: Invalid JSON response -> {response.text}"

    # If API returned error
    if "error" in data:
        return f"Cerebras Error: {data['error']}"

    # If OpenAI-style response
    if "choices" in data:
        return data["choices"][0]["message"]["content"]

    # If different format, print it for debugging
    return f"Cerebras Unexpected Response: {data}"
