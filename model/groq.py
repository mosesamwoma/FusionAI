import requests
from config.settings import GROQ_API_KEY


def generate(model_name, conversation):
    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": model_name,
            "messages": conversation,
        },
    )

    data = response.json()

    if "error" in data:
        return f"Groq Error: {data['error']}"

    return data["choices"][0]["message"]["content"]
