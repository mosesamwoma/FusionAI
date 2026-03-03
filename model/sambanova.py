import requests
from config.settings import SAMBANOVA_API_KEY


def generate(model_name, prompt):
    response = requests.post(
        "https://api.sambanova.ai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {SAMBANOVA_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": model_name,
            "messages": [
                {"role": "user", "content": prompt}
            ],
        },
    )

    data = response.json()

    if "error" in data:
        return f"SambaNova Error: {data['error']}"

    return data["choices"][0]["message"]["content"]
