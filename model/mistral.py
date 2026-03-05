import requests
from config.settings import MISTRAL_API_KEY


def generate(model_name, conversation):
    if not MISTRAL_API_KEY:
        return "Mistral API key not configured."

    try:
        response = requests.post(
            "https://api.mistral.ai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {MISTRAL_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": model_name,
                "messages": conversation,
                "temperature": 0.7,
            },
            timeout=30,
        )

        response.raise_for_status()
        data = response.json()

        if "error" in data:
            return f"Mistral Error: {data['error']}"

        return data["choices"][0]["message"]["content"]

    except requests.exceptions.RequestException as e:
        return f"Mistral Request Error: {str(e)}"
