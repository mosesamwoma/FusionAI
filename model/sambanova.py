import requests
from config.settings import SAMBANOVA_API_KEY


def generate(model_name, conversation):
    try:
        response = requests.post(
            "https://api.sambanova.ai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {SAMBANOVA_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": model_name,
                "messages": conversation,
            },
            timeout=30,
        )
        data = response.json()
        if "error" in data:
            return f"SambaNova Error: {data['error']}"
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"SambaNova Error: {str(e)}"
