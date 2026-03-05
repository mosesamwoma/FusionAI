import requests
from config.settings import NVIDIA_API_KEY


def generate(model_name, conversation):
    try:
        response = requests.post(
            "https://integrate.api.nvidia.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {NVIDIA_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": model_name,
                "messages": conversation,
                "max_tokens": 512,
                "temperature": 0.7,
            },
            timeout=30,
        )
        data = response.json()
        if "error" in data:
            return f"Nvidia Error: {data['error']}"
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Nvidia Error: {str(e)}"
