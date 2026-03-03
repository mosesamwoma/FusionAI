import requests
from config.settings import CLOUDFLARE_API_KEY


def generate(model_name, prompt):
    try:
        response = requests.post(
            "https://gateway.ai.cloudflare.com/v1/dd8a7a266fa56eeecdc6fccca35334d9/musa/compat/chat/completions",
            headers={
                "Authorization": f"Bearer {CLOUDFLARE_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": model_name,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
            },
            timeout=30,
        )

        data = response.json()

        if "error" in data:
            return f"Cloudflare Error: {data['error']}"

        return data["choices"][0]["message"]["content"]

    except Exception as e:
        return f"Cloudflare Error: {str(e)}"
