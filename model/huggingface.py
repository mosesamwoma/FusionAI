import requests
from config.settings import HF_TOKEN


def generate(model_name, prompt):

    url = f"https://router.huggingface.co/hf-inference/models/{model_name}"

    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json",
    }

    response = requests.post(
        url,
        headers=headers,
        json={
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 300,
                "temperature": 0.7,
            },
        },
    )

    try:
        data = response.json()
    except Exception:
        return f"HuggingFace Error: Invalid JSON response {response.text}"

    # Handle API errors safely
    if isinstance(data, dict) and "error" in data:
        return f"HuggingFace Error: {data['error']}"

    if isinstance(data, list) and len(data) > 0:
        return data[0].get("generated_text", "No generated text returned")

    return f"HuggingFace Unexpected Response: {data}"
