import requests
from config.settings import COHERE_API_KEY


def generate(model_name, conversation):
    try:
        messages = []
        for msg in conversation:
            role = "USER" if msg["role"] == "user" else "CHATBOT"
            messages.append({"role": role, "message": msg["content"]})

        last_message = messages[-1]["message"]
        history = messages[:-1]

        response = requests.post(
            "https://api.cohere.com/v1/chat",
            headers={
                "Authorization": f"Bearer {COHERE_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": model_name,
                "message": last_message,
                "chat_history": history,
                "max_tokens": 512,
            },
            timeout=30,
        )
        data = response.json()
        if "message" in data and "text" not in data:
            return f"Cohere Error: {data['message']}"
        return data["text"]
    except Exception as e:
        return f"Cohere Error: {str(e)}"
