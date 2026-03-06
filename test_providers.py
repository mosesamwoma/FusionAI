from model.client import generate

test = [{"role": "user", "content": "hi"}]


def safe(provider, model):
    try:
        result = generate(provider, model, test)

        if not result:
            return "No response"

        return result[:60]

    except Exception as e:
        return f"Error: {str(e)}"


print("groq-8b:", safe("groq", "llama-3.1-8b-instant"))
print("groq-70b:", safe("groq", "llama-3.3-70b-versatile"))

print("cerebras-8b:", safe("cerebras", "llama3.1-8b"))

print("gemini-2.5:", safe("gemini", "gemini-2.5-flash"))

print("sambanova-8b:", safe("sambanova", "Meta-Llama-3.1-8B-Instruct"))

print("mistral-small:", safe("mistral", "mistral-small-latest"))
print("mistral-7b:", safe("mistral", "open-mistral-7b"))

print("nvidia-8b:", safe("nvidia", "meta/llama-3.1-8b-instruct"))
print("nvidia-70b:", safe("nvidia", "meta/llama-3.1-70b-instruct"))

print("cohere-a:", safe("cohere", "command-a-03-2025"))

print("openrouter-gemma-12b:", safe("openrouter", "google/gemma-3-12b-it"))
print("openrouter-gemma-4b:", safe("openrouter", "google/gemma-3-4b-it"))

print("openrouter-llama:", safe("openrouter", "meta-llama/llama-3.1-8b-instruct"))
