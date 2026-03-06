from model.client import generate

test = [{"role": "user", "content": "hi"}]

print("groq-8b:", generate("groq", "llama-3.1-8b-instant", test)[:60])
print("groq-70b:", generate("groq", "llama-3.3-70b-versatile", test)[:60])
print("cerebras-8b:", generate("cerebras", "llama3.1-8b", test)[:60])
print("gemini-2.5:", generate("gemini", "gemini-2.5-flash", test)[:60])
print("sambanova-8b:", generate("sambanova",
      "Meta-Llama-3.1-8B-Instruct", test)[:60])
print("mistral-small:", generate("mistral", "mistral-small-latest", test)[:60])
print("mistral-7b:", generate("mistral", "open-mistral-7b", test)[:60])
print("nvidia-8b:", generate("nvidia",
      "meta/llama-3.1-8b-instruct", test)[:60])
print("nvidia-70b:", generate("nvidia",
      "meta/llama-3.1-70b-instruct", test)[:60])
print("cohere-a:", generate("cohere", "command-a-03-2025", test)[:60])
print("openrouter-gemma-12b:", generate("openrouter",
      "google/gemma-3-12b-it:free", test)[:60])
print("openrouter-gemma-4b:", generate("openrouter",
      "google/gemma-3-4b-it:free", test)[:60])
