from model.client import generate

test = [{"role": "user", "content": "hi"}]

print("groq:", generate("groq", "llama-3.1-8b-instant", test)[:60])
print("cerebras:", generate("cerebras", "llama3.1-8b", test)[:60])
print("gemini:", generate("gemini", "gemini-2.5-flash", test)[:60])
print("sambanova:", generate("sambanova",
      "Meta-Llama-3.1-8B-Instruct", test)[:60])
print("mistral:", generate("mistral", "mistral-small-latest", test)[:60])
print("nvidia:", generate("nvidia", "meta/llama-3.1-8b-instruct", test)[:60])
print("cohere:", generate("cohere", "command-a-03-2025", test)[:60])
print("openrouter:", generate("openrouter", "openrouter/auto", test)[:60])
