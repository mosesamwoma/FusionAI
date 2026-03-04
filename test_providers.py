from model.client import generate

test = [{"role": "user", "content": "hi"}]

print("groq:", generate("groq", "llama-3.1-8b-instant", test)[:60])
print("cerebras:", generate("cerebras", "llama3.1-8b", test)[:60])
print("gemini:", generate("gemini", "gemini-2.0-flash", test)[:60])
print("sambanova:", generate("sambanova",
      "Meta-Llama-3.1-8B-Instruct", test)[:60])
print("mistral:", generate("mistral", "mistral-small-latest", test)[:60])
