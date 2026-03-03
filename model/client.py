from model import groq, cerebras, gemini, cloudflare, sambanova, mistral


def generate(provider, model_name, prompt):
    provider = provider.lower().strip()

    if provider == "groq":
        return groq.generate(model_name, prompt)
    if provider == "cerebras":
        return cerebras.generate(model_name, prompt)
    if provider == "gemini":
        return gemini.generate(model_name, prompt)
    if provider == "cloudflare":
        return cloudflare.generate(model_name, prompt)
    if provider == "sambanova":
        return sambanova.generate(model_name, prompt)
    if provider == "mistral":
        return mistral.generate(model_name, prompt)

    raise ValueError(f"Unknown provider: {provider}")
