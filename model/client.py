from model import groq, cerebras, gemini, cloudflare


def generate(provider, model_name, prompt):
    if provider == "groq":
        return groq.generate(model_name, prompt)
    if provider == "cerebras":
        return cerebras.generate(model_name, prompt)
    if provider == "gemini":
        return gemini.generate(model_name, prompt)
    if provider == "cloudflare":
        return cloudflare.generate(model_name, prompt)
    raise ValueError(f"Unknown provider: {provider}")
