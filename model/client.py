from model import groq, cerebras, gemini, nvidia, sambanova, mistral, cohere, openrouter


def generate(provider, model_name, conversation):
    provider = provider.lower().strip()

    if provider == "groq":
        return groq.generate(model_name, conversation)
    if provider == "cerebras":
        return cerebras.generate(model_name, conversation)
    if provider == "gemini":
        return gemini.generate(model_name, conversation)
    if provider == "sambanova":
        return sambanova.generate(model_name, conversation)
    if provider == "mistral":
        return mistral.generate(model_name, conversation)
    if provider == "nvidia":
        return nvidia.generate(model_name, conversation)
    if provider == "cohere":
        return cohere.generate(model_name, conversation)
    if provider == "openrouter":
        return openrouter.generate(model_name, conversation)

    raise ValueError(f"Unknown provider: {provider}")