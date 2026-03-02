from model import groq, cerebras


def generate(provider, model_name, prompt):

    if provider == "groq":
        return groq.generate(model_name, prompt)

    if provider == "cerebras":
        return cerebras.generate(model_name, prompt)

    raise ValueError(f"Unknown provider: {provider}")
