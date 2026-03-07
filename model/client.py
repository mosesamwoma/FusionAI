from model import groq, cerebras, gemini, nvidia, sambanova, mistral, cohere, openrouter

PROVIDER_MAP = {
    "groq": groq,
    "cerebras": cerebras,
    "gemini": gemini,
    "sambanova": sambanova,
    "mistral": mistral,
    "nvidia": nvidia,
    "cohere": cohere,
    "openrouter": openrouter,
}


def generate(provider, model_name, conversation):
    provider = provider.lower().strip()
    if provider not in PROVIDER_MAP:
        raise ValueError(f"Unknown provider: {provider}")
    return PROVIDER_MAP[provider].generate(model_name, conversation)


async def async_generate(session, provider, model_name, conversation):
    provider = provider.lower().strip()
    if provider not in PROVIDER_MAP:
        return None
    return await PROVIDER_MAP[provider].async_generate(session, model_name, conversation)
