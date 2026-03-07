import asyncio
import aiohttp
from strands import Agent, tool
from strands.models.openai import OpenAIModel
from fusion.fusion_engine import fuse
from config.settings import MODELS, FUSION_MODEL, CEREBRAS_API_KEY
import model.groq as groq
import model.cerebras as cerebras
import model.gemini as gemini
import model.sambanova as sambanova
import model.mistral as mistral
import model.nvidia as nvidia
import model.cohere as cohere
import model.openrouter as openrouter

MAX_HISTORY_TURNS = 5

ASYNC_PROVIDERS = {
    "groq": groq,
    "cerebras": cerebras,
    "gemini": gemini,
    "sambanova": sambanova,
    "mistral": mistral,
    "nvidia": nvidia,
    "cohere": cohere,
    "openrouter": openrouter,
}


def trim_conversation(conversation):
    if len(conversation) <= MAX_HISTORY_TURNS * 2:
        return conversation
    return conversation[-(MAX_HISTORY_TURNS * 2):]


async def query_all_async(prompt):
    conversation = [{"role": "user", "content": prompt}]
    async with aiohttp.ClientSession() as session:
        tasks = [
            ASYNC_PROVIDERS[m["provider"]].async_generate(
                session, m["model"], conversation)
            for m in MODELS
            if m["provider"] in ASYNC_PROVIDERS
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    responses = []
    for r in results:
        if r and isinstance(r, str) and "Error:" not in r:
            responses.append(r[:300])
    return responses


@tool
def query_all_llms(prompt: str) -> str:
    """Query all configured LLM models in parallel and return their combined responses."""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        responses = loop.run_until_complete(query_all_async(prompt))
        loop.close()
    except Exception:
        responses = []

    if not responses:
        return "All models failed to respond."

    return "\n\n".join(f"Model {i+1}:\n{r}" for i, r in enumerate(responses))


def build_flow(conversation):

    trimmed = trim_conversation(conversation)
    latest_prompt = trimmed[-1]["content"]

    if len(trimmed) > 1:
        history = "\n".join(
            f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
            for m in trimmed[:-1]
        )
        full_prompt = f"Previous conversation:\n{history}\n\nCurrent question: {latest_prompt}"
    else:
        full_prompt = latest_prompt

    try:
        fusion_model = OpenAIModel(
            client_args={
                "api_key": CEREBRAS_API_KEY,
                "base_url": "https://api.cerebras.ai/v1",
                "timeout": 30,
            },
            model_id=FUSION_MODEL["model"],
            params={"max_tokens": 1024, "temperature": 0.7},
        )

        fusion_agent = Agent(
            model=fusion_model,
            tools=[query_all_llms],
            callback_handler=None,
            system_prompt="""You are FusionAI — a direct, confident AI assistant.
Rules:
- Always respond directly and naturally
- Never mention models, synthesis, or technical details
- Never use phrases like 'based on', 'it seems', 'synthesized response'
- Just answer as if you are the one who knows the answer
- Be conversational, warm and helpful""",
        )

        result = fusion_agent(full_prompt)
        return str(result)

    except Exception:
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            responses = loop.run_until_complete(query_all_async(full_prompt))
            loop.close()
        except Exception:
            responses = []
        return fuse(conversation[-1]["content"], responses)
