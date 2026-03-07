import concurrent.futures
from strands import Agent, tool
from strands.models.openai import OpenAIModel
from fusion.fusion_engine import fuse
from model.client import generate
from config.settings import MODELS, FUSION_MODEL, GROQ_API_KEY


MAX_HISTORY_TURNS = 5


def trim_conversation(conversation):
    if len(conversation) <= MAX_HISTORY_TURNS * 2:
        return conversation
    return conversation[-(MAX_HISTORY_TURNS * 2):]


def query_all_models(prompt):
    def query(m):
        try:
            result = generate(m["provider"], m["model"], [
                              {"role": "user", "content": prompt}])
            if result and "Error:" not in result:
                return result[:600]
            return None
        except Exception:
            return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(MODELS)) as executor:
        futures = {executor.submit(query, m): m for m in MODELS}
        done, not_done = concurrent.futures.wait(futures, timeout=20)
        for f in not_done:
            f.cancel()

        responses = []
        for future in done:
            try:
                result = future.result()
                if result:
                    responses.append(result)
            except Exception:
                continue
    return responses


@tool
def query_all_llms(prompt: str) -> str:
    """Query all configured LLM models in parallel and return their combined responses."""
    responses = query_all_models(prompt)
    if not responses:
        return "All models failed to respond."
    combined = "\n\n".join(
        f"Model {i+1}:\n{r}" for i, r in enumerate(responses))
    return combined


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

    # Try Strands fusion first
    try:
        fusion_model = OpenAIModel(
            client_args={
                "api_key": GROQ_API_KEY,
                "base_url": "https://api.groq.com/openai/v1",
                "timeout": 30,
            },
            model_id=FUSION_MODEL["model"],
            params={"max_tokens": 1024, "temperature": 0.7},
        )

        fusion_agent = Agent(
            model=fusion_model,
            tools=[query_all_llms],
            callback_handler=None,
            system_prompt="""You are FusionAI — an expert AI response synthesizer.
When given a question:
1. Call the query_all_llms tool with the current question
2. Synthesize all responses into ONE complete, clear, accurate final answer
3. Never mention models, tools, APIs or technical details
4. Always produce a complete answer""",
        )

        result = fusion_agent(full_prompt)
        return str(result)

    except Exception:
        # Fallback to direct fusion if Strands fails
        responses = query_all_models(full_prompt)
        return fuse(conversation[-1]["content"], responses)
