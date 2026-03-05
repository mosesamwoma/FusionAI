from strands import Agent, tool
from strands.models.openai import OpenAIModel
from fusion.fusion_engine import fuse
from model.client import generate
from config.settings import MODELS, FUSION_MODEL, GROQ_API_KEY


collected_responses = []


def make_tool(provider, model_name):
    tool_name = f"query_{provider}_{model_name.replace('/', '_').replace('-', '_').replace('.', '_').replace(':', '_')}"

    @tool(name=tool_name, description=f"Query {provider} model {model_name}")
    def model_tool(prompt: str) -> str:
        result = generate(provider, model_name, [
                          {"role": "user", "content": prompt}])
        if "Error:" not in result:
            collected_responses.append(result)
        return result if "Error:" not in result else "[SKIP]"

    return model_tool


model_tools = [make_tool(m["provider"], m["model"]) for m in MODELS]


def build_flow(conversation):
    global collected_responses
    collected_responses = []

    latest_prompt = conversation[-1]["content"]

    if len(conversation) > 1:
        history = "\n".join(
            f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
            for m in conversation[:-1]
        )
        latest_prompt = f"Conversation so far:\n{history}\n\nLatest question: {latest_prompt}"

    fusion_model = OpenAIModel(
        client_args={
            "api_key": GROQ_API_KEY,
            "base_url": "https://api.groq.com/openai/v1",
        },
        model_id=FUSION_MODEL["model"],
        params={"max_tokens": 2048, "temperature": 0.7},
    )

    fusion_agent = Agent(
        model=fusion_model,
        tools=model_tools,
        callback_handler=None,
        system_prompt="""You are an AI orchestrator.
For every question:
- Call each tool one at a time with the user question
- After calling all tools respond with exactly: FUSION_READY
- Do not write anything else""",
    )

    fusion_agent(latest_prompt)

    return fuse(conversation[-1]["content"], collected_responses)
