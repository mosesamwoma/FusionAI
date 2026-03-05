from model.client import generate
from config.settings import MODELS, FUSION_MODEL


def build_flow(conversation):

    responses = []
    for m in MODELS:
        try:
            result = generate(m["provider"], m["model"], conversation)
            if (
                result
                and isinstance(result, str)
                and "error" not in result.lower()
                and "api key" not in result.lower()
                and "invalid" not in result.lower()
                and "timeout" not in result.lower()
            ):
                responses.append((m["provider"], m["model"], result))
        except Exception:
            continue

    if not responses:
        return "All providers failed. Please try again."

    latest_prompt = conversation[-1]["content"]

    combined = "\n\n".join(
        f"[{provider}/{model}]\n{result}"
        for provider, model, result in responses
    )

    fusion_prompt = f"""
You are an expert AI response synthesizer.
Merge the answers below into ONE clear, coherent, improved response.
- Keep only correct and useful information.
- Remove contradictions.
- Do NOT mention APIs, models, or technical issues.
- Produce a clean final answer only.

User Question:
{latest_prompt}

Model Answers:
{combined}
"""

    final = generate(
        FUSION_MODEL["provider"],
        FUSION_MODEL["model"],
        conversation + [{"role": "user", "content": fusion_prompt}],
    )
    return final
