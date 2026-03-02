from model.client import generate
from config.settings import MODELS, FUSION_MODEL


def build_flow(prompt):

    responses = []
    for m in MODELS:
        result = generate(m["provider"], m["model"], prompt)
        if "Error:" not in result:
            responses.append((m["provider"], m["model"], result))

    combined = "\n\n".join(
        f"Answer {i+1} ({provider}/{model}):\n{result}"
        for i, (provider, model, result) in enumerate(responses)
    )

    fusion_prompt = f"""You are an AI judge.
Combine the following answers into one single improved response.

Question:
{prompt}

{combined}
"""

    final = generate(FUSION_MODEL["provider"], FUSION_MODEL["model"], fusion_prompt)
    return final
