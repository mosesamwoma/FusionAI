from model.client import generate
from config.settings import FUSION_MODEL


def fuse(question, answers):

    valid_answers = [
        a for a in answers
        if "Error:" not in a and "[SKIP]" not in a
    ]

    if not valid_answers:
        return "All models failed to respond. Please try again."

    combined = "\n\n".join(
        f"Answer {i+1}:\n{a}"
        for i, a in enumerate(valid_answers)
    )

    fusion_prompt = f"""
You are an AI judge.

Combine the following answers into one clear,
accurate and improved final answer.

Do NOT mention models, APIs, or technical issues.
Produce a clean final answer only.

Question:
{question}

{combined}
"""

    return generate(
        FUSION_MODEL["provider"],
        FUSION_MODEL["model"],
        [{"role": "user", "content": fusion_prompt}],
    )
