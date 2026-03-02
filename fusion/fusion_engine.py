from model.client import generate
from config.settings import FUSION_MODEL


def fuse(question, answers):

    valid_answers = [
        a for a in answers
        if "Error:" not in a
    ]

    combined = "\n\n".join(
        [f"Answer {i+1}:\n{a}" for i, a in enumerate(valid_answers)]
    )

    fusion_prompt = f"""
You are an AI judge.

Combine the following answers into one clear,
accurate and improved final answer.

Question:
{question}

{combined}
"""

    return generate(
        FUSION_MODEL["provider"],
        FUSION_MODEL["model"],
        fusion_prompt
    )
