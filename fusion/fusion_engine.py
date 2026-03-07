from model.client import generate
from config.settings import FUSION_MODEL


def fuse(question, answers):

    valid_answers = [
        a for a in answers
        if a and "Error:" not in a and "[SKIP]" not in a
    ]

    if not valid_answers:
        return "All models failed to respond. Please try again."

    combined = "\n\n".join(
        f"Answer {i+1}:\n{a[:600]}"
        for i, a in enumerate(valid_answers)
    )

    fusion_prompt = f"""
You are an expert AI response synthesizer.
Combine ALL answers into one complete, clear, accurate and detailed final answer.
Do NOT cut off or truncate the response — always finish completely.
Do NOT mention models, APIs, or technical issues.
Produce a clean complete final answer only.

Question: {question}

{combined}
"""

    return generate(
        FUSION_MODEL["provider"],
        FUSION_MODEL["model"],
        [{"role": "user", "content": fusion_prompt}],
    )
