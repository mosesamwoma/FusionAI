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
Combine the answers below into one single direct response.
No preamble. No meta-commentary. No mention of models or synthesis.
Just answer directly as if you are the one answering.

Question: {question}

{combined}
"""

    return generate(
        FUSION_MODEL["provider"],
        FUSION_MODEL["model"],
        [{"role": "user", "content": fusion_prompt}],
    )
