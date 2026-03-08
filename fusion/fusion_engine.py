import aiohttp
from model.client import generate, async_generate
from config.settings import FUSION_MODEL


def fuse(question, answers):
    valid_answers = [
        a for a in answers
        if a and "Error:" not in a and "[SKIP]" not in a
    ]

    if not valid_answers:
        return "I'm unable to get a response right now. Please try again."

    combined = "\n\n".join(
        f"Answer {i+1}:\n{a[:800]}"
        for i, a in enumerate(valid_answers)
    )

    fusion_prompt = f"""You are FusionAI. Give complete, detailed answers.
- Answer every part fully using proper structure a), b), i), ii) etc
- Include code examples where needed
- Never cut off mid-answer
- No intro, no outro, no disclaimers

Question: {question}

Reference answers:
{combined}
"""

    try:
        result = generate(
            FUSION_MODEL["provider"],
            FUSION_MODEL["model"],
            [{"role": "user", "content": fusion_prompt}],
        )
        return result if result else valid_answers[0]
    except Exception:
        return valid_answers[0]


async def async_fuse(question, answers):
    valid_answers = [
        a for a in answers
        if a and "Error:" not in a and "[SKIP]" not in a
    ]

    if not valid_answers:
        return "I'm unable to get a response right now. Please try again."

    combined = "\n\n".join(
        f"Answer {i+1}:\n{a[:800]}"
        for i, a in enumerate(valid_answers)
    )

    fusion_prompt = f"""You are FusionAI. Give complete, detailed answers.
- Answer every part fully using proper structure a), b), i), ii) etc
- Include code examples where needed
- Never cut off mid-answer
- No intro, no outro, no disclaimers

Question: {question}

Reference answers:
{combined}
"""

    try:
        async with aiohttp.ClientSession() as session:
            result = await async_generate(
                session,
                FUSION_MODEL["provider"],
                FUSION_MODEL["model"],
                [{"role": "user", "content": fusion_prompt}],
            )
            if result:
                return result
            return fuse(question, answers)
    except Exception:
        return fuse(question, answers)
