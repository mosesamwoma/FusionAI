import asyncio
import aiohttp
from model.client import generate, async_generate
from config.settings import FUSION_MODEL


def fuse(question, answers):

    valid_answers = [
        a for a in answers
        if a and "Error:" not in a and "[SKIP]" not in a
    ]

    if not valid_answers:
        return "All models failed to respond. Please try again."

    combined = "\n\n".join(
        f"Answer {i+1}:\n{a[:800]}"  # increased from 200
        for i, a in enumerate(valid_answers)
    )

    fusion_prompt = f"""You are answering an exam or assignment question. Give complete, detailed, thorough answers.
- Answer every single part fully using proper structure a), b), i), ii) etc
- Include full working code examples with explanation where needed
- Write in full sentences and paragraphs with depth
- Define every term, explain every concept fully
- Never cut off mid-answer
- No intro, no outro, no disclaimers

Question: {question}

Reference answers:
{combined}
"""

    return generate(
        FUSION_MODEL["provider"],
        FUSION_MODEL["model"],
        [{"role": "user", "content": fusion_prompt}],
    )


async def async_fuse(question, answers):
    """Async version of fuse — uses async_generate for fusion model."""

    valid_answers = [
        a for a in answers
        if a and "Error:" not in a and "[SKIP]" not in a
    ]

    if not valid_answers:
        return "All models failed to respond. Please try again."

    combined = "\n\n".join(
        f"Answer {i+1}:\n{a[:800]}"  # increased from 200
        for i, a in enumerate(valid_answers)
    )

    fusion_prompt = f"""You are answering an exam or assignment question. Give complete, detailed, thorough answers.
- Answer every single part fully using proper structure a), b), i), ii) etc
- Include full working code examples with explanation where needed
- Write in full sentences and paragraphs with depth
- Define every term, explain every concept fully
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
            return result if result else fuse(question, answers)
    except Exception:
        return fuse(question, answers)
