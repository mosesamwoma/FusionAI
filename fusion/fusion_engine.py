import re
from collections import defaultdict
import aiohttp
from model.client import generate, async_generate
from config.settings import FUSION_MODEL


# ─── Algorithmic Fusion ───────────────────────────────────────────────────────

def split_sentences(text):
    text = text.strip()
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if len(s.strip()) > 20]


def normalize(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def word_overlap_score(a, b):
    words_a = set(normalize(a).split())
    words_b = set(normalize(b).split())
    if not words_a or not words_b:
        return 0.0
    return len(words_a & words_b) / len(words_a | words_b)


def is_duplicate(sentence, accepted, threshold=0.7):
    for s in accepted:
        if word_overlap_score(sentence, s) >= threshold:
            return True
    return False


def score_sentences(all_sentences_per_response):
    indexed = []
    for resp_idx, sentences in enumerate(all_sentences_per_response):
        for sent in sentences:
            indexed.append((resp_idx, sent))

    scores = defaultdict(float)
    for i, (resp_i, sent_i) in enumerate(indexed):
        for j, (resp_j, sent_j) in enumerate(indexed):
            if i == j or resp_i == resp_j:
                continue
            overlap = word_overlap_score(sent_i, sent_j)
            if overlap > 0.3:
                scores[i] += overlap

    return indexed, scores


def select_best_sentences(indexed, scores, top_n=12):
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    accepted = []
    for idx, score in ranked:
        sentence = indexed[idx][1]
        if not is_duplicate(sentence, accepted):
            accepted.append(sentence)
        if len(accepted) >= top_n:
            break
    return accepted


def algorithmic_fuse(responses):
    """
    Score and rank sentences across all responses by cross-model agreement.
    Returns the top pre-filtered sentences as a condensed string.
    """
    if not responses:
        return ""

    all_sentences = [split_sentences(r) for r in responses]
    indexed, scores = score_sentences(all_sentences)

    if not scores:
        return max(responses, key=len)

    best = select_best_sentences(indexed, scores, top_n=12)

    if not best:
        return max(responses, key=len)

    return " ".join(best)


# ─── Model Fusion ─────────────────────────────────────────────────────────────

def build_fusion_prompt(question, content):
    return f"""You are FusionAI, a helpful AI assistant. Synthesize the reference answers into one natural, well-formatted response.

RULES:
- For casual conversation (greetings, small talk) — respond naturally and briefly
- For technical or academic questions — be detailed and structured
- For document or exam content — preserve structure, use headings, numbered lists, code blocks
- For questions with multiple parts — use proper markdown: ##, ###, **, -, 1. 2. 3.
- Use markdown code blocks (```java, ```python etc) for all code
- Never cut off mid-answer
- No intro like "Here is..." or "Sure!" — go straight to the answer
- No outro, no disclaimers
- Match the tone to the question

Question: {question}

Reference answers:
{content}"""


def fuse(question, answers):
    valid = [a for a in answers if a and "Error:" not in a and "[SKIP]" not in a]

    if not valid:
        return "I'm unable to get a response right now. Please try again."

    # Step 1: Algorithm pre-filters to best sentences across all models
    pre_filtered = algorithmic_fuse(valid)

    # Step 2: Model synthesizes the already-best content
    fusion_prompt = build_fusion_prompt(question, pre_filtered)

    try:
        result = generate(
            FUSION_MODEL["provider"],
            FUSION_MODEL["model"],
            [{"role": "user", "content": fusion_prompt}],
        )
        return result if result else pre_filtered
    except Exception:
        return pre_filtered


async def async_fuse(question, answers):
    valid = [a for a in answers if a and "Error:" not in a and "[SKIP]" not in a]

    if not valid:
        return "I'm unable to get a response right now. Please try again."

    # Step 1: Algorithm pre-filters to best sentences across all models
    pre_filtered = algorithmic_fuse(valid)

    # Step 2: Model synthesizes the already-best content
    fusion_prompt = build_fusion_prompt(question, pre_filtered)

    try:
        async with aiohttp.ClientSession() as session:
            result = await async_generate(
                session,
                FUSION_MODEL["provider"],
                FUSION_MODEL["model"],
                [{"role": "user", "content": fusion_prompt}],
            )
            return result if result else pre_filtered
    except Exception:
        return pre_filtered
