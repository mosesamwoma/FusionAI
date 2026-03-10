import re
import hashlib
import time
from collections import defaultdict
import aiohttp
from model.client import generate, async_generate
from config.settings import FUSION_MODEL


CACHE = {}
CACHE_TTL = 86400


CODE_KEYWORDS = [
    "code", "program", "script", "function", "class", "algorithm",
    "implement", "write", "syntax", "example", "snippet", "hello world",
    "print", "def ", "import ", "return ", "```"
]


def get_cache_key(question):
    return hashlib.md5(question.lower().strip().encode()).hexdigest()


def cache_get(question):
    key = get_cache_key(question)
    if key in CACHE:
        result, timestamp = CACHE[key]
        if time.time() - timestamp < CACHE_TTL:
            print(f"Cache hit: {question[:50]}")
            return result
        del CACHE[key]
    return None


def cache_set(question, result):
    key = get_cache_key(question)
    CACHE[key] = (result, time.time())


def is_code_question(question):
    q = question.lower()
    return any(kw in q for kw in CODE_KEYWORDS)


def has_code_blocks(responses):
    return any("```" in r for r in responses)


def lcs_length(a, b):
    tokens_a = a.lower().split()
    tokens_b = b.lower().split()
    m, n = len(tokens_a), len(tokens_b)
    if m == 0 or n == 0:
        return 0
    dp = [[0] * (n + 1) for _ in range(2)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if tokens_a[i-1] == tokens_b[j-1]:
                dp[i % 2][j] = dp[(i-1) % 2][j-1] + 1
            else:
                dp[i % 2][j] = max(dp[(i-1) % 2][j], dp[i % 2][j-1])
    return dp[m % 2][n]


_rouge_cache = {}


def rouge_l(a, b):
    key = (a, b) if a <= b else (b, a)
    if key in _rouge_cache:
        return _rouge_cache[key]
    len_a = len(a.split())
    len_b = len(b.split())
    if len_a == 0 or len_b == 0:
        _rouge_cache[key] = 0.0
        return 0.0
    lcs = lcs_length(a, b)
    precision = lcs / len_b
    recall = lcs / len_a
    if precision + recall == 0:
        _rouge_cache[key] = 0.0
        return 0.0
    score = 2 * precision * recall / (precision + recall)
    _rouge_cache[key] = score
    return score


MODEL_WEIGHTS = {
    "groq-70b":         1.5,
    "cerebras-70b":     1.5,
    "gemini-2.5-flash": 1.5,
    "nvidia-70b":       1.5,
    "mistral-small":    1.2,
    "cohere-a":         1.2,
    "groq-8b":          1.0,
    "cerebras-8b":      1.0,
    "sambanova-8b":     1.0,
    "nvidia-8b":        1.0,
    "mistral-7b":       1.0,
    "openrouter-llama": 0.8,
    "openrouter-gemma": 0.8,
}

DEFAULT_WEIGHT = 1.0


def split_sentences(text, max_sentences=3):
    text = text.strip()
    sentences = re.split(r'(?<=[.!?])\s+', text)
    result = [s.strip() for s in sentences if len(s.strip()) > 20]
    return result[:max_sentences]


def score_sentences_weighted(all_sentences_per_response):
    weights = list(MODEL_WEIGHTS.values())
    indexed = []
    for resp_idx, sentences in enumerate(all_sentences_per_response):
        weight = weights[resp_idx] if resp_idx < len(
            weights) else DEFAULT_WEIGHT
        for sent in sentences:
            indexed.append((resp_idx, sent, weight))

    scores = defaultdict(float)
    for i, (resp_i, sent_i, _) in enumerate(indexed):
        for j, (resp_j, sent_j, weight_j) in enumerate(indexed):
            if i == j or resp_i == resp_j:
                continue
            rl = rouge_l(sent_i, sent_j)
            if rl > 0.3:
                scores[i] += rl * weight_j

    return indexed, scores


def mmr_select(indexed, scores, top_n=12, lambda_param=0.7):
    if not scores:
        return []
    ranked = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    selected = []
    selected_texts = []
    for idx in ranked:
        sentence = indexed[idx][1]
        if not selected_texts:
            selected.append(sentence)
            selected_texts.append(sentence)
        else:
            max_sim = max(rouge_l(sentence, s) for s in selected_texts)
            mmr_score = lambda_param * \
                scores[idx] - (1 - lambda_param) * max_sim
            if mmr_score > 0:
                selected.append(sentence)
                selected_texts.append(sentence)
        if len(selected) >= top_n:
            break
    return selected


def algorithmic_fuse(responses):
    if not responses:
        return ""
    _rouge_cache.clear()
    all_sentences = [split_sentences(r, max_sentences=3) for r in responses]
    indexed, scores = score_sentences_weighted(all_sentences)
    if not scores:
        return max(responses, key=len)
    best = mmr_select(indexed, scores, top_n=12)
    if not best:
        return max(responses, key=len)
    return " ".join(best)


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
    cached = cache_get(question)
    if cached:
        return cached

    valid = [a for a in answers if a and "Error:" not in a and "[SKIP]" not in a]
    if not valid:
        return "I'm unable to get a response right now. Please try again."

    if is_code_question(question) or has_code_blocks(valid):
        combined = "\n\n".join(
            f"Answer {i+1}:\n{a[:800]}" for i, a in enumerate(valid))
    else:
        combined = algorithmic_fuse(valid)

    fusion_prompt = build_fusion_prompt(question, combined)

    try:
        result = generate(
            FUSION_MODEL["provider"],
            FUSION_MODEL["model"],
            [{"role": "user", "content": fusion_prompt}],
        )
        if result:
            cache_set(question, result)
        return result if result else combined
    except Exception:
        return combined


async def async_fuse(question, answers):
    cached = cache_get(question)
    if cached:
        return cached

    valid = [a for a in answers if a and "Error:" not in a and "[SKIP]" not in a]
    if not valid:
        return "I'm unable to get a response right now. Please try again."

    if is_code_question(question) or has_code_blocks(valid):
        combined = "\n\n".join(
            f"Answer {i+1}:\n{a[:800]}" for i, a in enumerate(valid))
    else:
        combined = algorithmic_fuse(valid)

    fusion_prompt = build_fusion_prompt(question, combined)

    try:
        async with aiohttp.ClientSession() as session:
            result = await async_generate(
                session,
                FUSION_MODEL["provider"],
                FUSION_MODEL["model"],
                [{"role": "user", "content": fusion_prompt}],
            )
            if result:
                cache_set(question, result)
            return result if result else combined
    except Exception:
        return combined
