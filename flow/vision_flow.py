import asyncio
import aiohttp
import re
import json
from config.settings import MODELS, CEREBRAS_API_KEY, GROQ_API_KEY, SAMBANOVA_API_KEY
from fusion.fusion_engine import fuse, algorithmic_fuse, is_code_question, has_code_blocks, cache_get, cache_set
import model.groq as groq
import model.cerebras as cerebras
import model.gemini as gemini
import model.sambanova as sambanova
import model.mistral as mistral
import model.nvidia as nvidia
import model.cohere as cohere
import model.openrouter as openrouter

MAX_HISTORY_TURNS = 3
MIN_RESPONSES = 13
COLLECT_TIMEOUT = 15

ASYNC_PROVIDERS = {
    "groq": groq,
    "cerebras": cerebras,
    "gemini": gemini,
    "sambanova": sambanova,
    "mistral": mistral,
    "nvidia": nvidia,
    "cohere": cohere,
    "openrouter": openrouter,
}

LEAKED_PHRASES = [
    "I will continue to find a fitting response",
    "I will continue to find",
    "I will continue",
    "I'll continue",
    "Let me search",
    "I'll search",
    "I'll find",
    "I'm going to",
    "I will now",
    "Let me look",
    "I will look",
    "I need to search",
    "I'll look",
    "allow me to",
    "I will use",
    "I'll use the",
    "In this response, I have provided",
    "I have provided the answers",
    "I have also made sure to follow the guidelines",
    "I have made sure to",
    "in a clear and concise manner",
    "as per the guidelines",
    "as requested",
    "as per your request",
    "I hope this helps",
    "I hope this answer",
    "Please let me know if",
    "Feel free to ask",
    "Note that",
    "Please note",
]


def clean_response(text):
    if not text:
        return ""
    text = re.sub(r'\{[^{}]*"type"\s*:\s*"function"[^{}]*\}',
                  '', text, flags=re.DOTALL)
    text = re.sub(
        r'\{[^{}]*"name"\s*:\s*"query_all_llms"[^{}]*\}', '', text, flags=re.DOTALL)
    text = re.sub(r'```json.*?```', '', text, flags=re.DOTALL)
    for phrase in LEAKED_PHRASES:
        text = text.replace(phrase, "")
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def trim_conversation(conversation):
    if len(conversation) <= MAX_HISTORY_TURNS * 2:
        return conversation
    return conversation[-(MAX_HISTORY_TURNS * 2):]


async def groq_ocr_async(session, image_data, image_mime):
    try:
        async with session.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "llama-3.2-11b-vision-preview",
                "messages": [{
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{image_mime};base64,{image_data}"}
                        },
                        {
                            "type": "text",
                            "text": "Extract ALL text from this image exactly as it appears. Preserve all formatting, numbering, and structure. Return only the extracted text, nothing else."
                        }
                    ]
                }],
                "max_tokens": 4000,
            },
            timeout=aiohttp.ClientTimeout(total=15),
        ) as response:
            try:
                data = await response.json(content_type=None)
            except Exception:
                return None
            if not isinstance(data, dict):
                return None
            if "error" in data:
                return None
            if "choices" in data and data["choices"]:
                return data["choices"][0]["message"]["content"]
            return None
    except Exception:
        return None


async def sambanova_ocr_async(session, image_data, image_mime):
    try:
        async with session.post(
            "https://api.sambanova.ai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {SAMBANOVA_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "Llama-3.2-11B-Vision-Instruct",
                "messages": [{
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{image_mime};base64,{image_data}"}
                        },
                        {
                            "type": "text",
                            "text": "Extract ALL text from this image exactly as it appears. Preserve all formatting, numbering, and structure. Return only the extracted text, nothing else."
                        }
                    ]
                }],
                "max_tokens": 4000,
            },
            timeout=aiohttp.ClientTimeout(total=15),
        ) as response:
            try:
                data = await response.json(content_type=None)
            except Exception:
                return None
            if not isinstance(data, dict):
                return None
            if "error" in data:
                return None
            if "choices" in data and data["choices"]:
                return data["choices"][0]["message"]["content"]
            return None
    except Exception:
        return None


async def collect_all_responses(session, prompt, image_data=None, image_mime=None):
    conversation = [{"role": "user", "content": prompt}]
    responses = []
    ready = asyncio.Event()

    async def fetch_one(provider_module, model_name, is_vision_model):
        try:
            if is_vision_model and image_data:
                result = await provider_module.async_generate(
                    session, model_name, conversation,
                    image_data=image_data,
                    image_mime=image_mime
                )
            else:
                result = await provider_module.async_generate(
                    session, model_name, conversation
                )
            if result and isinstance(result, str) and "Error:" not in result:
                responses.append(result[:800])
                if len(responses) >= MIN_RESPONSES:
                    ready.set()
        except Exception:
            pass

    tasks = [
        asyncio.create_task(
            fetch_one(ASYNC_PROVIDERS[m["provider"]],
                      m["model"], m.get("vision", False))
        )
        for m in MODELS if m["provider"] in ASYNC_PROVIDERS
    ]

    try:
        await asyncio.wait_for(ready.wait(), timeout=COLLECT_TIMEOUT)
    except asyncio.TimeoutError:
        pass

    for t in tasks:
        if not t.done():
            t.cancel()

    print(f"Vision got {len(responses)} responses")
    return responses


async def groq_fuse_stream(session, question, responses):
    if is_code_question(question) or has_code_blocks(responses):
        pre_filtered = "\n\n".join(
            f"Answer {i+1}:\n{r[:800]}" for i, r in enumerate(responses))
    else:
        pre_filtered = algorithmic_fuse(responses)

    fusion_prompt = f"""You are FusionAI, a helpful AI assistant. Synthesize the reference answers into one natural response.

RULES:
- For casual conversation (greetings, small talk) — respond naturally and briefly
- For technical or academic questions — be detailed and structured
- For document or image content — extract and answer all questions thoroughly
- For questions with multiple parts — use proper structure a), b), i), ii) etc
- Include code examples only when relevant
- Never cut off mid-answer
- No intro, no outro, no disclaimers
- Match the tone to the question

Question: {question}

Reference answers:
{pre_filtered}"""

    full_response = []

    try:
        async with session.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": [{"role": "user", "content": fusion_prompt}],
                "max_tokens": 8192,
                "temperature": 0.7,
                "stream": True,
            },
            timeout=aiohttp.ClientTimeout(total=120),
        ) as response:
            async for line in response.content:
                line = line.decode("utf-8").strip()
                if not line or not line.startswith("data:"):
                    continue
                data_str = line[5:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    data = json.loads(data_str)
                    token = data["choices"][0]["delta"].get("content", "")
                    if token:
                        full_response.append(token)
                        yield token
                except Exception:
                    continue
    except Exception as e:
        print(f"Groq vision stream error: {e}")
        return
    finally:
        if full_response:
            cache_set(question, "".join(full_response))


async def groq_fuse_async(session, question, responses):
    if is_code_question(question) or has_code_blocks(responses):
        pre_filtered = "\n\n".join(
            f"Answer {i+1}:\n{r[:800]}" for i, r in enumerate(responses))
    else:
        pre_filtered = algorithmic_fuse(responses)

    fusion_prompt = f"""You are FusionAI, a helpful AI assistant. Synthesize the reference answers into one natural response.

RULES:
- For casual conversation (greetings, small talk) — respond naturally and briefly
- For technical or academic questions — be detailed and structured
- For document or image content — extract and answer all questions thoroughly
- For questions with multiple parts — use proper structure a), b), i), ii) etc
- Include code examples only when relevant
- Never cut off mid-answer
- No intro, no outro, no disclaimers
- Match the tone to the question

Question: {question}

Reference answers:
{pre_filtered}"""

    try:
        async with session.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": [{"role": "user", "content": fusion_prompt}],
                "max_tokens": 8192,
                "temperature": 0.7,
            },
            timeout=aiohttp.ClientTimeout(total=120),
        ) as response:
            try:
                data = await response.json(content_type=None)
            except Exception as e:
                print(f"Groq JSON parse error: {e}")
                return None
            if not isinstance(data, dict):
                return None
            if "error" in data:
                print(f"Groq API error: {data['error']}")
                return None
            if "choices" in data and data["choices"]:
                content = data["choices"][0]["message"]["content"]
                if content:
                    cache_set(question, content)
                return content if content else None
            return None
    except Exception as e:
        print(f"Groq vision fusion error: {e}")
        return None


async def run_vision_pipeline(prompt, question, image_data=None, image_mime=None):
    cached = cache_get(question)
    if cached and not image_data:
        print("Cache hit — skipping model queries")
        return cached

    async with aiohttp.ClientSession() as session:
        if image_data:
            ocr_task = groq_ocr_async(session, image_data, image_mime)
            query_task = collect_all_responses(
                session, prompt, image_data, image_mime)
            ocr_text, responses = await asyncio.gather(ocr_task, query_task)
            if ocr_text:
                responses.append(ocr_text[:800])
        else:
            responses = await collect_all_responses(session, prompt)

        if not responses:
            return "All models failed to respond. Please try again."

        result = await groq_fuse_async(session, question, responses)

        if result:
            cleaned = clean_response(result)
            return cleaned if cleaned else result

        fallback = fuse(question, responses)
        return fallback if fallback else "Something went wrong. Please try again."


def build_vision_flow(conversation, image_data=None, image_mime=None):
    trimmed = trim_conversation(conversation)
    latest_prompt = trimmed[-1]["content"]

    if len(trimmed) > 1:
        history = "\n".join(
            f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
            for m in trimmed[:-1]
        )
        full_prompt = f"Previous conversation:\n{history}\n\nCurrent question: {latest_prompt}"
    else:
        full_prompt = latest_prompt

    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(
            run_vision_pipeline(
                full_prompt,
                conversation[-1]["content"],
                image_data=image_data,
                image_mime=image_mime
            )
        )
        loop.close()
        return result if result else "Something went wrong. Please try again."
    except Exception as e:
        print(f"build_vision_flow error: {e}")
        import traceback
        traceback.print_exc()
        fallback = fuse(conversation[-1]["content"], [])
        return fallback if fallback else "Something went wrong. Please try again."
