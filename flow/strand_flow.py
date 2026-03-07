import asyncio
import aiohttp
import re
from strands import Agent, tool
from strands.models.openai import OpenAIModel
from fusion.fusion_engine import fuse
from config.settings import MODELS, FUSION_MODEL, CEREBRAS_API_KEY, GROQ_API_KEY, SAMBANOVA_API_KEY
import model.groq as groq
import model.cerebras as cerebras
import model.gemini as gemini
import model.sambanova as sambanova
import model.mistral as mistral
import model.nvidia as nvidia
import model.cohere as cohere
import model.openrouter as openrouter

MAX_HISTORY_TURNS = 3

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


class RequestContext:
    """Per-request context — thread-safe, no globals."""

    def __init__(self, image_data=None, image_mime=None):
        self.image_data = image_data
        self.image_mime = image_mime
        self.has_image = image_data is not None


class FinalTextCapture:
    """Captures only final assistant text — blocks tool call leaking."""

    def __init__(self):
        self._buffer = ""
        self._in_tool = False

    def __call__(self, **kwargs):
        event = kwargs.get("event", {})
        if event.get("type") == "tool_use":
            self._in_tool = True
            return
        if event.get("type") == "tool_result":
            self._in_tool = False
            return
        if not self._in_tool:
            delta = kwargs.get("data", "")
            if isinstance(delta, str) and delta:
                self._buffer += delta

    def get_result(self):
        return self._buffer.strip()


class StreamingCapture:
    """Captures text chunks for streaming — yields them via queue."""

    def __init__(self):
        self._queue = asyncio.Queue() if False else None
        self._chunks = []
        self._in_tool = False
        self._done = False

    def __call__(self, **kwargs):
        event = kwargs.get("event", {})
        if event.get("type") == "tool_use":
            self._in_tool = True
            return
        if event.get("type") == "tool_result":
            self._in_tool = False
            return
        if not self._in_tool:
            delta = kwargs.get("data", "")
            if isinstance(delta, str) and delta:
                self._chunks.append(delta)

    def get_chunks(self):
        return self._chunks

    def get_result(self):
        return "".join(self._chunks).strip()


def clean_response(text):
    text = re.sub(r'\{[^{}]*"type"\s*:\s*"function"[^{}]*\}',
                  '', text, flags=re.DOTALL)
    text = re.sub(
        r'\{[^{}]*"name"\s*:\s*"query_all_llms"[^{}]*\}', '', text, flags=re.DOTALL)
    text = re.sub(r'```json.*?```', '', text, flags=re.DOTALL)
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
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
                "messages": [
                    {
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
                    }
                ],
                "max_tokens": 4000,
            },
            timeout=aiohttp.ClientTimeout(total=15),
        ) as response:
            data = await response.json()
            if "error" in data:
                return None
            return data["choices"][0]["message"]["content"]
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
                "messages": [
                    {
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
                    }
                ],
                "max_tokens": 4000,
            },
            timeout=aiohttp.ClientTimeout(total=15),
        ) as response:
            data = await response.json()
            if "error" in data:
                return None
            return data["choices"][0]["message"]["content"]
    except Exception:
        return None


async def query_all_async(session, prompt, ctx: RequestContext):
    conversation = [{"role": "user", "content": prompt}]
    tasks = []

    for m in MODELS:
        if m["provider"] not in ASYNC_PROVIDERS:
            continue
        if m.get("vision") and ctx.has_image:
            tasks.append(
                ASYNC_PROVIDERS[m["provider"]].async_generate(
                    session, m["model"], conversation,
                    image_data=ctx.image_data,
                    image_mime=ctx.image_mime
                )
            )
        else:
            tasks.append(
                ASYNC_PROVIDERS[m["provider"]].async_generate(
                    session, m["model"], conversation
                )
            )

    results = await asyncio.gather(*tasks, return_exceptions=True)

    responses = []
    for r in results:
        if r and isinstance(r, str) and "Error:" not in r:
            responses.append(r[:800])
    return responses


def pre_fuse(responses):
    if len(responses) <= 4:
        return responses
    chunks = [responses[i:i+4] for i in range(0, len(responses), 4)]
    merged = []
    for chunk in chunks:
        combined = " | ".join(chunk)
        merged.append(combined[:4000])
    return merged


def make_query_tool(ctx: RequestContext):
    @tool
    def query_all_llms(prompt: str) -> str:
        """Query all LLMs and return combined responses."""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            async def _run():
                async with aiohttp.ClientSession() as session:
                    if ctx.has_image:
                        ocr_task = groq_ocr_async(
                            session, ctx.image_data, ctx.image_mime)
                        query_task = query_all_async(session, prompt, ctx)
                        ocr_text, responses = await asyncio.gather(ocr_task, query_task)

                        if ocr_text:
                            enriched_prompt = f"{prompt}\n\nExtracted Content:\n{ocr_text}"
                            text_ctx = RequestContext()
                            responses = await query_all_async(session, enriched_prompt, text_ctx)
                        return responses
                    else:
                        return await query_all_async(session, prompt, ctx)

            responses = loop.run_until_complete(_run())
            loop.close()
        except Exception:
            responses = []

        if not responses:
            return "All models failed to respond."

        merged = pre_fuse(responses)
        return "\n\n".join(f"Group {i+1}:\n{r}" for i, r in enumerate(merged))

    return query_all_llms


def get_fusion_agent(capture, ctx):
    fusion_model = OpenAIModel(
        client_args={
            "api_key": CEREBRAS_API_KEY,
            "base_url": "https://api.cerebras.ai/v1",
            "timeout": 180,
        },
        model_id="llama3.3-70b",
        params={"max_tokens": 16000, "temperature": 0.7},
    )

    query_tool = make_query_tool(ctx)

    return Agent(
        model=fusion_model,
        tools=[query_tool],
        callback_handler=capture,
        system_prompt="""You are FusionAI — a powerful AI assistant backed by 13 models.
STRICT RULES:
- Give extremely detailed, comprehensive, thorough answers — never cut off
- Answer every single part fully using proper structure a), b), i), ii) etc
- Include full working code examples with line-by-line explanation where needed
- Write in full sentences and paragraphs with depth
- Define every term, explain every concept fully
- Cover every point exhaustively — more detail is always better
- Never mention tools, models, APIs or technical details
- Never say what you have done or are about to do
- No intro, no outro, no disclaimers
- Just the answer, complete, detailed and direct""",
    )


def build_flow(conversation, image_data=None, image_mime=None):
    """Normal non-streaming flow — used for PDFs and fallback."""
    ctx = RequestContext(image_data=image_data, image_mime=image_mime)

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
        capture = FinalTextCapture()
        fusion_agent = get_fusion_agent(capture, ctx)
        fusion_agent(full_prompt)

        response = capture.get_result()
        if not response:
            raise Exception("Empty capture")

        return clean_response(response)

    except Exception:
        try:
            async def _fallback():
                async with aiohttp.ClientSession() as session:
                    return await query_all_async(session, full_prompt, ctx)

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            responses = loop.run_until_complete(_fallback())
            loop.close()
        except Exception:
            responses = []

        return fuse(conversation[-1]["content"], responses)


def build_flow_stream(conversation, image_data=None, image_mime=None):
    """
    Streaming flow — yields text chunks as they arrive.
    Used for text and image queries only — NOT for PDFs.
    """
    ctx = RequestContext(image_data=image_data, image_mime=image_mime)

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
        capture = StreamingCapture()
        fusion_agent = get_fusion_agent(capture, ctx)
        fusion_agent(full_prompt)

        chunks = capture.get_chunks()
        if not chunks:
            raise Exception("Empty stream")

        for chunk in chunks:
            cleaned = clean_response(chunk) if chunk else ""
            if cleaned:
                yield cleaned

    except Exception:
        try:
            async def _fallback():
                async with aiohttp.ClientSession() as session:
                    return await query_all_async(session, full_prompt, ctx)

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            responses = loop.run_until_complete(_fallback())
            loop.close()
        except Exception:
            responses = []

        result = fuse(conversation[-1]["content"], responses)
        yield result
