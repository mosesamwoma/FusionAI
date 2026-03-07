# FusionAI

A multi-model AI fusion system that queries multiple LLMs simultaneously and synthesizes their responses into one superior answer using [Strands Agents](https://github.com/strands-agents/sdk-python).

## How It Works

1. A user prompt is sent to all 13 configured LLMs in parallel using `asyncio` + `aiohttp`
2. A Strands Agent `@tool` called `query_all_llms` handles all async model queries
3. The Strands Agent collects all responses and synthesizes them into one final answer
4. Automatic conversation trimming prevents token limit issues

## Models Used

| Provider | Models |
|----------|--------|
| Groq | llama-3.1-8b-instant, llama-3.3-70b-versatile |
| Cerebras | llama3.1-8b |
| Gemini | gemini-2.5-flash |
| SambaNova | Meta-Llama-3.1-8B-Instruct |
| Mistral | mistral-small-latest, open-mistral-7b |
| Nvidia | meta/llama-3.1-8b-instruct, meta/llama-3.1-70b-instruct |
| Cohere | command-a-03-2025 |
| OpenRouter | gemma-3-12b-it, gemma-3-4b-it, llama-3.1-8b-instruct |

## Requirements

- Python 3.11+
- Strands Agents SDK
- aiohttp

## Configuration

Create a `.env` file in the project root:
```
GROQ_API_KEY=your_groq_key
CEREBRAS_API_KEY=your_cerebras_key
GEMINI_API_KEY=your_gemini_key
SAMBANOVA_API_KEY=your_sambanova_key
MISTRAL_API_KEY=your_mistral_key
NVIDIA_API_KEY=your_nvidia_key
COHERE_API_KEY=your_cohere_key
OPENROUTER_API_KEY=your_openrouter_key
```

## Activate Environment
```bash
.venv\Scripts\Activate.ps1
```

## Install Dependencies
```bash
pip install -r requirements.txt
```

## Test Providers

Before running the app, verify all APIs are working:
```bash
python test_providers.py
```

Expected output:
```
groq-8b: How can I assist you today?
groq-70b: It's nice to meet you. Is there something I can help you with
cerebras-8b: How can I assist you today?
gemini-2.5: Hi there! How can I help you today?
sambanova-8b: How can I assist you today?
mistral-small: Hello! 😊 How can I help you today?
mistral-7b: Hello! 😊 How can I help you today?
nvidia-8b: How can I assist you today?
nvidia-70b: How can I assist you today?
cohere-a: Hello! How can I assist you today?
openrouter-gemma-12b: Hi there! How can I help you today?
openrouter-gemma-4b: Hi there! How can I help you today?
openrouter-llama: Hi there! How can I help you today?
```

If a provider shows an error, check its API key in `.env`.

## Run CLI
```bash
python main.py
```

## Run Web App
```bash
python app.py
```

Then open **http://127.0.0.1:5000** in your browser.

## Usage Example
```
You: What is data science?

FusionAI: Data science is an interdisciplinary field that combines statistics,
computer science, and domain expertise to extract insights and knowledge from data,
driving informed business decisions, strategy, and operations,
and solving complex problems across various fields.
```

## Status & Limitations

> ⚠️ This project is experimental and not production-ready.

- Response time depends on the slowest API provider
- API rate limits and daily quotas may affect availability
- Failed or timed out models are skipped silently
- Conversation history trimmed to last 5 turns to avoid token limits
- No streaming — response appears only after all models finish
- Single user only — no multi-user support yet
- Fusion quality depends on how many models respond successfully

## Completed Features

- ✅ 13 models queried in parallel using asyncio
- ✅ Strands Agent for intelligent response fusion
- ✅ Flask web UI with modern dark theme
- ✅ CLI interface
- ✅ Chat history persisted across sessions
- ✅ Async requests for 30-50% speed improvement
- ✅ Automatic conversation trimming
- ✅ Graceful fallback if Strands fails

## Future Improvements

- Docker container for easy deployment
- Deploy as a REST API
- Support image and multimodal inputs
- Voice input and output
- Markdown rendering in UI
- Support more AI providers (OpenAI, Anthropic, Together AI)
- Response streaming for faster perceived speed
- Multi-user support with authentication
- Database migration and management tools
- User profiles with personalized memory