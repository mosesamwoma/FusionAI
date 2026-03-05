# FusionAI

A multi-model AI fusion system that queries multiple LLMs simultaneously and synthesizes their responses into one superior answer using [Strands Agents](https://github.com/strands-agents/sdk-python).

## How It Works

1. A user prompt is sent to all configured LLMs in parallel
2. All responses are collected and combined
3. A **Strands Agent** acts as an AI judge and fuses them into one final answer

## Models Used

| Provider | Model |
|----------|-------|
| Groq | llama-3.1-8b-instant |
| Cerebras | llama3.1-8b |
| Gemini | gemini-2.5-flash |
| SambaNova | Meta-Llama-3.1-8B-Instruct |
| Mistral | mistral-small-latest |
| Nvidia | meta/llama-3.1-8b-instruct |
| Cohere | command-a-03-2025 |
| OpenRouter | openrouter/auto |

## Requirements

- Python 3.11+
- Strands Agents SDK

## Configuration

Create a `.env` file in the project root with the following keys:
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
groq: How can I assist you today?
cerebras: How can I assist you today?
gemini: Hi there! How can I help you today?
sambanova: How can I assist you today?
mistral: Hello! How can I assist you today?
nvidia: How can I assist you today?
cohere: Hello! How can I assist you today?
openrouter: Hi there! How can I help today?
```

If a provider shows an error, check its API key in `.env`.

## Run
```bash
python main.py
```

## Usage Example
```
You: What is machine learning?

FusionAI: Machine learning is a subset of artificial intelligence that enables
systems to learn and improve from experience without being explicitly
programmed. It focuses on developing algorithms that can access data
and use it to learn for themselves...
```

## Status & Limitations

> ⚠️ This project is experimental and not production-ready.

- Response time depends on the slowest model in the pool
- Free tier API keys have rate limits and daily quotas
- Failed or timed out models are skipped silently
- No persistent memory — conversation resets on every run
- No streaming — response appears only after all models finish
- Single user only — no multi-user or session support
- Not optimized for long conversations or large prompts
- Fusion quality depends on how many models respond successfully

## Future Improvements

- Add a web UI (Streamlit or Flask)
- Add chat history and memory across sessions
- Support more AI providers (OpenAI, Anthropic, Together AI)
- Allow users to select which models to include in the fusion
- Add response scoring and ranking before fusion
- Deploy as a REST API
- Support image and multimodal inputs
- Add model performance tracking and analytics
- Add parallel async requests for faster fusion
- Add a Docker container for easy deployment
- Support voice input and output