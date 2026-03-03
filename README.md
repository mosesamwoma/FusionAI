# FusionAI

A multi-model AI fusion system that queries multiple LLMs simultaneously and synthesizes their responses into one superior answer using [Strands Agents](https://github.com/strands-agents/sdk-python).

## How It Works

1. A user prompt is sent to **Groq**, **Cerebras**, **Gemini**, and **Cloudflare AI Gateway** in parallel
2. All responses are collected and combined
3. A **Strands Agent** acts as an AI judge and fuses them into one final answer

## Models Used

| Provider | Model |
|-----------|-------|
| Groq | llama-3.1-8b-instant |
| Cerebras | llama3.1-8b |
| Gemini | gemini-1.5-flash |
| Cloudflare Gateway | openai/gpt-4o-mini |

## Activate Environment
```bash
.venv\Scripts\Activate.ps1
```

## Install Dependencies
```bash
pip install -r requirements.txt
```

## Run
```bash
python main.py
```

## Configuration

Create a `.env` file in the project root with the following keys:
```
GROQ_API_KEY=your_groq_key
CEREBRAS_API_KEY=your_cerebras_key
GEMINI_API_KEY=your_gemini_key
CF_AIG_TOKEN=your_cloudflare_gateway_token
```

## Usage Example
```
Ask: What is machine learning?

FINAL ANSWER:
Machine learning is a subset of artificial intelligence that enables
systems to learn and improve from experience without being explicitly
programmed. It focuses on developing algorithms that can access data
and use it to learn for themselves...
```

## Requirements

- Python 3.11+
- Strands Agents SDK

## Status & Limitations

> ⚠️ This project is experimental and not production-ready.

- Response time depends on the slowest model in the pool
- API costs apply depending on your provider usage
- No persistent memory or chat history yet
- Error handling is basic — one failing model is skipped silently

## Future Improvements

- Add a web UI (Streamlit or Flask)
- Add chat history and memory across sessions
- Support more AI providers (OpenAI, Mistral, Cohere)
- Allow users to select which models to include in the fusion
- Add response scoring and ranking before fusion
- Deploy as a REST API
