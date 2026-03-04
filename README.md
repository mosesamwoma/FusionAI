# FusionAI

A multi-model AI fusion system that queries multiple LLMs simultaneously and synthesizes their responses into one superior answer using [Strands Agents](https://github.com/strands-agents/sdk-python).

## How It Works

1. A user prompt is sent to **Groq**, **Cerebras**, **Gemini**, **SambaNova**, and **Mistral** in parallel
2. All responses are collected and combined
3. A **Strands Agent** acts as an AI judge and fuses them into one final answer

## Models Used

| Provider | Model |
|----------|-------|
| Groq | llama-3.1-8b-instant |
| Cerebras | llama3.1-8b |
| Gemini | gemini-2.0-flash |
| SambaNova | Meta-Llama-3.1-8B-Instruct |
| Mistral | mistral-small-latest |

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
SAMBANOVA_API_KEY=your_sambanova_key
MISTRAL_API_KEY=your_mistral_key
```

## Usage Example

```
You: What is machine learning?

FusionAI: Machine learning is a subset of artificial intelligence that enables
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
- Failed or timed out models are skipped silently

## Future Improvements

- Add a web UI (Streamlit or Flask)
- Add chat history and memory across sessions
- Support more AI providers (OpenAI, Cohere, Cloudflare)
- Allow users to select which models to include in the fusion
- Add response scoring and ranking before fusion
- Deploy as a REST API