# FusionAI

A multi-model AI fusion system that queries multiple LLMs simultaneously and synthesizes their responses into one superior answer using [Strands Agents](https://github.com/strands-agents/sdk-python).

## How It Works

1. A user prompt is sent to all configured LLMs via Strands Agent tools
2. Each model is registered as a `@tool` that the Strands Agent calls
3. All responses are collected and passed to the Fusion Engine
4. The Fusion Engine synthesizes them into one superior final answer

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
| OpenRouter | gemma-3-12b-it:free, gemma-3-4b-it:free |

## Requirements

- Python 3.11+
- Strands Agents SDK

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
```

If a provider shows an error, check its API key in `.env`.

## Run
```bash
python main.py
```

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

- Response time depends on the slowest model in the pool
- API rate limits and daily quotas may affect availability
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
- Deploy as a REST API
- Support image and multimodal inputs
- Add model performance tracking and analytics
- Add parallel async requests for faster fusion
- Add a Docker container for easy deployment
- Support voice input and output