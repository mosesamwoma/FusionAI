import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SAMBANOVA_API_KEY = os.getenv("SAMBANOVA_API_KEY")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

MODELS = [
    # Groq
    {"provider": "groq", "model": "llama-3.1-8b-instant"},
    {"provider": "groq", "model": "llama-3.3-70b-versatile"},

    # Cerebras
    {"provider": "cerebras", "model": "llama3.1-8b"},

    # Gemini
    {"provider": "gemini", "model": "gemini-2.5-flash"},

    # SambaNova
    {"provider": "sambanova", "model": "Meta-Llama-3.1-8B-Instruct"},

    # Mistral
    {"provider": "mistral", "model": "mistral-small-latest"},
    {"provider": "mistral", "model": "open-mistral-7b"},

    # Nvidia
    {"provider": "nvidia", "model": "meta/llama-3.1-8b-instruct"},
    {"provider": "nvidia", "model": "meta/llama-3.1-70b-instruct"},

    # Cohere
    {"provider": "cohere", "model": "command-a-03-2025"},

    # OpenRouter
    {"provider": "openrouter", "model": "google/gemma-3-12b-it:free"},
    {"provider": "openrouter", "model": "google/gemma-3-4b-it:free"},
    {"provider": "openrouter", "model": "meta-llama/llama-3.1-8b-instruct:free"},
]

FUSION_MODEL = {
    "provider": "groq",
    "model": "llama-3.3-70b-versatile",
}
