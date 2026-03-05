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
    {"provider": "groq", "model": "llama-3.1-8b-instant"},
    {"provider": "cerebras", "model": "llama3.1-8b"},
    {"provider": "gemini", "model": "gemini-2.5-flash"},
    {"provider": "sambanova", "model": "Meta-Llama-3.1-8B-Instruct"},
    {"provider": "mistral", "model": "mistral-small-latest"},
    {"provider": "nvidia", "model": "meta/llama-3.1-8b-instruct"},
    {"provider": "cohere", "model": "command-a-03-2025"},
    {"provider": "openrouter", "model": "openrouter/auto"},
]

FUSION_MODEL = {
    "provider": "groq",
    "model": "llama-3.1-8b-instant",
}
