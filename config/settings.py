import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env", override=True)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SAMBANOVA_API_KEY = os.getenv("SAMBANOVA_API_KEY")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

MODELS = [
    {"provider": "groq", "model": "llama-3.1-8b-instant", "vision": False},
    {"provider": "groq", "model": "llama-3.3-70b-versatile", "vision": False},
    {"provider": "cerebras", "model": "llama3.1-8b", "vision": False},
    {"provider": "gemini", "model": "gemini-2.5-flash", "vision": True},
    {"provider": "sambanova", "model": "Meta-Llama-3.1-8B-Instruct", "vision": False},
    {"provider": "mistral", "model": "mistral-small-latest", "vision": True},
    {"provider": "mistral", "model": "open-mistral-7b", "vision": False},
    {"provider": "nvidia", "model": "meta/llama-3.1-8b-instruct", "vision": False},
    {"provider": "nvidia", "model": "meta/llama-3.1-70b-instruct", "vision": False},
    {"provider": "cohere", "model": "command-a-03-2025", "vision": False},
    {"provider": "openrouter",
        "model": "meta-llama/llama-3.2-11b-vision-instruct:free", "vision": True},
    {"provider": "openrouter", "model": "google/gemma-3-4b-it:free", "vision": False},
    {"provider": "openrouter",
        "model": "meta-llama/llama-3.1-8b-instruct:free", "vision": False},
]

FUSION_MODEL = {
    "provider": "cerebras",
    "model": "llama3.1-8b",
}
