import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
CLOUDFLARE_API_KEY = os.getenv("CLOUDFLARE_API_KEY")
SAMBANOVA_API_KEY = os.getenv("SAMBANOVA_API_KEY")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

MODELS = [
    {"provider": "groq", "model": "llama-3.1-8b-instant"},
    {"provider": "cerebras", "model": "llama3.1-8b"},
    {"provider": "gemini", "model": "gemini-1.5-flash"},
    {"provider": "cloudflare", "model": "claude-3-5-sonnet"},
    {"provider": "sambanova", "model": "llama3.1-8b"},
    {"provider": "mistral", "model": "mistral-small-latest"},
]

FUSION_MODEL = {
    "provider": "groq",
    "model": "llama-3.1-8b-instant",
}
