import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

HF_TOKEN = os.getenv("HF_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
CF_AIG_TOKEN = os.getenv("CF_AIG_TOKEN")

MODELS = [
    {"provider": "groq", "model": "llama-3.1-8b-instant"},
    {"provider": "cerebras", "model": "llama3.1-8b"},
    {"provider": "gemini", "model": "gemini-1.5-flash"},
    {"provider": "cloudflare", "model": "openai/gpt-4o-mini"},
]

FUSION_MODEL = {
    "provider": "groq",
    "model": "llama-3.1-8b-instant",
}
