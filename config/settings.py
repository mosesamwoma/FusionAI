import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

HF_TOKEN = os.getenv("HF_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

MODELS = [
    {
        "provider": "groq",
        "model": "llama-3.1-8b-instant",
    },
    {
        "provider": "cerebras",
        "model": "llama3.1-8b",
    },
]

FUSION_MODEL = {
    "provider": "groq",
    "model": "llama-3.1-8b-instant",
}
