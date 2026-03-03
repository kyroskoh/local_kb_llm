"""
Entry point for the Local Knowledge Base app.
Loads .env and starts the NiceGUI chat UI.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root
load_dotenv(Path(__file__).resolve().parent / ".env")

from src.app import run_app

if __name__ == "__main__":
    # Unset → default GGUF path; "" or "openai" → OpenAI; path → that GGUF
    run_app(llm_model_path=os.environ.get("LLM_MODEL_PATH"))
