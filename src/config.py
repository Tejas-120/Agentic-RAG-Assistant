
from pathlib import Path
import os

from dotenv import load_dotenv

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load the existing .env file from the project root
load_dotenv(PROJECT_ROOT / ".env")

# Main project paths
PDF_PATH = PROJECT_ROOT / "documents" / "Project_Report.pdf"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"

# Multimodal extraction paths
MULTIMODAL_DIR = PROJECT_ROOT / "data" / "multimodal"
IMAGE_DIR = MULTIMODAL_DIR / "images"
JSONL_PATH = MULTIMODAL_DIR / "extracted_content.jsonl"

# Gemini configuration
CHAT_MODEL = os.getenv(
    "GEMINI_CHAT_MODEL",
    "gemini-3.1-flash-lite",
)

EMBEDDING_MODEL = os.getenv(
    "GEMINI_EMBEDDING_MODEL",
    "gemini-embedding-001",
)
