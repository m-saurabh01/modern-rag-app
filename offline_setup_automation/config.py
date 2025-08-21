"""
Configuration settings for offline setup automation scripts.
"""
import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).parent
DOWNLOADS_DIR = BASE_DIR / "downloads"
CACHE_DIR = BASE_DIR / "cache"

# Create directories if they don't exist
DOWNLOADS_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

# Python Dependencies
PYTHON_CACHE_DIR = CACHE_DIR / "python_packages"
WHEELS_DIR = CACHE_DIR / "wheels"
REQUIREMENTS_FILE = BASE_DIR / "requirements.txt"

# AI Models
AI_MODELS_DIR = CACHE_DIR / "ai_models"
SENTENCE_TRANSFORMERS_DIR = AI_MODELS_DIR / "sentence_transformers"
SPACY_MODELS_DIR = AI_MODELS_DIR / "spacy"
NLTK_DATA_DIR = AI_MODELS_DIR / "nltk_data"

# Software Installers
INSTALLERS_DIR = DOWNLOADS_DIR / "installers"

# URLs and versions
SOFTWARE_URLS = {
    "python": "https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe",
    "git": "https://github.com/git-for-windows/git/releases/download/v2.44.0.windows.1/Git-2.44.0-64-bit.exe",
    "nodejs": "https://nodejs.org/dist/v20.11.1/node-v20.11.1-x64.msi",
    "vcredist": "https://aka.ms/vs/17/release/vc_redist.x64.exe",
    "tesseract": "https://github.com/UB-Mannheim/tesseract/releases/download/v5.3.3.20231005/tesseract-ocr-w64-setup-5.3.3.20231005.exe"
}

# AI Models to download
SENTENCE_TRANSFORMER_MODELS = [
    "all-MiniLM-L6-v2",
    "all-mpnet-base-v2", 
    "multi-qa-MiniLM-L6-cos-v1"
]

SPACY_MODELS = [
    "en_core_web_sm",
    "en_core_web_md"
]

NLTK_DATASETS = [
    "punkt",
    "stopwords", 
    "wordnet",
    "averaged_perceptron_tagger",
    "vader_lexicon",
    "omw-1.4"
]

# Environment variables
os.environ['SENTENCE_TRANSFORMERS_HOME'] = str(SENTENCE_TRANSFORMERS_DIR)
os.environ['NLTK_DATA'] = str(NLTK_DATA_DIR)

print(f"✅ Configuration loaded")
print(f"   Base directory: {BASE_DIR}")
print(f"   Downloads: {DOWNLOADS_DIR}")
print(f"   Cache: {CACHE_DIR}")
