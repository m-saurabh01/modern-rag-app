"""
Download AI models for offline installation.
This script downloads all required AI models for the Modern RAG Application.
"""
import os
import sys
from pathlib import Path
import nltk
import spacy
from sentence_transformers import SentenceTransformer
from config import (
    AI_MODELS_DIR,
    SENTENCE_TRANSFORMERS_DIR, 
    SPACY_MODELS_DIR,
    NLTK_DATA_DIR,
    SENTENCE_TRANSFORMER_MODELS,
    SPACY_MODELS,
    NLTK_DATASETS
)

def download_sentence_transformers():
    """Download Sentence Transformer models."""
    print("📦 Downloading Sentence Transformer models...")
    SENTENCE_TRANSFORMERS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Set environment variable for cache directory
    os.environ['SENTENCE_TRANSFORMERS_HOME'] = str(SENTENCE_TRANSFORMERS_DIR)
    
    for model_name in SENTENCE_TRANSFORMER_MODELS:
        try:
            print(f"   Downloading {model_name}...")
            model = SentenceTransformer(model_name)
            print(f"   ✅ {model_name} downloaded successfully")
        except Exception as e:
            print(f"   ❌ Failed to download {model_name}: {e}")
    
    print("✅ Sentence Transformer models download completed")

def download_spacy_models():
    """Download Spacy language models."""
    print("📦 Downloading Spacy models...")
    SPACY_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    for model_name in SPACY_MODELS:
        try:
            print(f"   Downloading {model_name}...")
            # Download the model
            os.system(f"python -m spacy download {model_name}")
            print(f"   ✅ {model_name} downloaded successfully")
        except Exception as e:
            print(f"   ❌ Failed to download {model_name}: {e}")
    
    # Copy models to cache directory
    try:
        import shutil
        spacy_path = spacy.__file__.replace('__init__.py', '')
        source_path = os.path.join(spacy_path, 'data')
        
        if os.path.exists(source_path):
            shutil.copytree(source_path, SPACY_MODELS_DIR / 'data', dirs_exist_ok=True)
            print("   ✅ Spacy models copied to cache")
    except Exception as e:
        print(f"   ⚠️  Could not copy spacy models to cache: {e}")
    
    print("✅ Spacy models download completed")

def download_nltk_data():
    """Download NLTK datasets."""
    print("📦 Downloading NLTK datasets...")
    NLTK_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Set NLTK data path
    nltk.data.path.append(str(NLTK_DATA_DIR))
    
    for dataset in NLTK_DATASETS:
        try:
            print(f"   Downloading {dataset}...")
            nltk.download(dataset, download_dir=str(NLTK_DATA_DIR))
            print(f"   ✅ {dataset} downloaded successfully")
        except Exception as e:
            print(f"   ❌ Failed to download {dataset}: {e}")
    
    print("✅ NLTK datasets download completed")

def main():
    """Main function to download all AI models."""
    print("🤖 AI Models Downloader for Modern RAG Application")
    print("=" * 60)
    
    # Create base directory
    AI_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    try:
        # Download all model types
        download_sentence_transformers()
        print()
        download_spacy_models() 
        print()
        download_nltk_data()
        
        print("\n" + "=" * 60)
        print("🎉 All AI models downloaded successfully!")
        print(f"📁 Models saved to: {AI_MODELS_DIR}")
        
        # Show directory sizes
        print("\n📊 Download Summary:")
        if SENTENCE_TRANSFORMERS_DIR.exists():
            print(f"   Sentence Transformers: {SENTENCE_TRANSFORMERS_DIR}")
        if SPACY_MODELS_DIR.exists():
            print(f"   Spacy Models: {SPACY_MODELS_DIR}")  
        if NLTK_DATA_DIR.exists():
            print(f"   NLTK Data: {NLTK_DATA_DIR}")
            
    except KeyboardInterrupt:
        print("\n❌ Download interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during download: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
