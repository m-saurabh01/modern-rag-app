# Offline Setup Automation

This directory contains Python scripts to automatically download all dependencies needed for offline installation of the Modern RAG Application on Windows systems.

## Overview

These automation scripts are designed to be run on an **internet-connected machine** to download all necessary components, which can then be transferred to an **offline Windows machine** for installation.

## Scripts

### 1. `config.py`
Central configuration file containing:
- Directory structure definitions
- Software download URLs
- AI model configurations
- Environment settings

### 2. `download_ai_models.py`
Downloads AI/ML models required by the application:
- **Sentence Transformer Models**: all-MiniLM-L6-v2, paraphrase-MiniLM-L3-v2
- **Spacy Language Models**: en_core_web_sm, en_core_web_md
- **NLTK Data**: punkt, stopwords, wordnet, vader_lexicon

**Usage:**
```bash
python download_ai_models.py
python download_ai_models.py list  # List downloaded models
```

### 3. `download_installers.py`
Downloads Windows software installers:
- **Python 3.11.x** (Windows x64 installer)
- **Git for Windows** (latest version)
- **Node.js LTS** (Windows x64 installer)
- **Visual C++ Redistributable** (x64)
- **Tesseract OCR** (Windows installer)

**Usage:**
```bash
python download_installers.py
python download_installers.py list  # List downloaded installers
```

### 4. `download_python_deps.py`
Downloads Python packages and dependencies:
- Creates comprehensive requirements.txt
- Downloads all wheel files and source distributions
- Supports multiple Python versions
- Creates offline installation scripts

**Usage:**
```bash
python download_python_deps.py
python download_python_deps.py list         # List downloaded packages
python download_python_deps.py alternatives # Download for multiple Python versions
```

## Quick Start

### Step 1: Install Prerequisites
```bash
pip install requests sentence-transformers spacy nltk
```

### Step 2: Run All Downloads
```bash
# Download everything
python download_ai_models.py
python download_installers.py
python download_python_deps.py
```

### Step 3: Verify Downloads
```bash
# Check what was downloaded
python download_ai_models.py list
python download_installers.py list
python download_python_deps.py list
```

## Directory Structure After Download

```
offline_setup_automation/
├── downloads/
│   ├── ai_models/
│   │   ├── sentence_transformers/
│   │   │   ├── all-MiniLM-L6-v2/
│   │   │   └── paraphrase-MiniLM-L3-v2/
│   │   ├── spacy/
│   │   │   ├── en_core_web_sm/
│   │   │   └── en_core_web_md/
│   │   └── nltk_data/
│   │       ├── tokenizers/punkt/
│   │       ├── corpora/stopwords/
│   │       ├── corpora/wordnet/
│   │       └── vader_lexicon/
│   ├── installers/
│   │   ├── python-3.11.x-amd64.exe
│   │   ├── Git-x.x.x-64-bit.exe
│   │   ├── node-vx.x.x-x64.msi
│   │   ├── vc_redist.x64.exe
│   │   └── tesseract-x.x.x-win64.exe
│   └── python_deps/
│       ├── requirements.txt
│       ├── install_offline.sh
│       ├── install_offline.bat
│       ├── *.whl (wheel files)
│       └── *.tar.gz (source distributions)
└── cache/ (temporary download cache)
```

## Transfer to Offline Machine

1. **Copy the entire `downloads/` directory** to your offline Windows machine
2. Follow the **Windows Offline Setup Guide** in `docs/WINDOWS_OFFLINE_SETUP_GUIDE.md`
3. Use the provided installation scripts for automated setup

## Advanced Usage

### Custom Configuration
Edit `config.py` to modify:
- Download directories
- Software versions
- AI model selections
- Cache settings

### Selective Downloads
```bash
# Download only specific components
python download_ai_models.py    # Only AI models
python download_installers.py   # Only software installers
python download_python_deps.py  # Only Python packages
```

### Download Size Estimation

| Component | Approximate Size |
|-----------|-----------------|
| AI Models | 1.5 - 2.0 GB |
| Software Installers | 500 - 800 MB |
| Python Dependencies | 2.0 - 3.0 GB |
| **Total** | **4.0 - 5.8 GB** |

## Error Handling

All scripts include comprehensive error handling:
- **Network timeouts**: Automatic retries
- **Missing dependencies**: Clear error messages
- **Disk space**: Size validation before download
- **Partial downloads**: Resume capability where possible

## Troubleshooting

### Common Issues

1. **Network Errors**
   ```bash
   # Retry with verbose output
   python download_ai_models.py --verbose
   ```

2. **Disk Space**
   ```bash
   # Check available space
   df -h  # Linux/macOS
   dir    # Windows
   ```

3. **Python Version Compatibility**
   ```bash
   # Check Python version
   python --version
   # Should be Python 3.8+
   ```

### Support

- Check the main documentation in `docs/`
- Review error messages for specific guidance
- Ensure internet connectivity for all downloads

## Integration with Main Application

These downloaded components integrate with:
- **Main RAG Application**: Uses downloaded AI models and Python packages
- **Windows Setup Guide**: References downloaded installers
- **Docker Deployment**: Can use downloaded packages for container builds

---

**Note**: Run these scripts on a machine with reliable internet connection and sufficient disk space (6+ GB recommended).
