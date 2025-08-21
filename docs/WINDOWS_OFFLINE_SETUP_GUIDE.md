# Windows Offline Setup Guide for Modern RAG Application

## 📋 Overview

This comprehensive guide provides step-by-step instructions for setting up the Modern RAG Application on Windows in a completely offline environment. Follow this guide to prepare all necessary components, dependencies, and files for transfer to an air-gapped Windows system.

## 🎯 Prerequisites

### System Requirements
- **Windows 10/11** (64-bit)
- **RAM**: Minimum 8GB, Recommended 16GB+
- **Storage**: Minimum 10GB free space
- **CPU**: Multi-core processor (Intel i5/AMD Ryzen 5 or better)

### Administrative Access
- Local administrator privileges on target Windows machine
- Ability to install software and modify system PATH

## 📦 Phase 1: Download Required Software

### 1.1 Python Installation
**Download Location**: [python.org](https://www.python.org/downloads/windows/)

```
Download: python-3.11.x-amd64.exe (Latest Python 3.11.x)
File Size: ~30MB
Purpose: Core Python runtime
```

**Installation Notes**:
- ✅ Check "Add Python to PATH"
- ✅ Check "Install for all users"
- ✅ Custom installation with pip included

### 1.2 Git for Windows
**Download Location**: [git-scm.com](https://git-scm.com/download/win)

```
Download: Git-x.xx.x-64-bit.exe
File Size: ~50MB
Purpose: Version control and dependency management
```

**Installation Notes**:
- ✅ Include Git Bash
- ✅ Use Windows Terminal as default
- ✅ Enable symbolic links

### 1.3 Microsoft Visual C++ Redistributable
**Download Location**: [Microsoft Support](https://support.microsoft.com/en-us/help/2977003)

```
Download: vc_redist.x64.exe (Latest version)
File Size: ~25MB
Purpose: Required for Python packages with C extensions
```

### 1.4 Node.js (for Frontend)
**Download Location**: [nodejs.org](https://nodejs.org/en/download/)

```
Download: node-v20.xx.x-x64.msi (LTS version)
File Size: ~30MB
Purpose: Frontend build tools and package management
```


## 🔧 Phase 2: Prepare Python Environment

### 2.1 Create Offline pip Package Cache

On an internet-connected machine, create a pip cache directory:

```powershell
# Create directory structure
mkdir C:\RAG_Offline_Setup
mkdir C:\RAG_Offline_Setup\pip_cache
mkdir C:\RAG_Offline_Setup\wheels
mkdir C:\RAG_Offline_Setup\application
mkdir C:\RAG_Offline_Setup\models
mkdir C:\RAG_Offline_Setup\tools

cd C:\RAG_Offline_Setup
```

### 2.2 Download Python Dependencies

Create a requirements file for offline installation:

**requirements_offline.txt**:
```txt
# Core Framework
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
pydantic-settings==2.1.0

# AI/ML Dependencies
torch==2.1.1+cpu
transformers==4.36.0
sentence-transformers==2.2.2
numpy==1.24.4
scipy==1.11.4
scikit-learn==1.3.2

# Vector Database
chromadb==0.4.18
qdrant-client==1.7.0

# Document Processing
PyPDF2==3.0.1
pdfplumber==0.10.3
pytesseract==0.3.10
Pillow==10.1.0
python-multipart==0.0.6

# Text Processing
spacy==3.7.2
nltk==3.8.1
beautifulsoup4==4.12.2
markdownify==0.11.6

# Database & Storage
sqlalchemy==2.0.23
alembic==1.13.1
redis==5.0.1
aiofiles==23.2.1

# HTTP & Networking
httpx==0.25.2
websockets==12.0
python-socketio==5.10.0

# Utilities
python-dotenv==1.0.0
click==8.1.7
rich==13.7.0
loguru==0.7.2
typer==0.9.0

# Development & Testing
pytest==7.4.3
pytest-asyncio==0.21.1
pytest-cov==4.1.0
black==23.11.0
isort==5.12.0
mypy==1.7.1

# Optional: Jupyter for analysis
jupyter==1.0.0
ipykernel==6.26.0
```

### 2.3 Download pip packages offline

```powershell
# Download all packages with dependencies
pip download -r requirements_offline.txt -d C:\RAG_Offline_Setup\pip_cache

# Download wheel files for better compatibility
pip wheel -r requirements_offline.txt -w C:\RAG_Offline_Setup\wheels
```

### 2.4 Download AI Models

#### Sentence Transformers Models
Create **download_models.py**:
```python
from sentence_transformers import SentenceTransformer
import os

# Set cache directory
os.environ['SENTENCE_TRANSFORMERS_HOME'] = r'C:\RAG_Offline_Setup\models\sentence_transformers'

# Download required models
models = [
    'all-MiniLM-L6-v2',          # Fast, lightweight embeddings
    'all-mpnet-base-v2',         # High quality embeddings
    'multi-qa-MiniLM-L6-cos-v1', # Question-answering optimized
]

for model_name in models:
    print(f"Downloading {model_name}...")
    model = SentenceTransformer(model_name)
    print(f"✅ {model_name} downloaded successfully")

print("All models downloaded!")
```

Run the script:
```powershell
cd C:\RAG_Offline_Setup
python download_models.py
```

## 📁 Phase 3: Prepare Application Files

### 3.1 Clone Application Repository

```powershell
cd C:\RAG_Offline_Setup\application
git clone https://github.com/m-saurabh01/modern-rag-app.git .

# Remove .git directory to save space
Remove-Item -Recurse -Force .git
```

### 3.2 Download Frontend Dependencies

```powershell
cd C:\RAG_Offline_Setup\application\modern_rag_frontend

# Install and cache node modules
npm install

# Create offline cache
npm pack --pack-destination=C:\RAG_Offline_Setup\frontend_cache

# Cache all dependencies
npm cache pack --cache=C:\RAG_Offline_Setup\npm_cache
```

### 3.3 Create Configuration Templates

**config_template.env**:
```bash
# Modern RAG Application Configuration
ENVIRONMENT=production
DEBUG=false

# Database Configuration
DATABASE_URL=sqlite:///./rag_database.db
CHROMA_PERSIST_DIRECTORY=./storage/chromadb
DOCUMENTS_DIRECTORY=./storage/documents
PROCESSED_DIRECTORY=./storage/processed
BACKUP_DIRECTORY=./storage/backups

# AI Model Configuration
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DIMENSION=384
RERANKING_MODEL=sentence-transformers/ms-marco-MiniLM-L-6-v2

# LLM Configuration (Offline)
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama2

# Performance Configuration
MAX_WORKERS=4
CHUNK_SIZE=512
CHUNK_OVERLAP=50
MAX_RESPONSE_TIME=30.0

# Security
SECRET_KEY=your-secret-key-here
CORS_ORIGINS=["http://localhost:3000", "http://127.0.0.1:3000"]

# Logging
LOG_LEVEL=INFO
LOG_FILE=./logs/app.log
```

## 📋 Phase 5: Create Installation Scripts

### 5.1 Windows Installation Script

**install_windows.bat**:
```batch
@echo off
echo ================================================
echo Modern RAG Application - Windows Installation
echo ================================================

echo.
echo Step 1: Installing Python...
if exist python-3.11.*.exe (
    python-3.11.*.exe /quiet InstallAllUsers=1 PrependPath=1 Include_test=0
    echo ✅ Python installed
) else (
    echo ❌ Python installer not found!
    pause
    exit /b 1
)

echo.
echo Step 2: Installing Git...
if exist Git-*.exe (
    Git-*.exe /VERYSILENT /NORESTART
    echo ✅ Git installed
) else (
    echo ❌ Git installer not found!
)

echo.
echo Step 3: Installing Visual C++ Redistributable...
if exist vc_redist.x64.exe (
    vc_redist.x64.exe /quiet /norestart
    echo ✅ VC++ Redistributable installed
) else (
    echo ❌ VC++ Redistributable not found!
)

echo.
echo Step 4: Installing Node.js...
if exist node-*.msi (
    msiexec /i node-*.msi /quiet /norestart
    echo ✅ Node.js installed
) else (
    echo ❌ Node.js installer not found!
)

echo.
echo Step 5: Setting up Python environment...
python -m venv rag_env
call rag_env\Scripts\activate.bat

echo.
echo Step 6: Installing Python packages offline...
pip install --no-index --find-links wheels -r requirements_offline.txt
echo ✅ Python packages installed

echo.
echo Step 7: Setting up models...
xcopy /E /I models %USERPROFILE%\.cache\
echo ✅ AI models configured

echo.
echo Step 8: Configuring application...
copy config_template.env .env
echo ✅ Configuration template created

echo.
echo ================================================
echo Installation completed successfully!
echo ================================================
echo.
echo Next steps:
echo 1. Edit .env file with your specific settings
echo 2. Run: rag_env\Scripts\activate.bat
echo 3. Run: python -m modern_rag_app.main
echo.
pause
```

## ✅ Final Checklist - Pre-Transfer Verification

### Software Installers Checklist
- [ ] **Python 3.11.x** (python-3.11.x-amd64.exe) - ~30MB
- [ ] **Git for Windows** (Git-x.xx.x-64-bit.exe) - ~50MB  
- [ ] **Visual C++ Redistributable** (vc_redist.x64.exe) - ~25MB
- [ ] **Node.js LTS** (node-v20.xx.x-x64.msi) - ~30MB
- [ ] **Tesseract OCR** (tesseract-ocr-w64-setup-v5.3.0.exe) - ~65MB [Optional]
- [ ] **Ollama** (ollama-windows-amd64.exe) - ~15MB [Optional]

### Python Dependencies Checklist
- [ ] **Pip cache directory** (pip_cache/) - All .whl and .tar.gz files
- [ ] **Wheel files** (wheels/) - All built wheel packages
- [ ] **Requirements file** (requirements_offline.txt) - Package list
- [ ] **Total package size**: ~2-3GB

### AI Models Checklist
- [ ] **Sentence Transformers models**:
  - [ ] all-MiniLM-L6-v2 (~90MB)
  - [ ] all-mpnet-base-v2 (~420MB) 
  - [ ] multi-qa-MiniLM-L6-cos-v1 (~90MB)
- [ ] **Spacy models**:
  - [ ] en_core_web_sm (~15MB)
  - [ ] en_core_web_md (~50MB)
- [ ] **NLTK data**:
  - [ ] punkt, stopwords, wordnet, etc. (~50MB)
- [ ] **Total models size**: ~700MB

### Application Files Checklist
- [ ] **Complete source code** (modern_rag_app/) - Application directory
- [ ] **Frontend code** (modern_rag_frontend/) - React application
- [ ] **Node modules cache** (npm_cache/) - Frontend dependencies
- [ ] **Configuration templates** (.env template)
- [ ] **Documentation** (All .md files)

### Scripts and Utilities Checklist
- [ ] **install_windows.bat** - Main installation script
- [ ] **verify_setup.py** - Setup verification
- [ ] **start_application.bat** - Application launcher
- [ ] **stop_application.bat** - Application stopper
- [ ] **config_template.env** - Configuration template

### Final Package Verification
- [ ] **Total package size**: ~6-8GB
- [ ] **Directory structure** follows recommended layout
- [ ] **All file paths** use Windows format (\\ separators)
- [ ] **No internet dependencies** in package
- [ ] **Installation instructions** included
- [ ] **Troubleshooting guide** included

## 🎯 Compression and Transfer

### Creating Final Package
```powershell
# Create compressed archive
cd C:\
Compress-Archive -Path "RAG_Offline_Deployment" -DestinationPath "Modern_RAG_Offline_v1.0.zip" -CompressionLevel Optimal

# Verify archive integrity
Expand-Archive -Path "Modern_RAG_Offline_v1.0.zip" -DestinationPath "C:\Temp\Verify" -Force
```

### Transfer Methods
1. **USB Drive**: Use USB 3.0+ for faster transfer (8GB+ capacity)
2. **Network Transfer**: SCP/SFTP to staging area
3. **Physical Media**: DVD/Blu-ray for permanent archive
4. **Cloud Storage**: Temporary cloud storage for internal transfer

---

**This offline package provides everything needed to deploy the Modern RAG Application on Windows systems without internet connectivity.**
