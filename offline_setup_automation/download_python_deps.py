"""
Download Python dependencies for offline installation.
This script downloads all Python packages and their dependencies.
"""
import os
import sys
import subprocess
import shutil
from pathlib import Path
from config import PYTHON_DEPS_DIR

# Requirements for the RAG application
REQUIREMENTS_LIST = [
    # Core framework
    "fastapi==0.104.1",
    "uvicorn[standard]==0.24.0",
    "pydantic==2.5.0",
    "python-multipart==0.0.6",
    
    # AI/ML packages
    "sentence-transformers==2.2.2",
    "transformers==4.36.0",
    "torch==2.1.1",
    "torchvision==0.16.1",
    "torchaudio==2.1.1",
    "numpy==1.24.3",
    "scikit-learn==1.3.2",
    
    # NLP packages
    "spacy==3.7.2",
    "nltk==3.8.1",
    "regex==2023.10.3",
    
    # Vector database
    "chromadb==0.4.18",
    "qdrant-client==1.7.0",
    
    # PDF processing
    "PyPDF2==3.0.1",
    "pdfplumber==0.10.3",
    "pytesseract==0.3.10",
    "pillow==10.1.0",
    
    # Text processing
    "langdetect==1.0.9",
    "textstat==0.7.3",
    "beautifulsoup4==4.12.2",
    "lxml==4.9.3",
    
    # HTTP and utilities
    "requests==2.31.0",
    "httpx==0.25.2",
    "aiofiles==23.2.1",
    "python-dotenv==1.0.0",
    
    # Development and testing
    "pytest==7.4.3",
    "pytest-asyncio==0.21.1",
    "black==23.11.0",
    "flake8==6.1.0",
    
    # Optional extras
    "jupyter==1.0.0",
    "matplotlib==3.8.2",
    "seaborn==0.13.0",
    "pandas==2.1.4"
]

def create_requirements_file():
    """Create a requirements.txt file."""
    requirements_path = PYTHON_DEPS_DIR / "requirements.txt"
    
    print("📝 Creating requirements.txt file...")
    with open(requirements_path, 'w') as f:
        for req in REQUIREMENTS_LIST:
            f.write(req + '\n')
    
    print(f"   ✅ Created: {requirements_path}")
    return requirements_path

def download_dependencies():
    """Download Python dependencies using pip download."""
    print("🐍 Python Dependencies Downloader")
    print("=" * 50)
    
    # Create dependencies directory
    PYTHON_DEPS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Create requirements file
    requirements_path = create_requirements_file()
    
    print(f"\n📦 Downloading dependencies to: {PYTHON_DEPS_DIR}")
    
    try:
        # Download packages
        cmd = [
            sys.executable, "-m", "pip", "download",
            "--dest", str(PYTHON_DEPS_DIR),
            "--requirement", str(requirements_path),
            "--no-cache-dir"
        ]
        
        print("🔄 Running pip download command...")
        print(f"Command: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=PYTHON_DEPS_DIR.parent
        )
        
        if result.returncode == 0:
            print("✅ Dependencies downloaded successfully!")
            
            # Count downloaded files
            wheel_files = list(PYTHON_DEPS_DIR.glob("*.whl"))
            tar_files = list(PYTHON_DEPS_DIR.glob("*.tar.gz"))
            total_files = len(wheel_files) + len(tar_files)
            
            print(f"📊 Downloaded {total_files} packages:")
            print(f"   • {len(wheel_files)} wheel files (.whl)")
            print(f"   • {len(tar_files)} source distributions (.tar.gz)")
            
            return True
        else:
            print("❌ Failed to download dependencies!")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Error downloading dependencies: {e}")
        return False

def download_with_alternatives():
    """Download dependencies with alternative Python versions."""
    print("\n🔄 Downloading for multiple Python versions...")
    
    python_versions = ["3.8", "3.9", "3.10", "3.11", "3.12"]
    
    for version in python_versions:
        version_dir = PYTHON_DEPS_DIR / f"python{version}"
        version_dir.mkdir(exist_ok=True)
        
        print(f"\n📦 Downloading for Python {version}...")
        
        cmd = [
            sys.executable, "-m", "pip", "download",
            "--dest", str(version_dir),
            "--python-version", version,
            "--requirement", str(PYTHON_DEPS_DIR / "requirements.txt"),
            "--no-cache-dir",
            "--no-deps"  # Don't download dependencies to avoid conflicts
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                files = list(version_dir.glob("*"))
                print(f"   ✅ Downloaded {len(files)} packages for Python {version}")
            else:
                print(f"   ⚠️  Some packages may not be available for Python {version}")
        except Exception as e:
            print(f"   ❌ Error downloading for Python {version}: {e}")

def create_install_script():
    """Create installation script for offline use."""
    script_content = '''#!/bin/bash
# Offline Python Package Installation Script

echo "🐍 Installing Python packages from offline repository..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Install packages
echo "📥 Installing packages..."
pip install --no-index --find-links . -r requirements.txt

echo "✅ Installation complete!"
echo "To activate the environment, run: source venv/bin/activate"
'''
    
    script_path = PYTHON_DEPS_DIR / "install_offline.sh"
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    # Make executable
    os.chmod(script_path, 0o755)
    
    # Also create Windows batch file
    batch_content = '''@echo off
echo 🐍 Installing Python packages from offline repository...

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo 📦 Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo 🔄 Activating virtual environment...
call venv\\Scripts\\activate.bat

REM Install packages
echo 📥 Installing packages...
pip install --no-index --find-links . -r requirements.txt

echo ✅ Installation complete!
echo To activate the environment, run: venv\\Scripts\\activate.bat
pause
'''
    
    batch_path = PYTHON_DEPS_DIR / "install_offline.bat"
    with open(batch_path, 'w') as f:
        f.write(batch_content)
    
    print(f"📜 Created installation scripts:")
    print(f"   • {script_path}")
    print(f"   • {batch_path}")

def calculate_total_size():
    """Calculate total size of downloaded dependencies."""
    if not PYTHON_DEPS_DIR.exists():
        return 0
    
    total_size = 0
    for filepath in PYTHON_DEPS_DIR.rglob("*"):
        if filepath.is_file():
            total_size += filepath.stat().st_size
    
    return total_size / 1024 / 1024  # Convert to MB

def list_dependencies():
    """List all downloaded dependencies."""
    if not PYTHON_DEPS_DIR.exists():
        print("No dependencies directory found.")
        return
    
    print("\n📋 Downloaded Dependencies:")
    
    # List wheel files
    wheel_files = list(PYTHON_DEPS_DIR.glob("*.whl"))
    if wheel_files:
        print("\n🎯 Wheel files (.whl):")
        for wheel in sorted(wheel_files):
            size_mb = wheel.stat().st_size / 1024 / 1024
            print(f"   {wheel.name} ({size_mb:.1f} MB)")
    
    # List source distributions
    tar_files = list(PYTHON_DEPS_DIR.glob("*.tar.gz"))
    if tar_files:
        print("\n📦 Source distributions (.tar.gz):")
        for tar in sorted(tar_files):
            size_mb = tar.stat().st_size / 1024 / 1024
            print(f"   {tar.name} ({size_mb:.1f} MB)")
    
    total_size = calculate_total_size()
    print(f"\n📊 Total size: {total_size:.1f} MB")

def main():
    """Main function."""
    if len(sys.argv) > 1:
        if sys.argv[1] == "list":
            list_dependencies()
            return
        elif sys.argv[1] == "alternatives":
            download_with_alternatives()
            return
    
    try:
        success = download_dependencies()
        
        if success:
            create_install_script()
            list_dependencies()
        else:
            print("\n❌ Failed to download all dependencies")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n❌ Download interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during download: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
