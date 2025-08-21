import os
import subprocess

REQUIREMENTS_FILE = "requirements_offline.txt"
PIP_CACHE_DIR = "pip_cache"
WHEELS_DIR = "wheels"

os.makedirs(PIP_CACHE_DIR, exist_ok=True)
os.makedirs(WHEELS_DIR, exist_ok=True)

# Download all packages and dependencies to pip_cache
subprocess.run([
    "pip", "download", "-r", REQUIREMENTS_FILE, "-d", PIP_CACHE_DIR
], check=True)

# Download all wheels for better compatibility
subprocess.run([
    "pip", "wheel", "-r", REQUIREMENTS_FILE, "-w", WHEELS_DIR
], check=True)

print("✅ All dependencies downloaded to pip_cache/ and wheels/")
