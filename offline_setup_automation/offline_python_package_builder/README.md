# Offline Python Package Builder

This folder contains scripts to download all required Python dependencies for the Modern RAG Application on an internet-connected PC. The downloaded packages can then be transferred to an offline environment for installation.

## Usage

1. **Edit `requirements_offline.txt`**
   - Add or update all required Python packages for your project.

2. **Run the download script**
   - Execute `download_python_deps.py` to download all dependencies and their wheels into the `pip_cache/` and `wheels/` folders.

3. **Transfer the folders**
   - Copy the `pip_cache/` and `wheels/` folders to your offline machine.

4. **Install offline**
   - On the offline machine, use pip with `--no-index --find-links wheels` to install all dependencies.

## Files
- `download_python_deps.py`: Script to download all dependencies and wheels.
- `requirements_offline.txt`: List of all required Python packages.
- `pip_cache/`: Folder where all package sources are downloaded.
- `wheels/`: Folder where all wheel files are downloaded.
