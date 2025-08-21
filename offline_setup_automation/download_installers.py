"""
Download software installers for offline installation.
This script downloads all required software installers for Windows.
"""
import os
import sys
import requests
from pathlib import Path
from urllib.parse import urlparse
from config import INSTALLERS_DIR, SOFTWARE_URLS

def download_file(url, filename):
    """Download a file from URL with progress indication."""
    try:
        print(f"   Downloading {filename}...")
        
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        filepath = INSTALLERS_DIR / filename
        
        with open(filepath, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)
                    downloaded += len(chunk)
                    
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"   Progress: {percent:.1f}%", end='\r')
        
        print(f"   ✅ {filename} downloaded ({downloaded / 1024 / 1024:.1f} MB)")
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"   ❌ Failed to download {filename}: {e}")
        return False
    except Exception as e:
        print(f"   ❌ Error downloading {filename}: {e}")
        return False

def get_filename_from_url(url):
    """Extract filename from URL."""
    parsed_url = urlparse(url)
    filename = os.path.basename(parsed_url.path)
    
    # Handle special cases
    if not filename or filename == 'vc_redist.x64.exe':
        if 'vc_redist' in url:
            return 'vc_redist.x64.exe'
        elif 'python' in url:
            return f"python-{url.split('/')[-2]}-amd64.exe"
    
    return filename

def download_installers():
    """Download all software installers."""
    print("💾 Software Installers Downloader")
    print("=" * 50)
    
    # Create installers directory
    INSTALLERS_DIR.mkdir(parents=True, exist_ok=True)
    
    success_count = 0
    total_count = len(SOFTWARE_URLS)
    
    for software_name, url in SOFTWARE_URLS.items():
        print(f"\n📦 Downloading {software_name.upper()}...")
        
        filename = get_filename_from_url(url)
        filepath = INSTALLERS_DIR / filename
        
        # Skip if file already exists
        if filepath.exists():
            file_size = filepath.stat().st_size / 1024 / 1024
            print(f"   ✅ {filename} already exists ({file_size:.1f} MB)")
            success_count += 1
            continue
        
        if download_file(url, filename):
            success_count += 1
    
    print(f"\n" + "=" * 50)
    print(f"📊 Download Summary: {success_count}/{total_count} successful")
    print(f"📁 Installers saved to: {INSTALLERS_DIR}")
    
    if success_count == total_count:
        print("🎉 All installers downloaded successfully!")
    else:
        print("⚠️  Some downloads failed. Check the errors above.")
    
    return success_count == total_count

def list_installers():
    """List all downloaded installers."""
    if not INSTALLERS_DIR.exists():
        print("No installers directory found.")
        return
    
    print("\n📋 Downloaded Installers:")
    total_size = 0
    
    for filepath in INSTALLERS_DIR.iterdir():
        if filepath.is_file():
            size_mb = filepath.stat().st_size / 1024 / 1024
            total_size += size_mb
            print(f"   {filepath.name} ({size_mb:.1f} MB)")
    
    print(f"\n📊 Total size: {total_size:.1f} MB")

def main():
    """Main function."""
    if len(sys.argv) > 1 and sys.argv[1] == "list":
        list_installers()
        return
    
    try:
        success = download_installers()
        list_installers()
        
        if not success:
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n❌ Download interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during download: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
