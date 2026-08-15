"""Download PyTorch wheel manually with resume support for slow connections."""
import urllib.request
import os
import sys
import time

# PyTorch 2.13.0 for cp311 win_amd64 from PyPI
URL = "https://files.pythonhosted.org/packages/cp311/t/torch/torch-2.13.0-cp311-cp311-win_amd64.whl"
FILENAME = "torch-2.13.0-cp311-cp311-win_amd64.whl"

def download_with_progress(url, filename):
    """Download file with progress display and retry."""
    print(f"Downloading {filename}...")
    
    # Check if partial download exists
    start_byte = 0
    if os.path.exists(filename):
        start_byte = os.path.getsize(filename)
        print(f"  Resuming from {start_byte / 1e6:.1f} MB")
    
    for attempt in range(10):
        try:
            req = urllib.request.Request(url)
            if start_byte > 0:
                req.add_header('Range', f'bytes={start_byte}-')
            
            response = urllib.request.urlopen(req, timeout=120)
            
            # Get total size
            total = int(response.headers.get('Content-Length', 0)) + start_byte
            
            mode = 'ab' if start_byte > 0 else 'wb'
            downloaded = start_byte
            chunk_size = 1024 * 256  # 256KB chunks
            last_print = time.time()
            
            with open(filename, mode) as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    if time.time() - last_print > 5:
                        pct = downloaded / total * 100 if total > 0 else 0
                        print(f"  {downloaded / 1e6:.1f} / {total / 1e6:.1f} MB ({pct:.0f}%)")
                        last_print = time.time()
            
            print(f"  Download complete: {downloaded / 1e6:.1f} MB")
            return True
            
        except Exception as e:
            print(f"  Attempt {attempt+1} failed: {e}")
            start_byte = os.path.getsize(filename) if os.path.exists(filename) else 0
            time.sleep(2)
    
    return False

if __name__ == '__main__':
    # First try to get the actual URL from PyPI JSON API
    try:
        import json
        api_url = "https://pypi.org/pypi/torch/2.13.0/json"
        resp = urllib.request.urlopen(api_url, timeout=30)
        data = json.loads(resp.read())
        for f in data['urls']:
            if 'cp311-cp311-win_amd64' in f['filename'] and f['filename'].endswith('.whl'):
                URL = f['url']
                FILENAME = f['filename']
                print(f"Found: {FILENAME}")
                print(f"URL: {URL}")
                print(f"Size: {f['size'] / 1e6:.1f} MB")
                break
    except Exception as e:
        print(f"Could not query PyPI API: {e}, using default URL")
    
    success = download_with_progress(URL, FILENAME)
    if success:
        print(f"\nNow installing: python -m pip install {FILENAME}")
        os.system(f'python -m pip install "{FILENAME}" torchvision pandas scikit-learn')
    else:
        print("Download failed after all retries")
        sys.exit(1)
