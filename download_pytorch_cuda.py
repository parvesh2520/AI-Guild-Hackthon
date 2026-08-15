"""Download CUDA PyTorch wheel manually with resume support for slow connections."""
import urllib.request
import os
import sys
import time

# PyTorch 2.6.0 with CUDA 12.4 for Windows Python 3.11
URL = "https://download-r2.pytorch.org/whl/cu124/torch-2.6.0%2Bcu124-cp311-cp311-win_amd64.whl"
FILENAME = "torch-2.6.0_cu124-cp311-cp311-win_amd64.whl"

def download_with_progress(url, filename):
    """Download file with progress display and retry."""
    print(f"Downloading {filename}...")
    
    start_byte = 0
    if os.path.exists(filename):
        start_byte = os.path.getsize(filename)
        print(f"  Resuming from {start_byte / 1e6:.1f} MB")
    
    for attempt in range(20):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'})
            if start_byte > 0:
                req.add_header('Range', f'bytes={start_byte}-')
            
            response = urllib.request.urlopen(req, timeout=120)
            
            if response.status == 206 or response.status == 200:
                total = int(response.headers.get('Content-Length', 0)) + start_byte
            else:
                total = 0
                
            mode = 'ab' if start_byte > 0 else 'wb'
            downloaded = start_byte
            chunk_size = 1024 * 512  # 512KB chunks
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
                        print(f"  {downloaded / 1e6:.1f} / {total / 1e6:.1f} MB ({pct:.1f}%)")
                        last_print = time.time()
            
            print(f"  Download complete: {downloaded / 1e6:.1f} MB")
            return True
            
        except Exception as e:
            print(f"  Attempt {attempt+1} failed: {e}")
            if os.path.exists(filename):
                start_byte = os.path.getsize(filename)
            time.sleep(5)
    
    return False

if __name__ == '__main__':
    success = download_with_progress(URL, FILENAME)
    if success:
        print(f"\nNow installing: python -m pip install {FILENAME}")
        os.system(f'python -m pip install "{FILENAME}" torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124')
    else:
        print("Download failed after all retries")
        sys.exit(1)
