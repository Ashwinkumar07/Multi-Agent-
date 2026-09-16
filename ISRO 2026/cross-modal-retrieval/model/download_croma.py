import os
import urllib.request
import sys
import time

def download_file(url, filepath):
    print(f"Downloading {url} to {filepath}...")
    start_time = time.time()
    
    def report_progress(block_num, block_size, total_size):
        if total_size > 0:
            downloaded = block_num * block_size
            percent = min(100, (downloaded / total_size) * 100)
            elapsed = time.time() - start_time
            speed = downloaded / (elapsed + 1e-6) / 1024 / 1024 # MB/s
            sys.stdout.write(f"\rProgress: {percent:.1f}% | Downloaded: {downloaded/1024/1024:.1f}MB / {total_size/1024/1024:.1f}MB | Speed: {speed:.2f} MB/s")
            sys.stdout.flush()

    try:
        urllib.request.urlretrieve(url, filepath, reporthook=report_progress)
        print(f"\nSuccessfully downloaded to {filepath}")
    except Exception as e:
        print(f"\nError downloading {url}: {e}")
        raise e

def main():
    os.makedirs("model", exist_ok=True)
    
    # Download use_croma.py
    use_croma_url = "https://raw.githubusercontent.com/antofuller/CROMA/main/use_croma.py"
    use_croma_path = "model/use_croma.py"
    download_file(use_croma_url, use_croma_path)
    
    # Download CROMA_base.pt weights
    croma_weights_url = "https://huggingface.co/antofuller/CROMA/resolve/main/CROMA_base.pt"
    croma_weights_path = "model/CROMA_base.pt"
    
    try:
        download_file(croma_weights_url, croma_weights_path)
    except Exception as e:
        print("\n" + "="*50)
        print("CRITICAL: Failed to download CROMA_base.pt weights.")
        print("If the download failed due to network constraints, please check internet connectivity or proxy settings.")
        print("="*50)
        sys.exit(1)

if __name__ == "__main__":
    main()
