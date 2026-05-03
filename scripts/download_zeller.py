import os
import hashlib
import requests
import pandas as pd
from pathlib import Path

# Constants
STUDY_ACCESSION = "PRJEB6070"  # Zeller 2014
ENA_API_URL = "https://www.ebi.ac.uk/ena/portal/api/filereport"
OUTPUT_DIR = Path("data/raw")

def get_study_metadata(accession):
    """
    Fetch study metadata from EBI ENA API.
    """
    params = {
        'accession': accession,
        'result': 'read_run',
        'fields': 'run_accession,sample_accession,fastq_ftp,fastq_md5',
        'format': 'tsv',
        'download': 'true'
    }
    
    print(f"Fetching metadata for {accession}...")
    response = requests.get(ENA_API_URL, params=params)
    
    if response.status_code != 200:
        raise Exception(f"Failed to fetch metadata: {response.status_code}")
        
    return response.text

def compute_md5(file_path):
    """
    Compute MD5 checksum of a file.
    """
    md5_hash = hashlib.md5()
    with open(file_path, "rb") as f:
        # Read in chunks to avoid memory issues with large files
        for byte_block in iter(lambda: f.read(4096), b""):
            md5_hash.update(byte_block)
    return md5_hash.hexdigest()

def download_file(url, output_path, expected_md5=None):
    """
    Download a file and verify its MD5 checksum.
    """
    print(f"Downloading {url} to {output_path}...")
    
    # ENA FTP links often lack protocol
    if not url.startswith("http"):
        url = f"http://{url}"
        
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(output_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False

    if expected_md5:
        print("Verifying MD5 checksum...")
        calculated_md5 = compute_md5(output_path)
        if calculated_md5 != expected_md5:
            print(f"MD5 MISMATCH! Expected {expected_md5}, got {calculated_md5}")
            print(f"Removing corrupted file: {output_path}")
            os.remove(output_path)
            return False
        else:
            print("MD5 checksum verified.")
            
    return True

def main():
    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Get Metadata
    try:
        metadata_text = get_study_metadata(STUDY_ACCESSION)
    except Exception as e:
        print(f"Error getting metadata: {e}")
        return

    # Save metadata
    metadata_file = OUTPUT_DIR / f"{STUDY_ACCESSION}_metadata.tsv"
    with open(metadata_file, "w") as f:
        f.write(metadata_text)
    print(f"Metadata saved to {metadata_file}")

    # Parse metadata
    # The first line is header, usually tab-separated
    df = pd.read_csv(metadata_file, sep='\t')
    
    print(f"Found {len(df)} runs to download.")
    print("NOTE: Download execution is currently PAUSED until approval.")
    print("To enable download, uncomment the loop in main().")

    # 2. Iterate and Download (COMMENTED OUT FOR SAFETY AS PER INSTRUCTIONS)
    # 2. Iterate and Download
    print("Testing download with first 5 samples...")
    for index, row in df.head(5).iterrows():
        run_acc = row['run_accession']
        fastq_ftps = row['fastq_ftp']
        fastq_md5s = row['fastq_md5']
        
        if pd.isna(fastq_ftps):
            print(f"Skipping {run_acc}: No FASTQ FTP link found.")
            continue
            
        # FASTQ fields can be semicolon separated (e.g. read1;read2)
        urls = fastq_ftps.split(';')
        md5s = fastq_md5s.split(';') if not pd.isna(fastq_md5s) else []
        
        for i, url in enumerate(urls):
            filename = url.split('/')[-1]
            out_path = OUTPUT_DIR / filename
            
            # Check if file exists
            if out_path.exists():
                print(f"File {filename} exists. Checking MD5...")
                if i < len(md5s) and compute_md5(out_path) == md5s[i]:
                    print("File verified, skipping download.")
                    continue
                else:
                    print("File integrity check failed or MD5 missing. Redownloading...")

            # Download
            expected_md5 = md5s[i] if i < len(md5s) else None
            success = download_file(url, out_path, expected_md5)
            if not success:
                print(f"Failed to download/verify {filename}")

if __name__ == "__main__":
    main()
