import pandas as pd
import requests
import os

def download_moleculenet_csv(url, name):
    print(f"Downloading {name}...")
    response = requests.get(url)
    with open(f"data/raw/{name}.csv", "wb") as f:
        f.write(response.content)
    print(f"Saved {name}.csv")

if __name__ == "__main__":
    os.makedirs("data/raw", exist_ok=True)
    
    # ClinTox
    clintox_url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/clintox.csv.gz"
    # Note: Requests can handle .gz if it's served with correct headers, 
    # but we might need to decompress manually if it fails.
    
    # ESOL (Delaney)
    esol_url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv"
    
    download_moleculenet_csv(esol_url, "ESOL")
    
    # For ClinTox, let's use a simpler source if available or just try it
    download_moleculenet_csv("https://raw.githubusercontent.com/deepchem/deepchem/master/datasets/clintox.csv.gz", "ClinTox_gz")
    
    # Decompress ClinTox if needed
    try:
        import gzip
        with gzip.open("data/raw/ClinTox_gz.csv", 'rb') as f_in:
            with open("data/raw/ClinTox.csv", 'wb') as f_out:
                f_out.write(f_in.read())
        print("Decompressed ClinTox")
    except Exception as e:
        print(f"ClinTox decompression failed: {e}")
