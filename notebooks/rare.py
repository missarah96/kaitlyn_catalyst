import os
import io
import re
import time
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, get_context
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# === Constants ===
CSV_PATH = "full_df_filtered.csv"
SERVICE_ACCOUNT_FILE = '../../credentials.json'
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
SAVE_DIR = "../datasets/rare_species_images"
NUM_WORKERS = 4
MAX_RETRIES = 3

# === Prepare directory ===
os.makedirs(SAVE_DIR, exist_ok=True)

# === Utility to extract Drive file ID ===
def extract_drive_id(url):
    try:
        return url.split('/d/')[1].split('/')[0]
    except Exception:
        return None

# === Utility to sanitize filenames ===
def sanitize(text):
    return re.sub(r'\W+', '', str(text)).strip().lower() or "unknown"

# === File Download Function ===
def download_file(row_data):
    index, row = row_data
    from google.oauth2 import service_account
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseDownload

    credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    drive_service = build('drive', 'v3', credentials=credentials, cache_discovery=False)

    file_id = extract_drive_id(row.get('filepath', ''))
    original_filename = os.path.splitext(row.get('filename', 'unnamed.jpg'))[0]
    site = sanitize(row.get('site', 'unknown'))
    species = sanitize(row.get('species', 'unknown'))

    new_filename = f"{original_filename}_{site}_{species}.jpg"
    file_path = os.path.join(SAVE_DIR, new_filename)

    if os.path.exists(file_path):
        return f"✔️ Already exists: {new_filename}"
    if not file_id:
        return f"⚠️ Invalid ID for {new_filename}"

    for attempt in range(MAX_RETRIES):
        try:
            request = drive_service.files().get_media(fileId=file_id)
            with io.FileIO(file_path, 'wb') as fh:
                downloader = MediaIoBaseDownload(fh, request)
                done = False
                while not done:
                    _, done = downloader.next_chunk()
            return f"✅ Downloaded: {new_filename}"
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(2)
                continue
            return f"⚠️ Failed to download {new_filename} (ID: {file_id}): {e}"

# === Batch Download ===
def download_all_images(df):
    rows = list(df.iterrows())
    with get_context("spawn").Pool(processes=NUM_WORKERS) as pool:
        for result in tqdm(pool.imap_unordered(download_file, rows), total=len(rows), desc="Downloading"):
            print(result)

# === Entry Point ===
if __name__ == "__main__":
    df = pd.read_csv(CSV_PATH)

    # ✅ Filter to rare species (<1000 images)
    species_counts = df["species"].value_counts()
    rare_species = species_counts[species_counts < 1000].index
    rare_df = df[df["species"].isin(rare_species)].copy()

    print(f"📦 Found {len(rare_df)} images across {len(rare_species)} rare species")
    download_all_images(rare_df)