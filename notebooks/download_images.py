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
SAVE_DIR = "../datasets/all_species_images"
LOG_PATH = "download_log.csv"
NUM_WORKERS = 4  # Recommended on Mac
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

# === Load previous download log ===
def load_download_log():
    if os.path.exists(LOG_PATH):
        return pd.read_csv(LOG_PATH)
    else:
        return pd.DataFrame(columns=["filename", "status"])

# === File Download Function ===
def download_file(row_data):
    index, row = row_data
    from google.oauth2 import service_account
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseDownload

    credentials = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    drive_service = build('drive', 'v3', credentials=credentials, cache_discovery=False)

    file_id = extract_drive_id(row.get('filepath', ''))
    original_filename = os.path.splitext(row.get('filename', 'unnamed.jpg'))[0]
    site = sanitize(row.get('site', 'unknown'))
    species = sanitize(row.get('species', 'unknown'))

    new_filename = f"{original_filename}_{site}_{species}.jpg"
    file_path = os.path.join(SAVE_DIR, new_filename)

    if os.path.exists(file_path):
        return (new_filename, "✔️ Already exists")
    if not file_id:
        return (new_filename, "⚠️ Invalid ID")

    for attempt in range(MAX_RETRIES):
        try:
            request = drive_service.files().get_media(fileId=file_id)
            with io.FileIO(file_path, 'wb') as fh:
                downloader = MediaIoBaseDownload(fh, request)
                done = False
                while not done:
                    _, done = downloader.next_chunk()
            return (new_filename, "✅ Downloaded")
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(2)
                continue
            return (new_filename, f"⚠️ Failed: {e}")

# === Batch Download Function ===
def download_all_images(df):
    log_df = load_download_log()
    downloaded_set = set(log_df.loc[log_df['status'].str.startswith("✅"), 'filename'])

    rows_to_download = []
    for index, row in df.iterrows():
        original_filename = os.path.splitext(row.get('filename', 'unnamed.jpg'))[0]
        site = sanitize(row.get('site', 'unknown'))
        species = sanitize(row.get('species', 'unknown'))
        new_filename = f"{original_filename}_{site}_{species}.jpg"
        if new_filename not in downloaded_set:
            rows_to_download.append((index, row))

    new_log_entries = []
    with get_context("spawn").Pool(processes=NUM_WORKERS) as pool:
        for result in tqdm(pool.imap_unordered(download_file, rows_to_download), total=len(rows_to_download), desc="Downloading"):
            filename, status = result
            print(f"{status}: {filename}")
            new_log_entries.append({"filename": filename, "status": status})

    if new_log_entries:
        new_df = pd.DataFrame(new_log_entries)
        updated_log = pd.concat([log_df, new_df], ignore_index=True)
        updated_log.drop_duplicates(subset=["filename"], keep="last").to_csv(LOG_PATH, index=False)

# === Entry Point ===
if __name__ == "__main__":
    df = pd.read_csv(CSV_PATH)
    download_all_images(df)