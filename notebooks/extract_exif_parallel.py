import os
import io
import json
import re
import time
import ssl
from PIL import Image, UnidentifiedImageError
from PIL.ExifTags import TAGS
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2 import service_account
from tqdm import tqdm

# === Parameters ===
SOURCE_FILE = "all_image_links_2018_2019_1.json"  # <-- Change this per run
AUTOSAVE_FILE = SOURCE_FILE.replace(".json", "_autosave_partial.json")
OUTPUT_FILE = SOURCE_FILE.replace(".json", "_with_datetime.json")

# === Google Drive Auth ===
SERVICE_ACCOUNT_FILE = '../../credentials.json'
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

credentials = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE, scopes=SCOPES)
service = build('drive', 'v3', credentials=credentials)

# === Helper ===
def extract_drive_id(url):
    match = re.search(r'/d/([a-zA-Z0-9_-]+)', url)
    return match.group(1) if match else None

# === Function to process a single record with retries ===
def process_record(record, max_retries=3):
    file_id = extract_drive_id(record["filepath"])
    if not file_id:
        record["datetime"] = None
        return record

    for attempt in range(1, max_retries + 1):
        try:
            request = service.files().get_media(fileId=file_id)
            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()
            fh.seek(0)

            try:
                image = Image.open(fh)
                exif_data = image._getexif()
                image.close()  # Avoid segfaults
                datetime_str = None
                if exif_data:
                    for tag_id, value in exif_data.items():
                        tag = TAGS.get(tag_id, tag_id)
                        if tag == "DateTimeOriginal":
                            datetime_str = value.replace(":", "-", 2)
                            break
                record["datetime"] = datetime_str
            except UnidentifiedImageError:
                print(f"⚠️ Unreadable image: {record['filename']}")
                record["datetime"] = None
            return record

        except Exception as e:
            if isinstance(e, ssl.SSLError) or "WRONG_VERSION_NUMBER" in str(e):
                print(f"⚠️ SSL issue with {record['filename']}: {e}")
            else:
                print(f"⚠️ General error (attempt {attempt}) processing {record['filename']}: {e}")
            time.sleep(attempt * 2)

    record["datetime"] = None
    return record

# === Load the JSON ===
with open(SOURCE_FILE, "r") as f:
    records = json.load(f)

# === Load autosaved progress if it exists ===
updated_records = []
if os.path.exists(AUTOSAVE_FILE):
    with open(AUTOSAVE_FILE, "r") as f:
        updated_records = json.load(f)
    print(f"🔄 Resuming from autosaved progress: {len(updated_records)} records already processed.")
else:
    print("🆕 Starting fresh...")

# === Sequential Processing with Resume ===
start_index = len(updated_records)
for i in tqdm(range(start_index, len(records)), desc="Extracting EXIF datetime"):
    record = records[i]
    updated = process_record(record)
    updated_records.append(updated)

    if i % 500 == 0 and i > 0:
        with open(AUTOSAVE_FILE, "w") as f:
            json.dump(updated_records, f, indent=2)

# === Final Save ===
with open(OUTPUT_FILE, "w") as f:
    json.dump(updated_records, f, indent=4)

print(f"✅ Done! Saved with datetime info to {OUTPUT_FILE}")
