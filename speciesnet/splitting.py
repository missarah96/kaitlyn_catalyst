from tqdm import tqdm
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import re

def extract_species_and_site_from_filename(filename):
    base = os.path.splitext(filename)[0]
    parts = base.split("_")
    site_pattern = re.compile(r"^[a-zA-Z]\d{2}$")
    for i, part in enumerate(parts):
        if site_pattern.match(part):
            site = part.lower()
            species_parts = parts[i+1:]
            species = "_".join(species_parts).lower()
            return species, site
    return "unknown", "unknown"

def stratified_group_train_test_split(df, stratify_col, group_col, test_size=0.3, random_state=42):
    groups = df[group_col].unique()
    test_groups = []
    best_diff = float("inf")

    # Try multiple random seeds to find a good stratified split
    for seed in range(1000, 1100):
        train_groups, val_groups = train_test_split(groups, test_size=test_size, random_state=seed)
        train_df = df[df[group_col].isin(train_groups)]
        val_df = df[df[group_col].isin(val_groups)]

        train_dist = train_df[stratify_col].value_counts(normalize=True)
        val_dist = val_df[stratify_col].value_counts(normalize=True)
        common_species = train_dist.index.intersection(val_dist.index)
        diff = (train_dist[common_species] - val_dist[common_species]).abs().mean()

        if diff < best_diff:
            best_diff = diff
            best_split = (train_df, val_df)

    return best_split

def stratified_site_split_from_folder(image_dir, test_size=0.3, random_state=42):
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    data = []

    for fname in image_files:
        species, site = extract_species_and_site_from_filename(fname)
        if species != "unknown" and site != "unknown":
            data.append({"filename": fname, "species": species, "site": site})

    df = pd.DataFrame(data)
    return stratified_group_train_test_split(df, stratify_col="species", group_col="site", test_size=test_size, random_state=random_state)


def filter_df_with_detections(df, image_dir, detector):
    records = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Filtering with detector"):
        file_path = os.path.join(image_dir, row["filename"])
        try:
            img = Image.open(file_path).convert("RGB")
            preprocessed = detector.preprocess(img)
            result = detector.predict(filepath=file_path, img=preprocessed)
            detections = result.get("detections", [])

            if detections:
                # Extract the first bounding box
                bbox = detections[0]["bbox"]  # format: [xmin, ymin, width, height]
                row_data = row.to_dict()
                row_data["bbox"] = bbox
                records.append(row_data)

        except Exception as e:
            print(f"Skipping {row.get('filename', 'unknown')}: {e}")

    return pd.DataFrame(records)