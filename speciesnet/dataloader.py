from tqdm import tqdm
from speciesnet.utils import BBox
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
import pandas as pd
import os
import re

class SpeciesImageDataset(Dataset):
    def __init__(self, df, image_dir, classifier, top4_only=False):
        self.image_dir = image_dir
        self.classifier = classifier
        
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        if top4_only:
            top4_species = (
                df['species'].value_counts()
                .nlargest(4)
                .index.tolist()
            )
            self.df = df[df['species'].isin(top4_species)].reset_index(drop=True)

            # Remap species labels to 0–3
            self.label_map = {species: idx for idx, species in enumerate(top4_species)}
        else:
            self.df = df.reset_index(drop=True)
            self.label_map = None  # Use ground_truth_index

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, row['filename'])
        img = Image.open(img_path).convert("RGB")

        # Get bbox if available
        bboxes = None
        if 'bbox' in row and isinstance(row['bbox'], (list, tuple)) and len(row['bbox']) == 4:
            bbox = BBox(*row['bbox'])
            bboxes = [bbox]

        preprocessed = self.classifier.preprocess(img, bboxes=bboxes)

        # === DEBUG: visualize the preprocessed (cropped + resized) image ===
        if idx == 0:
            import matplotlib.pyplot as plt
            plt.imshow(preprocessed.arr)
            plt.title(f"{row['filename']} | BBox: {row['bbox']}")
            plt.axis("off")
            plt.show()

        # Convert to tensor, normalize
        img_tensor = torch.tensor(preprocessed.arr).permute(2, 0, 1).float() / 255.0
        img_tensor = self.transform(img_tensor)

        # Remap label if needed
        if self.label_map:
            label = self.label_map[row['species']]
        else:
            label = int(row["ground_truth_index"])

        return img_tensor, label