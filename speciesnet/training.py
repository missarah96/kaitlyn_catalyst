import os
import gc
from tqdm import tqdm
import json
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image
from speciesnet.classifier import SpeciesNetClassifier
from speciesnet.detector import SpeciesNetDetector
from splitting import extract_species_and_site_from_filename, stratified_site_split_from_folder, filter_df_with_detections
from dataloader import SpeciesImageDataset
from model import AugmentedSpeciesNet
from sklearn.metrics import classification_report, accuracy_score
from torch.utils.data import DataLoader, TensorDataset
import wandb

# === Device Selection ===
device = (
    torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cuda") if torch.cuda.is_available()
    else torch.device("cpu")
)

# === Config ===
base_dir = os.path.expanduser("~/Desktop/Kaitlyn_Catalyst/ct_classifier")
csv_path = os.path.join(base_dir, "notebooks", "full_df_filtered.csv")
target_species_txt = os.path.join(base_dir, "target_species_top4.txt")
image_dir = os.path.join(base_dir, "datasets", "all_species_images")
classifier_model_name = os.path.expanduser("~/.cache/kagglehub/models/google/speciesnet/pyTorch/v4.0.1a/1")
num_epochs = 15
batch_size = 16

# === Initialize Weights & Biases ===
wandb.init(
    project="Species-Classification",
    config={
        "epochs": 15,
        "lr": 1e-4,
        "batch_size": 16,
        "model": "SpeciesNet-Top4",
        "label_smoothing": 0.1,
        "weight_decay": 5e-4,
        "dropout": 0.5
    }
)

wandb.define_metric("epoch")
wandb.define_metric("*", step_metric="epoch")

config = wandb.config

# Dynamically rename run based on sweep config
wandb.run.name = f"ls_{config.label_smoothing}-wd_{config.weight_decay}-do_{config.dropout}"

print(f"Using '{device}' device", flush=True)

# === Load Data ===
full_df = pd.read_csv(csv_path)
train_df, val_df = stratified_site_split_from_folder(image_dir, test_size=0.3)

# === Detector ===
detector = SpeciesNetDetector(model_name=classifier_model_name)

# === Filter by detector and cache ===
train_filtered_path = os.path.join(base_dir, "speciesnet/train_filtered.csv")
val_filtered_path = os.path.join(base_dir, "speciesnet/val_filtered.csv")

if os.path.exists(train_filtered_path) and os.path.exists(val_filtered_path):
    print("Using cached filtered CSVs.")
    train_filtered_df = pd.read_csv(train_filtered_path)
    val_filtered_df = pd.read_csv(val_filtered_path)
else:
    print("Running detector to filter train/val...")
    train_filtered_df = filter_df_with_detections(train_df, image_dir, detector)
    val_filtered_df = filter_df_with_detections(val_df, image_dir, detector)
    train_filtered_df.to_csv(train_filtered_path, index=False)
    val_filtered_df.to_csv(val_filtered_path, index=False)
    print("Filtered train/val saved.")

# === Load classifier and wrap model ===
classifier = SpeciesNetClassifier(model_name=classifier_model_name, target_species_txt=target_species_txt)
original_outputs = len(classifier.labels)
target_labels = len(classifier.target_labels)

# Use sweep values
label_smoothing = config.label_smoothing
weight_decay = config.weight_decay
dropout = config.dropout

# Update classifier with dynamic dropout
classifier.model = AugmentedSpeciesNet(
    classifier.model,
    original_outputs,
    target_labels,
    use_extra_head=True,
    dropout=dropout
)
classifier.model = classifier.model.to(device)

# === Loss & Optimizer ===
criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing).to(device)
optimizer = torch.optim.Adam(classifier.model.parameters(), lr=config.lr, weight_decay=weight_decay)

# === DataLoaders ===
train_dataset = SpeciesImageDataset(train_filtered_df, image_dir, classifier, top4_only=True)
val_dataset = SpeciesImageDataset(val_filtered_df, image_dir, classifier, top4_only=True)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

# === Training Loop ===
best_train_acc = 0.0
best_train_epoch = 0
best_train_preds = []
best_train_trues = []

best_val_acc = 0.0
best_val_epoch = 0
best_val_preds = []
best_val_trues = []

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    classifier.model.train()
    train_loss = 0.0
    train_preds, train_trues = [], []

    for x_batch, y_batch in tqdm(train_loader, desc="Training"):
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = classifier.model(x_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(classifier.model.parameters(), max_norm=1.0)
        optimizer.step()
        train_loss += loss.item()

        preds = torch.argmax(outputs, dim=1)
        train_preds.extend(preds.detach().cpu().numpy())
        train_trues.extend(y_batch.detach().cpu().numpy())

        del x_batch, y_batch, outputs, preds

    avg_train_loss = train_loss / len(train_loader)
    train_acc = accuracy_score(train_trues, train_preds)

    if train_acc > best_train_acc:
        best_train_acc = train_acc
        best_train_epoch = epoch + 1
        best_train_preds = train_preds.copy()
        best_train_trues = train_trues.copy()

    classifier.model.eval()
    val_loss = 0.0
    val_preds, val_trues = [], []

    with torch.no_grad():
        for x_batch, y_batch in tqdm(val_loader, desc="Validating"):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            outputs = classifier.model(x_batch)
            loss = criterion(outputs, y_batch)
            val_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            val_preds.extend(preds.detach().cpu().numpy())
            val_trues.extend(y_batch.detach().cpu().numpy())

            del x_batch, y_batch, outputs, preds

    avg_val_loss = val_loss / len(val_loader)
    val_acc = accuracy_score(val_trues, val_preds)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_val_epoch = epoch + 1
        best_val_preds = val_preds.copy()
        best_val_trues = val_trues.copy()

    with open("id_to_species_full.json", "r") as f:
        id_to_species = json.load(f)

    targetlabel_to_shortname = {
        info["target_label"]: info["short_name"].replace(" ", "_")
        for info in id_to_species.values()
    }
    shortnames = [targetlabel_to_shortname[label] for label in classifier.target_labels]

    # === Always compute and log classification reports ===
    train_report = classification_report(
        train_trues,
        train_preds,
        target_names=shortnames,
        labels=list(range(len(shortnames))),
        output_dict=True)

    val_report = classification_report(
        val_trues,
        val_preds,
        target_names=shortnames,
        labels=list(range(len(shortnames))),
        output_dict=True)

    wandb.log({
        "epoch": epoch + 1,
        "train/avg_loss": avg_train_loss,
        "train/accuracy": train_acc,
        "val/avg_loss": avg_val_loss,
        "val/accuracy": val_acc,
    })

    for short_name in shortnames:
        if short_name in train_report:
            wandb.log({
                f"{short_name}/train_precision": train_report[short_name]["precision"],
                f"{short_name}/train_recall": train_report[short_name]["recall"],
                f"{short_name}/train_f1": train_report[short_name]["f1-score"],
            }, step=epoch + 1)

        if short_name in val_report:
            wandb.log({
                f"{short_name}/val_precision": val_report[short_name]["precision"],
                f"{short_name}/val_recall": val_report[short_name]["recall"],
                f"{short_name}/val_f1": val_report[short_name]["f1-score"],
            }, step=epoch + 1)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"Best Train Epoch: {best_train_epoch} | Acc: {best_train_acc:.4f}")
    print(f"Best Val Epoch: {best_val_epoch} | Acc: {best_val_acc:.4f}")

# === Final confusion matrix logging ===
wandb.log({
    "best_train/confusion_matrix": wandb.plot.confusion_matrix(
        probs=None,
        y_true=best_train_trues,
        preds=best_train_preds,
        class_names=shortnames
    ),
    "best_val/confusion_matrix": wandb.plot.confusion_matrix(
        probs=None,
        y_true=best_val_trues,
        preds=best_val_preds,
        class_names=shortnames
    )
})

wandb.finish()