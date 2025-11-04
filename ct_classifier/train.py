'''
    Training script. Here, we load the training and validation datasets (and
    data loaders) and the model and train and validate the model accordingly.

    2022 Benjamin Kellenberger
    2025 Katie Grabowski
'''

import os
import numpy as np
import argparse
import yaml
import glob
from tqdm import trange
from datetime import datetime 
import wandb
import torch 
import copy
import torch.nn as nn  
from torch.utils.data import DataLoader 
from torch.optim import SGD 
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix 
import matplotlib.pyplot as plt
import seaborn as sns
import wandb.sklearn

# let's import our own classes and functions!
from util import init_seed
from dataset import CTDataset
from model import CustomResNet18



def create_dataloader(cfg, split='train'):
    '''
        Loads a dataset according to the provided split and wraps it in a
        PyTorch DataLoader object.
    '''
    dataset_instance = CTDataset(cfg, split)        # create an object instance of our CTDataset class

    dataLoader = DataLoader(
            dataset=dataset_instance,
            batch_size=cfg['batch_size'],
            shuffle=True,
            num_workers=cfg['num_workers']
        )
    return dataLoader

# def log_data_samples(dataLoader):
#     table = wandb.Table(columns=["Image", "Label", "Filename"])
    
#     for i, batch in enumerate(dataLoader):
#         if i >= 10:  # Log only 10 samples
#             break
        
#         images, labels, filenames = batch  

#         table.add_data(wandb.Image(images[0]), labels[0], filenames[0])  # Log first sample

#     wandb.log({"Sample Data": table})

    
def load_model(cfg):
    model_instance = CustomResNet18(cfg['num_classes'])  # create model instance

    # Path to saved checkpoints
    last_checkpoint = 'model_states/last.pt'
    
    if os.path.exists(last_checkpoint):
        try:
            print('Found last.pt checkpoint. Attempting to load...')
            state = torch.load(open(last_checkpoint, 'rb'), map_location=cfg['device'])
            model_instance.load_state_dict(state['model'])
            print('Checkpoint loaded successfully! Resuming training.')
            start_epoch = state.get('epoch', 0)  # 🛠️ optionally get epoch from checkpoint
        except Exception as e:
            print(f"⚠️ WARNING: Failed to load checkpoint due to error: {e}")
            print('Starting fresh model instead.')
            start_epoch = 0
    else:
        print('No checkpoint found. Starting new model.')
        start_epoch = 0

    return model_instance, start_epoch


def save_model(cfg, epoch, model, stats):
    os.makedirs('model_states', exist_ok=True)

    model_path = f'model_states/{epoch}.pt'
    stats_copy = copy.deepcopy(stats)
    stats_copy['model'] = model.state_dict()
    torch.save(stats_copy, open(model_path, 'wb'))

    # Log to wandb as artifact
    wandb.log({"epoch": epoch})
    artifact = wandb.Artifact(
        name=f"model_checkpoint_{epoch}",
        type="model",
        metadata={"epoch": epoch}
    )
    artifact.add_file(model_path)
    wandb.log_artifact(artifact)

    run = wandb.run
    if run is not None:
        run.link_artifact(artifact, "sarah_dsi/wandb-registry-model/best_model")

    # Save config once
    cfpath = 'model_states/configs_used_for_this_run.yaml'
    if not os.path.exists(cfpath):
        with open(cfpath, 'w') as f:
            yaml.dump(cfg, f)
      
    
def setup_optimizer(cfg, model):
    """
        The optimizer is what applies the gradients to the parameters and makes
        the model learn on the dataset.
    """
    optimizer = SGD(model.parameters(),
                    lr=cfg['learning_rate'],
                    weight_decay=cfg['weight_decay'])
    return optimizer


def log_predictions_table(phase, preds, labels):
    """
    Log prediction vs ground truth table for the best epoch.

    Args:
        phase (str): "train_best" or "validation_best"
        preds (list): list of predicted labels
        labels (list): list of true labels
        cfg (dict): configuration dictionary
    """
    table = wandb.Table(columns=["Predicted Label", "True Label"])
    
    for pred, true in zip(preds, labels):
        table.add_data(pred, true)

    wandb.log({f"{phase.capitalize()} Predictions": table})


def plot_confusion_matrix(y_true, y_pred, class_names, title, log_key):
    """
    Plots a confusion matrix and logs it to WandB.

    Args:
        y_true (list): True labels.
        y_pred (list): Predicted labels.
        class_names (list): List of class names (e.g., ["0", "1", ..., "15"]).
        title (str): Title for the plot.
        log_key (str): The key under which the image is logged to WandB.
    
    Returns:
        fig: The matplotlib figure with the plotted confusion matrix.
    """
    num_classes = len(class_names)
    # Create a confusion matrix that always includes all class indices
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    
    # Log the figure to WandB using the provided log_key
    wandb.log({log_key: wandb.Image(fig)})
    return fig
     
    
def train(cfg, dataLoader, model, optimizer, epoch):
    all_preds = []
    all_labels = []
    
    device = cfg['device'] 
    model.to(device)
    model.train()

    criterion = nn.CrossEntropyLoss()
    loss_total, oa_total = 0.0, 0.0  

    progressBar = trange(len(dataLoader))
    for idx, (data, labels, image_names) in enumerate(dataLoader):
        data, labels = data.to(device), labels.to(device)

        # Forward pass
        prediction = model(data)

        # Reset gradients
        optimizer.zero_grad()

        # Compute loss
        loss = criterion(prediction, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Log statistics
        loss_total += loss.item()
        pred_label = torch.argmax(prediction, dim=1)    
        oa = torch.mean((pred_label == labels).float()) 
        oa_total += oa.item()
        
        all_preds.extend(pred_label.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        progressBar.set_description(
            '[Train] Loss: {:.2f}; OA: {:.2f}%'.format(
                loss_total/(idx+1),
                100*oa_total/(idx+1)
            )
        )
        progressBar.update(1)

    progressBar.close()
    loss_total /= len(dataLoader)
    oa_total /= len(dataLoader)
    
    
    # Compute per-class precision, recall, and F1-score
    precision, recall, f1, _ = precision_recall_fscore_support(
    all_labels, all_preds, average=None, labels=list(range(cfg['num_classes'])), zero_division=0
)
    # Log to wandb for each class
    for i, (p, r, f1s) in enumerate(zip(precision, recall, f1)):
        wandb.log({
            f"Train Precision Class {i}": p,
            f"Train Recall Class {i}": r,
            f"Train F1-score Class {i}": f1s,
            f"epoch": epoch
        })

    return loss_total, oa_total, p, r, f1s, all_preds, all_labels
    

def validate(cfg, dataLoader, model, epoch):
    all_preds = []
    all_labels = []
    
    device = cfg['device']
    model.to(device)
    model.eval()
    
    criterion = nn.CrossEntropyLoss()
    loss_total, oa_total = 0.0, 0.0  

    progressBar = trange(len(dataLoader))
    
    with torch.no_grad():
        for idx, (data, labels, image_names) in enumerate(dataLoader):
            data, labels = data.to(device), labels.to(device)

            # Forward pass
            prediction = model(data)

            # Compute loss
            loss = criterion(prediction, labels)

            # Log statistics
            loss_total += loss.item()
            pred_label = torch.argmax(prediction, dim=1)
            oa = torch.mean((pred_label == labels).float())
            oa_total += oa.item()
            
            all_preds.extend(pred_label.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            progressBar.set_description(
                '[Val ] Loss: {:.2f}; OA: {:.2f}%'.format(
                    loss_total/(idx+1),
                    100*oa_total/(idx+1)
                )
            )
            progressBar.update(1)

    progressBar.close()
    loss_total /= len(dataLoader)
    oa_total /= len(dataLoader)
    
    # Compute per-class precision, recall, and F1-score
    precision, recall, f1, _ = precision_recall_fscore_support(
    all_labels, all_preds, average=None, labels=list(range(cfg['num_classes'])), zero_division=0
    )

    # Log to wandb for each class
    for i, (p, r, f1s) in enumerate(zip(precision, recall, f1)):
        wandb.log({
            f"Valid Precision Class {i}": p,
            f"Valid Recall Class {i}": r,
            f"Valid F1-score Class {i}": f1s,
            f"epoch": epoch
        })
    
    return loss_total, oa_total, p, r, f1s, all_preds, all_labels


def parse_args():
    parser = argparse.ArgumentParser(description='Train deep learning model.')
    parser.add_argument('--config', help='Path to config file', default='configs/exp_resnet18.yaml')
    parser.add_argument('--batch_size', type=int, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, help='Learning rate for training')
    parser.add_argument('--num_epochs', type=int, help='Number of epochs to train')
    parser.add_argument('--weight_decay', type=float, help='Weight decay for training')
    return parser.parse_args()


def main():
    args = parse_args()

    # load config
    print(f'Using config "{args.config}"')
    cfg = yaml.safe_load(open(args.config, 'r'))
    
    # Override config with sweep arguments (if provided)
    if args.batch_size:
        cfg['batch_size'] = args.batch_size
    if args.learning_rate:
        cfg['learning_rate'] = args.learning_rate
    if args.num_epochs:
        cfg['num_epochs'] = args.num_epochs
    if args.weight_decay:
        cfg['weight_decay'] = args.weight_decay
        
    wandb.login()

    wandb.init(
    project="cv4ecology",
    entity="catalyst_dsi",
    config=cfg
    ) # set the wandb project where this run will be logged
    
    wandb.define_metric("epoch")
    wandb.define_metric("*", step_metric="epoch")
    
    # init random number generator seed (set at the start)
    init_seed(cfg.get('seed', None))
    
    device_str = cfg['device']
    # If the device string starts with "cuda", check availability
    if device_str.startswith('cuda'):
        if not torch.cuda.is_available():
            print(f'WARNING: device set to "{device_str}" but CUDA is not available; falling back to CPU...')
            device_str = 'cpu'
    # If the device string starts with "mps", check availability (for Apple Silicon)
    elif device_str.startswith('mps'):
        if not torch.backends.mps.is_available():
            print(f'WARNING: device set to "{device_str}" but MPS is not available; falling back to CPU...')
            device_str = 'cpu'
    try:
        device = torch.device(device_str)
    except Exception as e:
        print(f'WARNING: {device_str} is not a valid device; falling back to CPU...')
        device = torch.device('cpu')
        device_str = 'cpu'
    cfg['device'] = device_str  # update config if needed
    
    # initialize data loaders for training and validation set
    dl_train = create_dataloader(cfg, split='train')
    dl_val = create_dataloader(cfg, split='val')
    
    # initialize model
    model, current_epoch = load_model(cfg)
    
    # Manually configure watching with logging disabled
    wandb.watch(model, log=None)
    
    # set up model optimizer
    optim = setup_optimizer(cfg, model)

    # Early stopping parameters
    patience = cfg.get('patience', 10)  # Number of epochs to wait for improvement, sets to 10 if not defined in config file
    print(f"Starting training with a patience value of {patience}") #useful info when running your model
    best_loss_val = float('inf')  # Best validation loss encountered
    epochs_without_improvement = 0  # Counter for patience
        
    # Track best epoch predictions
    best_train_preds, best_train_labels = None, None
    best_val_preds, best_val_labels = None, None
    best_epoch = None
        
    # we have everything now: data loaders, model, optimizer; let's do the epochs!
    numEpochs = cfg['num_epochs']
    while current_epoch < numEpochs:
        current_epoch += 1
        print(f'Epoch {current_epoch}/{numEpochs}')

        loss_train, oa_train, p_train, r_train, f1s_train, train_preds, train_labels = train(cfg, dl_train, model, optim, current_epoch)
        
        loss_val, oa_val, p_valid, r_valid, f1s_valid, val_preds, val_labels = validate(cfg, dl_val, model, current_epoch)

        # combine stats and save
        stats = {
            'epoch': current_epoch,
            'Train Loss': loss_train,
            'Valid Loss': loss_val,
            'Train Overall Accuracy': oa_train,
            'Valid Overall Accuracy': oa_val,
        }

        # this is checkpoint saving, this saves all models
        save_model(cfg, current_epoch, model, stats)
        #cfg: config
        save_model(cfg, 'last', model, stats) #save last model

        if loss_val < best_loss_val:
            best_loss_val = loss_val
            epochs_without_improvement = 0
            save_model(cfg, 'best', model, stats)
            print(f"Best model!!!! saving model at epoch {current_epoch}")
            
            # Save best preds and labels
            best_train_preds = train_preds
            best_train_labels = train_labels
            best_val_preds = val_preds
            best_val_labels = val_labels
            best_epoch = current_epoch

        else:
            epochs_without_improvement += 1
            print(f"No improvement in validation loss for {epochs_without_improvement} epoch(s).")

        if epochs_without_improvement >= patience:
            print(f"Early stopping triggered after {patience} epochs without improvement.")
            break

        wandb.log(stats)
        
    print("Unique best_train_preds:", np.unique(best_train_preds))
    print("Unique best_train_labels:", np.unique(best_train_labels))
    print("Unique best_val_preds:", np.unique(best_val_preds))
    print("Unique best_val_labels:", np.unique(best_val_labels))
    
    # Plot confusion matrices for best epoch
    if best_train_preds is not None and best_val_preds is not None:
        num_classes = cfg['num_classes']  
        class_names = [str(i) for i in range(num_classes)]
        
        fig_train = plot_confusion_matrix(
            best_train_labels, 
            best_train_preds, 
            class_names,
            f"Best Train Confusion Matrix (Epoch {best_epoch})",
            log_key=f"Best Train Confusion Matrix (Epoch {best_epoch})"
        )

        fig_val = plot_confusion_matrix(
            best_val_labels, 
            best_val_preds, 
            class_names,
            f"Best Validation Confusion Matrix (Epoch {best_epoch})",
            log_key=f"Best Validation Confusion Matrix (Epoch {best_epoch})"
        )
      
        # Log predictions table for best model
        log_predictions_table("train_best", best_train_preds, best_train_labels)
        log_predictions_table("validation_best", best_val_preds, best_val_labels)

    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    os.rename('model_states', f'model_states-{timestamp}')
    wandb.finish()
        

if __name__ == '__main__':
    # This block only gets executed if you call the "train.py" script directly
    # (i.e., "python ct_classifier/train.py").
    main()