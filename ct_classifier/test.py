import os
import argparse
import yaml
import glob
from tqdm import trange
import wandb
import torch # this imports pytorch
import torch.nn as nn # this contains our loss function
from torch.utils.data import DataLoader # the pytorch dataloader class will take care of all kind of parallelization during training
from torch.optim import SGD # this imports the optimizer
# let's import our own classes and functions!
from util import init_seed
from dataset import CTDataset
from model import CustomResNet18
from datetime import datetime
import pandas as pd
import torch.nn.functional as F
# import seaborn as sns #this is for the confusion matrix
# import matplotlib.pyplot as plt
# from sklearn.metrics import confusion_matrix

# the goal is to test the model on images and see which ones are problematic
# for that, we need filename, prediction, ground truth and confidence. Preferably in a dataframe with those columns.

#should this be train???
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

root_path = '/home/Kathryn/code/ct_classifier/model_states'


# Open the model
cfg = yaml.safe_load(open('/home/Kathryn/code/ct_classifier/configs/exp_resnet18.yaml', 'r')) #r means read
model = CustomResNet18(cfg['num_classes']) #this reads in the number of classes from the config file
state = torch.load(open(f"{root_path}/best.pt", 'rb'), map_location='cpu') #rb is read binary
model.load_state_dict(state['model'])
device = cfg['device']
model.to(device)
model.eval() #evaluation mode, you freeze all the parameters

#initialize empty lists for the three things we're interested in
preds = []
trues = []
confs = []
files = []

# predict on val images
#this is running the model on the val data
dataLoader = create_dataloader(cfg, split='val')
with torch.no_grad():
    for idx, (data, ground_truths, image_names) in enumerate(dataLoader):
        # put data and labels on device, device is cuda (defined in config file)
        data, labels = data.to(device), ground_truths.to(device)
        # forward pass
        prediction = model(data) #for every image, returns a list of all categories
        pred_labels = torch.argmax(prediction, dim=1) #returns only the highest class
        confidence_scores = F.softmax(prediction, dim=1).max(dim=1)[0].tolist() #softmax turns the numbers into numbers from 0 to 1
        pred_labels = pred_labels.tolist()  # Use pred_labels instead of labels
        ground_truths = ground_truths.tolist()
        # file_names = image_names.tolist()
        len_batch = len(pred_labels)
        for idx in range(len_batch): #idx changes every run, goes from 0 to batch size
            pred = pred_labels[idx]
            preds.append(pred) #add the prediction value to the list 

            true = ground_truths[idx]
            trues.append(true)

            conf = confidence_scores[idx]
            confs.append(conf)

            file = image_names[idx]
            files.append(file)

            # print(f"pred : {pred}")
            # print(f"true : {true}")
            # print(f"conf : {conf}")
            # print(f"file : {file}")
            # print("")

#filenames is a list already
#need to get predictions into a list

#create df with file name, prediction, ground truth and confidence values
#a row for every image, and a column for each of those (pred, truth, conf)

# Combine them into a dictionary where keys are column names
data = {
    'file': files,
    'pred': preds,
    'true': trues,
    'conf': confs
}

# Create the DataFrame
df = pd.DataFrame(data)

print(df.head())

df.to_csv(f'{root_path}/results.csv', index=False) #save it as a csv

# #note: DataFrame, list are both classes with functions that work on them



# # MAH adding stuff for lster use
# true_all
# true_super = true_all[mapping_of_finegrsined_to_superclasses]

# pred_all
# mapping_of_finegrsined_to_superclasses ={1:1,2:1,3:1,4:2,5:3}
# pred_super = pred_all[mapping_of_finegrsined_to_superclasses] # mapp from all the classes to just the supercalsses
# confidence_scores_super = F.softmax(pred_super, dim=1).max(dim=1)[0].tolist() #softmax turns the numbers into numbers from 0 to 1




