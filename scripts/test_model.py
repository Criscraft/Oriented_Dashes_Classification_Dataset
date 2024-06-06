import torch
import numpy as np
from scripts.model import Model
from scripts.toy_dataset import ToyDataset

"""
This script tests the HandpickedModel on the ToyDataset.
The accuracy of the model is 100%, i.e. it classifies all the images correctly.
"""

# Constants
BATCH_SIZE = 64
NO_CUDA = False

# Prepare the cuda device
use_cuda = not NO_CUDA and torch.cuda.is_available()
cuda_args = {'num_workers': 2, 'pin_memory': True} if use_cuda else {}
device = torch.device("cuda") if use_cuda else torch.device("cpu")

# Prepare the model
model = Model()
model = model.to(device)
model.eval()

# Create the dataset 
dataset = ToyDataset()
dataset.prepare({})
dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, **cuda_args)

# Test the model
with torch.no_grad():
    correct = []
    n_images = 0

    for batch in dataloader:
        images = batch['data']
        images = images.to(device)
        labels = batch['label']
        labels = labels.to(device)
        n_images += len(labels)
        
        outputs = model(images)
        pred = (outputs > 0).to(labels.dtype)
        correct_batch = (labels == pred).sum()
        correct.append(correct_batch.item())
        
    correct = np.array(correct)
    accuracy = correct.sum() / n_images
    print(f"Accuracy: {accuracy:.2f}")