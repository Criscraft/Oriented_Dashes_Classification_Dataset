import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from scripts.toy_dataset import ToyDataset

# Create the dataset 
dataset = ToyDataset()
dataset.prepare({})

# Get the first 8 images from the dataset
sample_indices = [0,1,3,2,4,7]
images = [dataset[i]['data'] for i in sample_indices]
labels = [dataset[i]['label'] for i in sample_indices]
labels = [dataset.class_names[label] for label in labels]

def tensor_to_pil(image):
    # Convert the tensor to a Pil image
    image = image.numpy().squeeze()
    image = image * 255
    image = image.astype(np.uint8)
    image = transforms.ToPILImage()(image)
    return image

def plot_grid(data, filename='', labels=[], ncol=3):
    # Plot a grid of images
    nrow = np.ceil(len(data) / float(ncol)).astype(int)
    fig, _ = plt.subplots(nrow, ncol, figsize=(ncol*2.5,nrow*2.5))
    if not labels:
        labels = ["" for _ in range(len(data))]
    for i, (image_tensor, label) in enumerate(zip(data, labels)):
        ax = plt.subplot(nrow, ncol, i+1)
        if image_tensor.shape[0] == 1:
            ax.imshow(tensor_to_pil(image_tensor), cmap="gray")
        else:
            ax.imshow(tensor_to_pil(image_tensor))
        ax.set_title(label)
        plt.axis('off')
    fig.tight_layout()
    
    if filename:
        fig.savefig(filename, transparent=False, dpi=80, bbox_inches="tight")

    return fig

# Plot the images
fig = plot_grid(images, filename='../results/toy_examples.pdf', labels=labels)
