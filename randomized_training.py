# import standard libraries
import time
import pathlib
import os
import pandas as pd 
import random

# import third party libraries
import numpy as np 
import torch
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader, Dataset
import torchvision
import matplotlib.pyplot as plt  
import torchvision.transforms as transforms
from google.colab import files
from google.colab import drive
drive.mount('/content/gdrive')

# send model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print (f"Device: {device}")

class ImageDataset(Dataset):
    """
    Creates a dataset from images classified by folder name.  Random
    sampling of images to prevent overfitting
    """

    def __init__(self, img_dir, transform=None, target_transform=None, image_type='.png'):
        # specify image labels by folder name 
        self.img_labels = [item.name for item in data_dir.glob('*')]

        # construct image name list: randomly sample 400 images for each epoch
        images = list(img_dir.glob('*/*' + image_type))
        random.shuffle(images)
        self.image_name_ls = images[:800]

        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.image_name_ls)

    def __getitem__(self, index):
        # path to image
        img_path = os.path.join(self.image_name_ls[index])
        image = torchvision.io.read_image(img_path) # convert image to tensor of ints , torchvision.io.ImageReadMode.GRAY
        image = image / 255. # convert ints to floats in range [0, 1]
        image = torchvision.transforms.Resize(size=[299, 299])(image) 

        # assign label to be a tensor based on the parent folder name
        label = os.path.basename(os.path.dirname(self.image_name_ls[index]))

        # convert image label to tensor
        label_tens = torch.tensor(self.img_labels.index(label))
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label_tens



def loss_gradient(model, input_tensor, true_output, output_dim):
    """
     Computes the gradient of the input wrt. the objective function

     Args:
        input: torch.Tensor() object of input
        model: Transformer() class object, trained neural network

     Returns:
        gradientxinput: arr[float] of input attributions

    """

    # change output to float
    true_output = true_output.reshape(1)
    input_tensor.requires_grad = True
    output = model.forward(input_tensor)
    loss = loss_fn(output, true_output)

    # backpropegate output gradient to input
    loss.backward(retain_graph=True)
    gradient = input_tensor.grad
    return gradient


def train_model(model, loss_fn, epochs):
    """
    Train the chosen model on the dataset `trainloader` and the
    corresponding classes `trainloader_y`. 

    Args:
        model: nn.Module() object
        loss_fn: torch.nn objective function
        epochs: int, number of training epochs desired

    Accesses: 
        trainloader:torch.utils.data.DataLoader object, training x
        testloader: torch.utils.data.DataLoader object, training y
        optimizer: torch.optim object

    Returns:
        None (modifies model in-place)
    """
    model.train()
    count = 0
    total_loss = 0
    start = time.time()
    for e in range(epochs):
        total_loss = 0

        for pair in zip(trainloader, trainloader_y):
            train_x, train_y = pair[0], pair[1]
            trainx = train_x.to(device)
            output = model(trainx)
            loss = loss_fn(output.to(device), train_y.to(device))
            loss = loss.to(device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss

        elapsed_time = time.time() - start
        start = time.time()
        print (f'Epoch {e} completed in {elapsed_time}')
        print (f'Total loss: {total_loss}')

        data_dir = '/content/gdrive/My Drive/randomized_resnet50_imagenet/randomized_model.pth'
        torch.save(model.state_dict(), data_dir)

    return

def test_model(test_dataloader, model):
    """
    Save the model of interest

    Args:
        test_dataloader: pairs of x, y
    """
    model.eval()
    correct, count = 0, 0
    with torch.no_grad():
        for pair in zip(trainloader, trainloader_y):
            train_x, train_y = pair[0], pair[1]
            trainx = train_x.to(device)
            output = model(trainx)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss

    print (f'Accuracy: {correct / count}')
    return correct / count


batch_size = 128

new_trainset = []
new_trainset_y = []
fraction = 1
for i in range(10000):
    new_trainset.append(torch.randn(3, 224, 224)/4 + 0.5)
    new_trainset_y.append(random.randint(0, 99))

target_input = new_trainset[0].reshape(3, 224, 224).permute(1, 2, 0).cpu().detach().numpy()
plt.figure(figsize=(15, 10))
plt.axis('off')
plt.imshow(target_input, alpha=1)
plt.tight_layout()
plt.show()
plt.close()

trainloader = torch.utils.data.DataLoader(new_trainset, batch_size=batch_size, shuffle=False)
trainloader_y = torch.utils.data.DataLoader(new_trainset_y, batch_size=batch_size, shuffle=False)

torch.cuda.empty_cache()
epochs = 90
loss_fn = nn.CrossEntropyLoss()
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=False)
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
train_model(model, loss_fn, epochs)

