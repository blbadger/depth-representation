# invertible_mlp.py

# import standard libraries
import time
import pathlib
import os
import pandas as pd 
import random

# import third party libraries
import seaborn as sns
import sklearn.decomposition as decomp
import numpy as np 
import torch
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from google.colab import files
from google.colab import drive

# drive.mount('/content/gdrive')

# send model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print (f"Device: {device}")

class ImageDataset(Dataset):
    """
    Creates a dataset from images classified by folder name.  Random
    sampling of images to prevent overfitting
    """

    def __init__(self, img_dir, image_type='.png'):
        self.image_name_ls = list(img_dir.glob('*/*' + image_type))
        self.img_labels = [item.name for item in data_dir.glob('*/*')]
        self.img_dir = img_dir

    def __len__(self):
        return len(self.image_name_ls)

    def __getitem__(self, index):
        # path to image
        img_path = os.path.join(self.image_name_ls[index])
        image = torchvision.io.read_image(img_path, mode=torchvision.io.ImageReadMode.RGB) # convert image to tensor of ints 
        image = image / 255. # convert ints to floats in range [0, 1]
        image = torchvision.transforms.Resize(size=[32, 32])(image)

        # assign label
        label = os.path.basename(img_path)
        return image, label

class MultiLayerPerceptron(nn.Module):

    def __init__(self, input_size, output_size):

        super().__init__()
        input_size = input_size
        hidden_size = input_size
        self.input2hidden = nn.Linear(input_size, hidden_size)
        self.hidden2hidden = nn.Linear(hidden_size, hidden_size)
        self.hidden2hidden2 = nn.Linear(hidden_size, hidden_size)
        self.hidden2output = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        """
        Forward pass through network

        Args:
            input: torch.Tensor object of network input, size [n_letters * length]

        Return: 
            output: torch.Tensor object of size output_size

        """
        input = torch.flatten(input, start_dim=1)
        out = self.input2hidden(input)
        out = self.hidden2hidden(out)
        out = self.hidden2hidden2(out)
        out = self.hidden2output(out)
        return out

transform = transforms.Compose(
    [transforms.ToTensor()])

batch_size = 512
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

def train_model(dataloader, model, optmizer, loss_fn, epochs):
    model.train()
    count = 0
    total_loss = 0
    start = time.time()
    train_array, test_array = [], []

    for e in range(epochs):
        print (f"Epoch {e+1} \n" + '~'*100)
        total_loss = 0
        count = 0

        for pair in trainloader:
            train_x, train_y= pair[0], pair[1]
            trainx = train_x.to(device)
            output = model(trainx)
            loss = loss_fn(output.to(device), train_y.to(device))
            loss = loss.to(device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss
            count += 1

        ave_loss = float(total_loss) / count
        elapsed_time = time.time() - start
        print (f"Average Loss: {ave_loss:.04}")
        test_array.append(test_model(testloader, model))
        start = time.time()
    print (test_array)
    return

def train_on_random(model, optimizer, loss_fn, epochs):
    model.train()
    count = 0
    total_loss = 0
    start = time.time()
    for e in range(epochs):
        total_loss = 0

        for pair in zip(new_trainloader, new_testloader):
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

    return


def test_model(test_dataloader, model):
    model.eval()
    correct, count = 0, 0
    batches = 0
    for batch, (x, y) in enumerate(test_dataloader):
        x = x.to(device)
        predictions = model(x)
        _, predicted = torch.max(predictions.data, 1)
        count += len(y)
        correct += (predicted == y.to(device)).sum().item()
        batches += 1

    print (f'Test accuracy: {correct / count}')
    return correct / count


new_trainset = []
new_testset = []
fraction = 1
for i in range(10000):
    new_trainset.append(torch.randn(1, 3, 32, 32)/4 + 0.5)
    new_testset.append(random.randint(0, 9))

new_trainloader = torch.utils.data.DataLoader(new_trainset, batch_size=batch_size, shuffle=False)
new_testloader = torch.utils.data.DataLoader(new_testset, batch_size=batch_size, shuffle=False)

mlp = MultiLayerPerceptron(32*32*3, 10).to(device)
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(mlp.parameters())

data_dir = '/content/gdrive/My Drive/Overfit_CIFAR/cifar_invertible_mlp'
torch.save(mlp.state_dict(), data_dir)
mlp.load_state_dict(torch.load(data_dir))

files.upload() 
!unzip dalmatian.zip

# dataset directory specification
data_dir = pathlib.Path('dalmatian',  fname='Combined')

def invert_model(mlp, image):
    """
    Note that the mlp and tensors added should be double type to avoid numerical 
    errors inherent in matrix inversion from affecting the output significantly.
    """
    weights1 = mlp.input2hidden.weight
    bias1 = mlp.input2hidden.bias

    weights2 = mlp.hidden2hidden.weight
    bias2 = mlp.hidden2hidden.bias

    weights3 = mlp.hidden2hidden2.weight
    bias3 = mlp.hidden2hidden2.bias

    inverse_weights1 = torch.inverse(weights1)
    inverse_weights2 = torch.inverse(weights2)
    inverse_weights3 = torch.inverse(weights3)

    image = torch.flatten(image)
    output = torch.matmul(weights1, image) + bias1
    output = torch.matmul(weights2, output) + bias2
    output = torch.matmul(weights3, output) + bias3
    model_output = output

    total = 1
    count = 0
    for i in range(total):
        shift = torch.randn(output.shape)/1e7
        shift = shift.double().to(device)
        output = shift + model_output

        reverse = torch.matmul(inverse_weights3, output - bias3)
        reverse = torch.matmul(inverse_weights2, reverse - bias2)
        reverse = torch.matmul(inverse_weights1, reverse - bias1)
        plt.figure(figsize=(10, 10))
        image_width = 32
        modified_input = reverse.reshape(3, image_width, image_width).permute(1, 2, 0).cpu().detach().numpy()
        plt.axis('off')
        plt.imshow(modified_input)
        plt.show()
        plt.close()

    return 

for pair in enumerate(trainloader):
    image = pair[1][0][0]
    break

image = image.to(device).double()
print (image)

priors, posteriors = [], []
mlp.double()
invert_model(mlp, image)
mlp.float()
train_model(trainloader, mlp, optimizer, loss_function, 10)
# train_on_random(mlp, optimizer, loss_function, 10)
mlp.double()

