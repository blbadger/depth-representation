# resnet_representations.py

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
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from google.colab import files
from google.colab import drive

drive.mount('/content/gdrive')

data_dir = pathlib.Path('/content/gdrive/My Drive/Inception_generation',  fname='Combined')
image_count = len(list(data_dir.glob('*.png')))

files.upload() # upload files
!unzip dalmatian.zip

# dataset directory specification
data_dir = pathlib.Path('dalmatian',  fname='Combined')

# send model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print (f"Device: {device}")

manualSeed = 999
random.seed(manualSeed)
torch.manual_seed(manualSeed)


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
        image = torchvision.transforms.Resize(size=[299, 299])(image)

        # assign label
        label = os.path.basename(img_path)
        return image, label


class NewResNet(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        """
        Select the output of interest
        """
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        # x = self.model.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.model.fc(x)
        return x


images = ImageDataset(data_dir, image_type='.jpg')

resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=False).to(device)
resnet.eval()

resnet2 = NewResNet(resnet)
resnet2.eval()

for i, image in enumerate(images):
    print (i)
    image = image[0].reshape(1, 3, 299, 299).to(device)
    print (torch.argmax(resnet(image)))
    target_tensor = resnet2(image)
    print (target_tensor.shape)
    break


# Generate a shifted input and compare to the original

target_tensor = target_tensor.detach().to(device)
plt.figure(figsize=(10, 10))
image_width = len(image[0][0])
target_input = image.reshape(3, image_width, image_width).permute(1, 2, 0).cpu().detach().numpy()
plt.axis('off')
plt.imshow(target_input)
plt.show()
plt.close()

modification = torch.randn(1, 3, 299, 299)/18
modification = modification.to(device)
modified_input = image + modification
modified_output = resnet2(modified_input)
print (modified_output.shape)
print (f'L2 distance between original and shifted inputs: {torch.sqrt(torch.sum((image - modified_input)**2))}')
print (f'L2 distance between target and slightly modified image: {torch.sqrt(torch.sum((target_tensor - modified_output)**2))}')

plt.figure(figsize=(10, 10))
image_width = len(modified_input[0][0])
modified_input = modified_input.reshape(3, image_width, image_width).permute(1, 2, 0).cpu().detach().numpy()
plt.axis('off')
plt.imshow(modified_input)
plt.show()
plt.close()

def random_crop(input_image, size):
    """
    Crop an image with a starting x, y coord from a uniform distribution

    Args:
        input_image: torch.tensor object to be cropped
        size: int, size of the desired image (size = length = width)

    Returns:
        input_image_cropped: torch.tensor
        crop_height: starting y coordinate
        crop_width: starting x coordinate
    """

    image_width = len(input_image[0][0])
    image_height = len(input_image[0])
    crop_width = random.randint(0, image_width - size)
    crop_height = random.randint(0, image_width - size)
    input_image_cropped = input_image[:, :, crop_height:crop_height + size, crop_width: crop_width + size]

    return input_image_cropped, crop_height, crop_width


def octave(single_input, target_output, iterations, learning_rates, sigmas, size, pad=False, crop=False):
    """
    Perform an octave (scaled) gradient descent on the input.

    Args;
        single_input: torch.tensor of the input
        target_output: torch.tensor of the desired output category
        iterations: int, the number of iterations desired
        learning_rates: arr[int, int], pair of integers corresponding to start and end learning rates
        sigmas: arr[int, int], pair of integers corresponding to the start and end Gaussian blur sigmas
        size: int, desired dimension of output image (size = length = width)

    kwargs:
        pad: bool, if True then padding is applied at each iteration of the octave
        crop: bool, if True then gradient descent is applied to cropped sections of the input

    Returns:
        single_input: torch.tensor of the transformed input
    """

    start_lr, end_lr = learning_rates
    start_sigma, end_sigma = sigmas
    iterations_arr, input_distances, output_distances = [], [], []
    for i in range(iterations):
        if crop:
            cropped_input, crop_height, crop_width = random_crop(single_input.detach(), size)
        else:
            cropped_input, crop_height, crop_width = random_crop(single_input.detach(), len(single_input[0][0]))
            size = len(single_input[0][0])

        single_input = single_input.detach() # remove the gradient for the input (if present)
        input_grad = layer_gradient(resnet, cropped_input, target_output) # compute input gradient
        single_input[:, :, crop_height:crop_height+size, crop_width:crop_width+size] -= (start_lr*(iterations-i)/iterations + end_lr*i/iterations)* input_grad # gradient descent step
        single_input[:, :, crop_height:crop_height+size, crop_width:crop_width+size] = torchvision.transforms.functional.gaussian_blur(single_input[:, :, crop_height:crop_height+size, crop_width:crop_width+size], 3, sigma=(start_sigma*(iterations-i)/iterations + end_sigma*i/iterations))

        if i % 1000 == 0 and i > 0:
            output = resnet2(single_input).to(device)
            output_distance = torch.sqrt(torch.sum((target_tensor - output)**2))
            print (f'L2 distance between target and generated image: {output_distance}')
            target_input = torch.tensor(single_input).reshape(1, 3, 299, 299).to(device)
            input_distance = torch.sqrt(torch.sum((single_input - image)**2))
            print (f'L2 distance on the input: {input_distance}')
            input_distances.append(float(input_distance))
            output_distances.append(float(output_distance))
            iterations_arr.append(iterations)

    print (iterations_arr)
    print (input_distances)
    print (output_distances)
    return single_input


def generate_singleinput(model, input_tensors, output_tensors, index, count, target_input, random_input=True):
    """
    Generates an input for a given output

    Args:
        input_tensor: torch.Tensor object, minibatch of inputs
        output_tensor: torch.Tensor object, minibatch of outputs
        index: int, target class index to generate
        cout: int, time step

    kwargs: 
        random_input: bool, if True then a scaled random normal distributionis used

    returns:
        None (saves .png image)
    """

    # manualSeed = 999
    # random.seed(manualSeed)
    # torch.manual_seed(manualSeed)

    class_index = index
 
    input_distances = []
    iterations_arr = []
    if random_input:
        single_input = (torch.randn(1, 3, 299, 299))/20 + 0.7 # scaled normal distribution initialization

    else:
        single_input = input_tensors[0]

    for i in range(1):
        iterations = 500
        single_input = single_input.to(device)
        single_input = single_input.reshape(1, 3, 299, 299)
        original_input = torch.clone(single_input).reshape(3, 299, 299).permute(1, 2, 0).cpu().detach().numpy()
        target_output = torch.tensor([class_index], dtype=int)

        single_input = octave(single_input, target_output, iterations, [0.1, 0.1], [2.4, 0.4], 0, pad=False, crop=False)

        output = resnet2(single_input).to(device)
        print (f'L2 distance between target and generated image: {torch.sqrt(torch.sum((target_tensor - output)**2))}')
        target_input = torch.tensor(target_input).reshape(1, 3, 299, 299).to(device)
        input_distance = torch.sqrt(torch.sum((single_input - image)**2))
        print (f'L2 distance on the input: {input_distance}')
        input_distances.append(float(input_distance))
        iterations_arr.append(iterations)

    print (iterations_arr)
    print (input_distances)
    plt.figure(figsize=(10, 10))
    image_width = len(single_input[0][0])
    target_input = single_input.reshape(3, image_width, image_width).permute(1, 2, 0).cpu().detach().numpy()
    plt.axis('off')
    plt.imshow(target_input)
    plt.show()
    plt.close()
    return single_input


def layer_gradient(model, input_tensor, desired_output):
    """
    Compute the gradient of the output (logits) with respect to the input 
    using an L1 metric to maximize the target classification.

    Args:
        model: torch.nn.model
        input_tensor: torch.tensor object corresponding to the input image
        true_output: torch.tensor object of the desired classification label

    Returns:
        gradient: torch.tensor.grad on the input tensor after backpropegation

    """
    input_tensor.requires_grad = True
    output = resnet2(input_tensor)
    loss = 0.15*torch.sum(torch.abs(target_tensor - output)) # target_tensor is the desired activation
    loss.backward()
    gradient = input_tensor.grad

    return gradient


generate_singleinput(resnet2, [], [], 0, 0, image)

