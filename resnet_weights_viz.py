import torch
import matplotlib.pyplot as plt
from google.colab import files
from google.colab import drive
drive.mount('/content/gdrive')

model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=False)
model = model.to(device)

model.load_state_dict(torch.load('/content/gdrive/My Drive/randomized_resnet50_imagenet/randomized_model.pth', map_location=torch.device('cpu')))

with torch.no_grad():
    image_batch = []
    print (model.layer1[1].conv2.weight.shape)
    for filter in model.layer4[1].conv2.weight[0]:
        print (filter.shape)
        print (torch.min(filter), torch.max(filter))
        filter = (filter - torch.min(filter)) / (torch.max(filter) - torch.min(filter))
        filter = filter.permute(1, 2, 0).cpu().detach().numpy()
        image_batch.append(filter)

def show_batch(input_batch, count=0, grayscale=False):
	"""
	Show a batch of images with gradientxinputs superimposed

	Args:
		input_batch: arr[torch.Tensor] of input images
		output_batch: arr[torch.Tensor] of classification labels
		gradxinput_batch: arr[torch.Tensor] of attributions per input image
	kwargs:
		individuals: Bool, if True then plots 1x3 image figs for each batch element
		count: int

	returns:
		None (saves .png img)

	"""

	plt.figure(figsize=(15, 15))
	for n in range(8*8):
		ax = plt.subplot(8, 8, n+1)
		plt.axis('off')
		if grayscale:
			plt.imshow(input_batch[n], cmap='gray_r')
		else:
			plt.imshow(input_batch[n])
		plt.tight_layout()

	plt.tight_layout()
	plt.show()
	plt.savefig('gan_set{0:04d}.png'.format(count), dpi=410)
	plt.close()
	return

show_batch(image_batch, grayscale=True)