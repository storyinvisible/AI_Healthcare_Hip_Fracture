import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
def plot_salient(model,criterion, ld_test, index, path,device):
    plt.figure(figsize = (20,20),dpi=300)
    image_index=0
    for i in range(3):
        for j in range(3):
            img, label = ld_test[index]
            img= img.to(device)
            label=torch.tensor([label]).to(device)
            img_grad = torch.unsqueeze(img, 0)
            img_grad.requires_grad_()
            log_prob = model(img_grad)
            loss = criterion(log_prob, label)
            loss.backward()
            plt.subplot(3, 3, image_index + 1)

            pre_process = transforms.Compose([transforms.Grayscale(num_output_channels=1)])
            datagrad = np.abs(img_grad.grad.data.to("cpu").numpy()[0, 0, :, :])
            datagrad2 = np.abs(img_grad.grad.data.to("cpu").numpy()[0, 1, :, :])
            datagrad3 = np.abs(img_grad.grad.data.to("cpu").numpy()[0, 2, :, :])
            datagrad = datagrad + datagrad2 + datagrad3
            # Remove x-axis and y-axis ticks from plot
            plt.xticks([], [])
            plt.yticks([], [])
            pred = torch.argmax(log_prob)

            # Labels for each image subplotx`
            plt.title(str(image_index))

            # Display image
            plt.imshow(1-datagrad,cmap='Greys')
            index += 1
            image_index+=1
    plt.savefig(path)

def plot_xray9(ld_test, model, index, path):
    plt.figure(figsize = (20,20),dpi=300)
    image_index=0
    for i in range(3):
        for j in range(3):
            img, label = ld_test[index]
            model.to("cpu")
            log_prob = model(torch.unsqueeze(img, 0))
            plt.subplot(3, 3, image_index + 1)
            
            img = torch.sum(img, axis=0)
            # Remove x-axis and y-axis ticks from plot
            plt.xticks([], [])
            plt.yticks([], [])
            pred = torch.argmax(log_prob)

            # Labels for each image subplotx`
            plt.title("Actual:{} predicted :{}".format(label, int(pred)))

            # Display image
            plt.imshow(img,cmap='Greys')
            image_index += 1
            index+=1
            if index > len(ld_test):
                break
    plt.savefig(path)

def grad_cam(vis,img ):
    pass