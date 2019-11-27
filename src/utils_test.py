import torch
import pytz
import datetime
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from deepvoxels import data_util
import torchvision.transforms as transforms



def get_log_dir():
    # Result logging directory
    tz = pytz.timezone("US/Eastern")
    d = datetime.datetime.now(tz)
    dir_name = d.strftime('%y_%m_%d_%H_%M_%S')
    return dir_name


#imsize = 512 if torch.cuda.is_available() else 128  # use small size if no gpu


def image_loader(image_name,device,loader):
    image = Image.open(image_name)
    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)


def imshow(tensor, unloader, title=None):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)      # remove the fake batch dimension
    image = unloader(image)
    plt.imsave("src/" + title +".png",image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001) # pause a bit so that plots are updated


def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)

