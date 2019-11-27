import os
import torch
import numpy as np
from PIL import Image
from functools import reduce
import argparse
from tqdm import tqdm
import utils_test
from torchvision import models
import matplotlib.pyplot as plt
from deepvoxels import data_util
from tensorboardX import SummaryWriter
from deepvoxels.util import custom_load 
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
device = data_util.get_device()

### here

class ContentLoss(nn.Module):

    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input


class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = utils_test.gram_matrix(target_feature).detach()

    def forward(self, input):
        G = utils_test.gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input

'''
PyTorch’s implementation of VGG is a module divided into two child Sequential modules: features (containing convolution and pooling layers), 
and classifier (containing fully connected layers).
 We will use the features module because we need the output of the individual convolution layers to measure content and style loss
'''
cnn = models.vgg19(pretrained=True).features.to(device).eval()


cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

# create a module to normalize input image so we can easily put it in a
# nn.Sequential
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std


content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default):
    cnn = copy.deepcopy(cnn)

    # normalization module
    normalization = Normalization(normalization_mean, normalization_std).to(device)

    # just in order to have an iterable access to or list of content/syle
    # losses
    content_losses = []
    style_losses = []

    # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
    # to put in modules that are supposed to be activated sequentially
    model = nn.Sequential(normalization)

    i = 0  # increment every time we see a conv
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            # The in-place version doesn't play very nicely with the ContentLoss
            # and StyleLoss we insert below. So we replace with out-of-place
            # ones here.
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            # add content loss:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # add style loss:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # now we trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses

def get_input_optimizer(input_img):
    # this line to show that input is a parameter that requires a gradient

    print(input_img.shape)
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    return optimizer

def get_mask(input_img):
    #print(input_img.shape)
    #print(input_img)
    xsum = input_img.cpu().numpy().sum(axis=1)
    xsum = xsum[:,np.newaxis,:,:]
    xsum = np.concatenate([xsum,xsum,xsum],axis=1)
    mask = (1.0-np.isclose(xsum,3.0) * 1.0)
    return mask

def run_style_transfer(cnn, normalization_mean, normalization_std,
                       content_img, style_img, input_img, num_steps=300,
                       style_weight=1000000, content_weight=1):
    """Run the style transfer."""
    print('Building the style transfer model..')
    model, style_losses, content_losses = get_style_model_and_losses(cnn,
        normalization_mean, normalization_std, style_img, content_img)
    mask = torch.tensor(get_mask(input_img)).to(device)
    optimizer = get_input_optimizer(input_img)

    print('Optimizing..')
    run = [0]
    while run[0] <= num_steps:

        def closure():
            # correct the values of updated input image
            input_img.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()
            tmp = input_img.grad
            #print(input_img.grad.shape, tmp.shape, mask.shape)
            input_img.grad =  torch.mul(tmp.float(),mask.float())

            run[0] += 1
            if run[0] % 50 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))
                print()

            return style_score + content_score
        optimizer.step(closure)

    # a last correction...
    input_img.data.clamp_(0, 1)

    return input_img





if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-src", "--src_img_path", required=True, type=str)
    parser.add_argument("-style", "--style_img_path", required=True, type=str)
    parser.add_argument("-n", "--num_iters", required=True, type=int)
    parser.add_argument("-l", "--log_dir", required=True, type=str)
    parser.add_argument("-sc", "--style_coeff", default=0.5, type=float)
    parser.add_argument("-cc", "--content_coeff", default=0.5, type=float)

    args = parser.parse_args()

    run_dir = os.path.join(args.log_dir, "runs", "img_style_transfer", utils_test.get_log_dir())

    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
    writer = SummaryWriter(run_dir, flush_secs=10)
    #img_size = (512, 512)
    

    # Load Images
    imsize = 512
    loader = transforms.Compose([transforms.Resize(imsize),transforms.ToTensor()])

    style_img = utils_test.image_loader(args.style_img_path,device, loader)
    content_img = utils_test.image_loader(args.src_img_path, device, loader)

    assert style_img.size() == content_img.size(), \
    "we need to import style and content images of the same size"


    output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                            content_img, style_img, content_img.clone())

    plt.figure()
    unloader = transforms.ToPILImage()
    utils_test.imshow(output, unloader ,title='Output Image')
