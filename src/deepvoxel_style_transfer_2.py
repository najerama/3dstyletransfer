# Code is borrowed from https://github.com/pytorch/tutorials/blob/master/advanced_source/neural_style_tutorial.py
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.models as models
import copy
import argparse
from tqdm import tqdm
from deepvoxels import data_util
import utils


class ContentLoss(nn.Module):
    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input


class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature.detach())

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input


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


def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)


class StyleTransferModel2:
    def __init__(self, content_img, style_img):
        device = data_util.get_device()
        self.device = device
        cnn = models.vgg19(pretrained=True).features.to(device).eval()
        cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
        cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)
        normalization = Normalization(cnn_normalization_mean, cnn_normalization_std).to(device)
        model = nn.Sequential(normalization)

        content_losses = []    
        content_layers = ['conv_4']
        style_losses = []
        style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

        i = 0  # increment every time we see a conv
        for layer in cnn.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = 'conv_{}'.format(i)
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(i)
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

        self.model = model
        self.style_losses = style_losses
        self.content_losses = content_losses

    def get_loss(self, input_img):
        self.model(input_img)
        style_score = 0
        content_score = 0

        for sl in self.style_losses:
            style_score += sl.loss
        for cl in self.content_losses:
            content_score += cl.loss
        return style_score, content_score


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

    imsize = 512
    device = data_util.get_device()
    loader = transforms.Compose([
        transforms.Resize(imsize),
        transforms.ToTensor()]
    )

    content_img = utils.image_loader(args.src_img_path, loader, device)
    style_img = utils.image_loader(args.style_img_path, loader, device)
    input_img = content_img.clone()

    model = StyleTransferModel2(content_img, style_img)
    for p in model.model.parameters():
        p.requires_grad = False

    optimizer = optim.LBFGS([input_img.requires_grad_()])
    for iter_num in tqdm(range(args.num_iters)):
        def closure():
            input_img.data.clamp_(0, 1)
            optimizer.zero_grad()
            ss, cs = model.get_loss(input_img)
            ss = args.style_coeff * ss 
            cs = args.content_coeff * cs
            l = ss + cs
            l.backward()
            return ss + cs
        optimizer.step(closure)
    input_img.data.clamp_(0, 1)
    
    unloader = transforms.ToPILImage()  # reconvert into PIL image
    image = input_img.cpu()
    image = image.squeeze(0)
    image = unloader(image)
    image.save("a.png")