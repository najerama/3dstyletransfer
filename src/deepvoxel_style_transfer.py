import torch
from torchvision import models
import numpy as np
from PIL import Image
from functools import reduce
import argparse
from tqdm import tqdm
import os
from tensorboardX import SummaryWriter
from torchvision import transforms

import sys
sys.path.insert(0, "/local/crv/sa3762/Dev/3dstyletransfer/")
sys.path.insert(0, "/local/crv/sa3762/Dev/3dstyletransfer/deepvoxels/")

from deepvoxels.util import custom_load 
from deepvoxels import data_util
import utils


class StyleTransferModel:
    def __init__(self):
        self.device = data_util.get_device()

        # load style transfer model
        vgg16 = models.vgg16(pretrained=True)
        self.style_model = list(vgg16.features.children())[:23]
        for l in self.style_model:
            l.to(self.device)
            l.eval()
            for parameters in l.parameters():
                parameters.requires_grad = False    
        self.feature_layers = [3, 8, 15, 22]

    def extract_raw_features(self, images):
        features = []
        out = images
        for i, l in enumerate(self.style_model):
            out = l(out)
            if i in self.feature_layers:
                features.append(out)
        return features

    def extract_style_features(self, images, masks):
        """
        images: [B, C, H, W]. Values between [-0.5, 0.5] obtained after mean, var normalized per channel
        masks: [B, H, W]. Values {0, 1}
        """        
        if masks is None:
            masks = torch.ones((images.shape[0], images.shape[2], images.shape[3]), device = self.device, dtype=torch.float)
        features = self.extract_raw_features(images)

        # Extract style matrices for each layer's features
        style_features = []
        for feature in features:  # [B, Cp, Hp, Wp]
            scale = int(masks.shape[-1] / feature.shape[-1])
            m = torch.nn.functional.avg_pool2d(masks[:, None, :, :], kernel_size=scale, stride=scale)
            dim = feature.shape[1]  # Number of feature channels Cp
            m = m.view((m.shape[0], -1))  # [B, H*W]
            f2 = feature.permute(0, 2, 3, 1)  # [B, Hp, Wp, Cp]
            f2 = f2.view((f2.shape[0], -1, f2.shape[-1]))  # [B, Hp*Wp, Cp]
            f2 = f2 * torch.sqrt(m)[:, :, None]  # Multiply mask to each channel.
            f2 = torch.matmul(f2.permute(0, 2, 1), f2)  # [B, Cp, Cp]
            f2 = f2 / (dim * m.sum(dim=1)[:, None, None])  # Normalize
            style_features.append(f2)
        return style_features

    def extract_content_features(self, images, masks):
        """
        images: [B, C, H, W]; [-0.5, 0.5]
        masks: [B, H, W]; {0, 1}
        """
        if masks is None:
            masks = torch.ones((images.shape[0], images.shape[2], images.shape[3]), device = self.device, dtype=torch.float)
        
        features = self.extract_raw_features(images)

        content_features = []
        for feature in features:  # [B, Cp, Hp, Wp]
            scale = int(masks.shape[-1] / feature.shape[-1])
            m = torch.nn.functional.avg_pool2d(masks[:, None, :, :], kernel_size=scale, stride=scale)  # [B, 1, Hp, Wp]

            feature = feature * m
            content_features.append(feature)        
        return content_features

    def get_loss(self, feat, feat_ref):
        loss  = [torch.sum((f-fr)**2) for f, fr in zip(feat, feat_ref)]
        loss_overall = torch.tensor(0., device=self.device)
        for l in loss:
            loss_overall = loss_overall + l
        batch_size = feat[0].shape[0]
        loss_overall = loss_overall/batch_size
        return loss_overall

    def get_style_loss(self, feat_src, feat_dst):
        return self.get_loss(feat_src, feat_dst)

    def get_content_loss(self, feat_src, feat_dst):
        return self.get_loss(feat_src, feat_dst)



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

    run_dir = os.path.join(args.log_dir, "runs", "img_style_transfer", utils.get_log_dir())
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
    writer = SummaryWriter(run_dir, flush_secs=10)
    img_size = (512, 512)
    device = data_util.get_device()

    mean = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=torch.float)[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=torch.float)[None, :, None, None]
    # normalize = transforms.Normalize(mean=mean, std=std)

    def normalize(images):
        return (images - mean) / std

    def denormalize(images):
        return images * std + mean
    
    def get_white_region_mask(images):
        """
            images: [B, C, H, W] in range [0, 1]
        """
        mask = (images > 0.999)
        mask = mask.all(dim=1)
        return mask.type_as(images)    
    
    # Load Images
    src_img = torch.tensor(utils.load_rgb(args.src_img_path, img_size), device=device, dtype=torch.float) + 0.5
    src_img = src_img.unsqueeze(dim=0)
    src_mask = get_white_region_mask(src_img)
    src_img = normalize(src_img)
    writer.add_image("Source Mask", src_mask, 0)

    style_img = torch.tensor(utils.load_rgb(args.style_img_path, img_size), device=device, dtype=torch.float) + 0.5
    style_img = style_img.unsqueeze(dim=0)
    style_img = normalize(style_img)
    writer.add_image("Input images src style", denormalize(torch.cat((src_img[0], style_img[0]), dim=-1)), 0)

    stm = StyleTransferModel()
    style_features = stm.extract_style_features(style_img, None)
    content_features = stm.extract_content_features(src_img, src_mask)

    src_img.requires_grad = True
    optimizer = torch.optim.Adam([src_img], lr=0.01)
    
    for num_iter in tqdm(range(args.num_iters)):
        optimizer.zero_grad()
        sf = stm.extract_style_features(src_img, src_mask)
        sl = stm.get_style_loss(sf, style_features)

        cf = stm.extract_content_features(src_img, src_mask)
        cl = stm.get_content_loss(cf, content_features)

        # l = (args.style_coeff) * sl + (args.content_coeff) * cl
        l = sl
        l.backward()
        # src_img.grad = src_img.grad * src_mask
        optimizer.step()
        if num_iter%10:
            writer.add_scalars("Loss", {
                "style loss(scaled)": sl.item() * args.style_coeff, 
                "content loss(scaled)": cl.item() * args.content_coeff,
                "ovearll loss": l.item()
            }, num_iter)
            writer.add_image("Style Transfer Image", denormalize(src_img), num_iter)
