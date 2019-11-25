import torch
from torchvision import models
import numpy as np
from PIL import Image
from functools import reduce
import argparse
from tqdm import tqdm
import os
from tensorboardX import SummaryWriter
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

    def extract_style_features(self, images):
        out = images
        results = []
        for i, l in enumerate(self.style_model):
            out = l(out)
            if i in self.feature_layers:
                feat = out
                feat = feat.permute((0,2,3,1))
                feat = feat.reshape((feat.shape[0], -1, feat.shape[-1]))
                feat_transpose = torch.transpose(feat, 2, 1)
                feat1 = torch.matmul(feat_transpose, feat)
                results.append(feat1)
        return results

    def extract_content_features(self, images):
        out = images
        results = []
        for i, l in enumerate(self.style_model):
            out = l(out)
            if i in self.feature_layers:
                results.append(out)
        return results

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

    # Load Images
    src_img = utils.load_rgb(args.src_img_path, img_size)
    style_img = utils.load_rgb(args.style_img_path, img_size)
    writer.add_image("Input images src style", np.concatenate((src_img, style_img), axis=-1)+0.5, 0)

    # Prepare Images
    src_img = np.expand_dims(src_img, axis=0)
    src_img = torch.tensor(src_img, device=device, dtype=torch.float)
    style_img = np.expand_dims(style_img, axis=0)
    style_img = torch.tensor(style_img, device=device, dtype=torch.float)
    
    stm = StyleTransferModel()
    style_features = stm.extract_style_features(style_img)
    content_features = stm.extract_content_features(src_img)

    src_img.requires_grad = True
    optimizer = torch.optim.Adam([src_img])
    
    for num_iter in tqdm(range(args.num_iters)):
        optimizer.zero_grad()
        sf = stm.extract_style_features(src_img)
        sl = stm.get_style_loss(sf, style_features)

        cf = stm.extract_content_features(src_img)
        cl = stm.get_content_loss(cf, content_features)

        l = (args.style_coeff) * sl + (args.content_coeff) * cl
        l.backward()
        optimizer.step()
        if num_iter%10:
            writer.add_scalars("Loss", {
                "style loss(scaled)": sl.item() * args.style_coeff, 
                "content loss(scaled)": cl.item() * args.content_coeff,
                "ovearll loss": l.item()
            }, num_iter)
            writer.add_image("Style Transfer Image", src_img+0.5, num_iter)
