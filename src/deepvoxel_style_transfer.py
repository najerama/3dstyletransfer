import torch
from torchvision import models
import numpy as np
from PIL import Image
from functools import reduce
from deepvoxels.util import custom_load 
from deepvoxels import data_util

class StyleTransferModel:
    def __init__(
        self,
        filepath_style,
        img_size
    ):
        self.img_size = img_size
        self.device = torch.device('cuda')

        # Load style image
        reference_image = self.load_rgb(filepath_style)
        reference_image = reference_image[None, :, :, :]
        reference_image = torch.tensor(reference_image, device=self.device, dtype=torch.float)

        # load style transfer model
        vgg16 = models.vgg16(pretrained=True)
        self.style_model = list(vgg16.features.children())[:23]
        for l in self.style_model:
            l.to(self.device)
        self.feature_layers = [3, 8, 15, 22]

        # cache style features for style image
        with torch.no_grad():
            self.features_ref = self.extract_style_features(reference_image)            
        self.features_ref = [feat.to(self.device) for feat in self.features_ref]

    def load_rgb(self, path):
        # Borrowed from class NovelViewTriplets
        img = data_util.load_img(path, square_crop=True, downsampling_order=1, target_size=self.img_size)
        img = img[:, :, :3].astype(np.float32) / 255. - 0.5
        img = img.transpose(2,0,1)
        return img

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

    def get_style_loss(self, features):
        loss  = [torch.sum((f-fr)**2) for f, fr in zip(features, self.features_ref)]
        loss_overall = torch.tensor(0., device=self.device)
        for l in loss:
            loss_overall = loss_overall + l
        batch_size = features[0].shape[0]
        loss_overall = loss_overall/batch_size
        return loss_overall

if __name__ == "__main__":
    # Extract style features for a dummy image
    dummy_image_path = "../deepvoxels-data/synthetic_scene/test/bus_test/rgb/000001.png"
    reference_image = Image.open(dummy_image_path)
    reference_image = np.array(reference_image)[:, :, :3]
    reference_image = reference_image.transpose((2,0,1))[None, :, :, :]
    reference_image = torch.tensor(reference_image, device=torch.device("cuda"), dtype=torch.float)

    style_image_path = "../style_transfer_3d/examples/data/styles/bailly1.jpg"
    style_transfer_model = StyleTransferModel(style_image_path, (512, 512))
    style_features = style_transfer_model.extract_style_features(reference_image)
    style_loss = style_transfer_model.get_style_loss(style_features) 
    print(style_loss)
