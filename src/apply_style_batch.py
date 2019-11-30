import argparse
from tqdm import tqdm
import os
from torchvision import transforms
from torch import optim
from deepvoxel_style_transfer_2 import StyleTransferModel2
import utils
from deepvoxels import data_util

def get_white_region_mask(images):
    """
        images: [B, C, H, W] in range [0, 1]
    """
    mask = (images < 0.999)
    mask = mask.all(dim=1)
    return mask.type_as(images)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--src_images_dir", required=True, type=str)
    parser.add_argument("-l", "--logs_dir", required=True, type=str)
    parser.add_argument("-st", "--style_image_path", required=True, type=str)
    parser.add_argument("-n", "--num_iters", required=True, type=int)
    parser.add_argument("-sc", "--style_coeff", default=10000, type=float)
    parser.add_argument("-cc", "--content_coeff", default=1, type=float)
    args = parser.parse_args()
    dst_dir = os.path.join(args.logs_dir, "data", utils.get_log_dir())
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    device = data_util.get_device()
    imsize = 512
    loader = transforms.Compose([
        transforms.Resize(imsize),
        transforms.ToTensor()]
    )

    style_img = utils.image_loader(args.style_image_path, loader, device)
    
    img_names = os.listdir(args.src_images_dir)
    for img_name in tqdm(img_names):
        img_path = os.path.join(args.src_images_dir, img_name)
        content_img = utils.image_loader(img_path, loader, device)
        input_img = content_img.clone()

        input_img_mask = get_white_region_mask(input_img)

        model = StyleTransferModel2(content_img, style_img)
        for p in model.model.parameters():
            p.requires_grad = False
        optimizer = optim.LBFGS([input_img.requires_grad_()])

        for iter_num in range(args.num_iters):
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
        input_img_masked = input_img_mask * input_img
        unloader = transforms.ToPILImage()  # reconvert into PIL image
        image = input_img_masked.cpu()
        image = image.squeeze(0)
        image = unloader(image)
        image.save(os.path.join(dst_dir, img_name))