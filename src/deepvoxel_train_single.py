import torch
import argparse
import os
import numpy as np
import datetime
import time
from tqdm import tqdm
import cv2
from torch.utils.data import DataLoader
from torch import optim
from tensorboardX import SummaryWriter
from PIL import Image
from random import sample
import pytz
import torchvision.transforms as transforms
from deepvoxels.projection import ProjectionHelper
from deepvoxels.dataio import TestDataset
from deepvoxels.deep_voxels import DeepVoxels
from deepvoxels.util import parse_intrinsics, custom_load
from deepvoxels import data_util
from deepvoxel_style_transfer import StyleTransferModel
from deepvoxel_style_transfer_2 import StyleTransferModel2
import utils

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--data_root", type=str, required=True)
parser.add_argument("--logging_root", type=str, required=True)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--style_image_path", type=str, required=True)
parser.add_argument("--num_iterations", type=int, required=True)
parser.add_argument("--img_sidelength", type=int, default=512)
parser.add_argument("--no_occlusion_net", action="store_true", default=False)
parser.add_argument("--style_coeff", default=0.5, type=float)
parser.add_argument("--content_coeff", default=0.5, type=float)
opt = parser.parse_args()
print("\n".join([f"({key}, {value})" for key, value in vars(opt).items()]))

# Some constants
device = data_util.get_device()
proj_image_dims = [64, 64]
grid_dim = 32
grid_dims = 3*[grid_dim]
num_grid_feats = 64
nf0 = 64
input_image_dims = [opt.img_sidelength, opt.img_sidelength]
_, grid_barycenter, scale, near_plane, _ = parse_intrinsics(
    os.path.join(opt.data_root, "intrinsics.txt"),
    trgt_sidelength=input_image_dims[0]
)
if near_plane == 0.0:
    near_plane = np.sqrt(3)/2
voxel_size = (1. / grid_dim) * scale
grid_origin = torch.tensor(np.eye(4)).float().to(device).squeeze()
grid_origin[:3, 3] = grid_barycenter

lift_intrinsic = parse_intrinsics(os.path.join(opt.data_root, "intrinsics.txt"), trgt_sidelength=proj_image_dims[0])[0]
proj_intrinsic = lift_intrinsic

depth_min = 0.
depth_max = grid_dim * voxel_size + near_plane
frustrum_depth = 2 * grid_dims[-1]

# Result logging directory
tz = pytz.timezone("US/Eastern")
d = datetime.datetime.now(tz)
dir_name = os.path.join(
    d.strftime('%m_%d'),
    d.strftime('%H-%M-%S_') + '_'.join(opt.checkpoint.strip('/').split('/')[-2:]) + '_' + opt.data_root.strip('/').split('/')[-1]
)
traj_dir = os.path.join(opt.logging_root, 'test_traj', dir_name)
depth_dir = os.path.join(traj_dir, 'depth')
runs_dir = os.path.join(opt.logging_root, "runs", dir_name)
data_util.cond_mkdir(traj_dir)
data_util.cond_mkdir(depth_dir)
data_util.cond_mkdir(runs_dir)

# Define DeepVoxel Model
model = DeepVoxels(
    lifting_img_dims = proj_image_dims,
    frustrum_img_dims = proj_image_dims,
    grid_dims = grid_dims,
    use_occlusion_net=not opt.no_occlusion_net,
    num_grid_feats = num_grid_feats,
    nf0 = nf0,
    img_sidelength = input_image_dims[0],
)
custom_load(model, opt.checkpoint)
# for param in model.parameters():
#     param.requires_grad = False
model.eval()
model.to(device)

# Project Module
projection = ProjectionHelper(
    projection_intrinsic = proj_intrinsic,
    lifting_intrinsic = lift_intrinsic,
    depth_min = depth_min,
    depth_max = depth_max,
    projection_image_dims = proj_image_dims,
    lifting_image_dims = proj_image_dims,
    grid_dims = grid_dims,
    voxel_size = voxel_size,
    device = device,
    frustrum_depth = frustrum_depth,
    near_plane = near_plane,
)

dataset = TestDataset(pose_dir=os.path.join(opt.data_root, 'pose'))
trgt_pose = dataset[sample(range(len(dataset)), 1)[0]]

dv = model.deepvoxels.detach().clone()
dv.requires_grad = True
dv_orig = model.deepvoxels.detach().clone()

optimizer = optim.Adam([dv], lr=0.01)
writer = SummaryWriter(runs_dir, flush_secs=20)

# Utility function
def get_images_from_poses(trgt_poses, dv):
    # Add some initial renderings of the network to writer
    trgt_poses = trgt_poses.to(device)
    proj_ind_3d = []
    proj_ind_2d = []
    for i in range(trgt_poses.shape[0]):
        proj_mapping = projection.compute_proj_idcs(trgt_poses[i], grid_origin)
        if proj_mapping is None:
            print("Invalid sample")
            continue
        proj_ind_3d.append(proj_mapping[0])
        proj_ind_2d.append(proj_mapping[1])
        
    output, _ = model(
        None,
        proj_ind_3d,
        proj_ind_2d,
        None, None, None, dv
    )
    output = torch.cat(output)
    return output


loader = transforms.Compose([
    transforms.Resize((opt.img_sidelength, opt.img_sidelength)),
    transforms.ToTensor()]
)

style_img = utils.image_loader(opt.style_image_path, loader, device)
writer.add_image("Input images src style", style_img[0], 0)

# Dummy Content Image
with torch.no_grad():
    output_image = get_images_from_poses(trgt_pose.unsqueeze(0), dv_orig).squeeze(0)
    writer.add_image("Initial-Rendered-Images", output_image + 0.5, 0)
content_img = output_image + 0.5
content_img = content_img.unsqueeze(0)
stm = StyleTransferModel2(content_img, style_img)

# Train
trgt_pose = trgt_pose.unsqueeze(0)
for epoch in tqdm(range(opt.num_iterations)):
    # for batch_num, trgt_poses in enumerate(tqdm(dataloader)):
    optimizer.zero_grad()
    
    output_images = get_images_from_poses(trgt_pose, dv)
    output_images = output_images + 0.5
    style_loss, content_loss = stm.get_loss(output_images)
    loss = style_loss * opt.style_coeff + content_loss * opt.content_coeff
    loss.backward()
    optimizer.step()
    
    writer.add_scalar("Overall-DV-SSE", torch.sum((dv.detach() - model.deepvoxels.detach())**2), epoch)
    writer.add_scalars("Loss", {
        "style loss(scaled)": style_loss.item() * opt.style_coeff, 
        "content loss(scaled)": content_loss.item() * opt.content_coeff,
        "ovearll loss": loss.item()
    }, epoch)
    writer.add_image("Rendered-Image", output_images[0], epoch)
writer.close()