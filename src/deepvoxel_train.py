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
from deepvoxels.projection import ProjectionHelper
from deepvoxels.dataio import TestDataset
from deepvoxels.deep_voxels import DeepVoxels
from deepvoxels.util import parse_intrinsics, custom_load
from deepvoxels import data_util
from deepvoxel_style_transfer import StyleTransferModel

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--data_root", type=str, required=True)
parser.add_argument("--logging_root", type=str, required=True)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--style_image_path", type=str, required=True)
parser.add_argument("--num_iterations", type=int, required=True)
parser.add_argument("--img_sidelength", type=int, default=512)
parser.add_argument("--no_occlusion_net", action="store_true", default=False)
opt = parser.parse_args()
print("\n".join([f"({key}, {value})" for key, value in vars(opt).items()]))

# Some constants
device = torch.device("cuda")
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
dir_name = os.path.join(
    datetime.datetime.now().strftime('%m_%d'),
    datetime.datetime.now().strftime('%H-%M-%S_') + '_'.join(opt.checkpoint.strip('/').split('/')[-2:]) + '_' + opt.data_root.strip('/').split('/')[-1]
)
traj_dir = os.path.join(opt.logging_root, 'test_traj', dir_name)
depth_dir = os.path.join(traj_dir, 'depth')
data_util.cond_mkdir(traj_dir)
data_util.cond_mkdir(depth_dir)

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

# Generating an image from trained checkpoint and projection file

dataset = TestDataset(pose_dir=os.path.join(opt.data_root, 'pose'))
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

custom_load(model, opt.checkpoint)
for param in model.parameters():
    param.requires_grad = True
model.train()

dv = model.deepvoxels.detach().clone()
dv.requires_grad = True

optimizer = optim.Adam([dv])

forward_time = 0.
print('Starting generation of images...')
iter = 0
depth_imgs = []

for epoch in range(opt.num_iterations):
    print(f"Epoch: {epoch}")
    for trgt_pose in tqdm(dataloader):
        start = time.time()

        trgt_pose = trgt_pose.squeeze().to(device)
        # compute projection mapping
        proj_mapping = projection.compute_proj_idcs(trgt_pose.squeeze(), grid_origin)
        if proj_mapping is None:
            print("Invalid sample")
            continue

        proj_ind_3d, proj_ind_2d = proj_mapping

        output, depth_maps = model(
            None,
            [proj_ind_3d],
            [proj_ind_2d],
            None, None, None, dv
        )
        end = time.time()
        forward_time += end - start

        stm = StyleTransferModel(opt.style_image_path)
        sf = stm.extract_style_features(output[0])
        sl = stm.get_style_loss(sf)
        sl.backward()

        optimizer.step()
        break

overall_dv_change = torch.sum((dv.detach() - model.deepvoxels.detach())**2)
print(overall_dv_change)

# Now, generate some images using the updated deepvoxel!

model.deepvoxels = dv.detach().clone()
model.eval()

print('Generating images with updated style...')
with torch.no_grad():
    iter = 0
    depth_imgs = []
    for trgt_pose in dataloader:
        trgt_pose = trgt_pose.squeeze().to(device)

        start = time.time()
        # compute projection mapping
        proj_mapping = projection.compute_proj_idcs(trgt_pose.squeeze(), grid_origin)
        if proj_mapping is None:  # invalid sample
            print('(invalid sample)')
            continue

        proj_ind_3d, proj_ind_2d = proj_mapping

        # Run through model
        output, depth_maps, = model(None,
                                    [proj_ind_3d], [proj_ind_2d],
                                    None, None,
                                    None)
        end = time.time()
        forward_time += end - start

        output[0] = output[0][:, :, 5:-5, 5:-5]
        print("Iter %d" % iter)

        output_img = np.array(output[0].squeeze().cpu().detach().numpy())
        output_img = output_img.transpose(1, 2, 0)
        output_img += 0.5
        output_img *= 2 ** 16 - 1
        output_img = output_img.round().clip(0, 2 ** 16 - 1)

        depth_img = depth_maps[0].squeeze(0).cpu().detach().numpy()
        depth_img = depth_img.transpose(1, 2, 0)
        depth_imgs.append(depth_img)

        cv2.imwrite(os.path.join(traj_dir, "img_%05d.png" % iter), output_img.astype(np.uint16)[:, :, ::-1])

        iter += 1
        break

    depth_imgs = np.stack(depth_imgs, axis=0)
    depth_imgs = (depth_imgs - np.amin(depth_imgs)) / (np.amax(depth_imgs) - np.amin(depth_imgs))
    depth_imgs *= 2**16 - 1
    depth_imgs = depth_imgs.round()

    for i in range(len(depth_imgs)):
        cv2.imwrite(os.path.join(depth_dir, "img_%05d.png" % i), depth_imgs[i].astype(np.uint16))

print("Average forward pass time over %d examples is %f"%(iter, forward_time/iter))


