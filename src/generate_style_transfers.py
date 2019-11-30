'''
Combined run_deepvoxels.py with deep-transfer : load pretrained auto-encoder to be used for generating styled views.
'''
import argparse
import os, re, time, datetime

import torch
from torch import nn
import torchvision
import cv2
import numpy as np
from dataio import *
from torch.utils.data import DataLoader
from deep_voxels import DeepVoxels
from projection import ProjectionHelper


import time
import util
import argparse
from losses import *
from data_util import *
#from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from timeit import default_timer as timer

# deep transfer
import deep_transfer.PairDataset as PairDataset
import deep_transfer.autoencoder as autoencoder




def parse_args():
    parser = argparse.ArgumentParser()
    ### Deep Voxel specific arguments
 
    parser.add_argument('--data_root', required=True,help='Path to directory that holds the object data. See dataio.py for directory structure etc..')
    parser.add_argument('--logging_root', required=True,help='Path to directory where to write tensorboard logs and checkpoints.')

    parser.add_argument('--experiment_name', type=str, default='', help='(optional) Name for experiment.')
    parser.add_argument('--max_epoch', type=int, default=400, help='Maximum number of epochs to train for.')
    parser.add_argument('--lr', type=float, default=0.0004, help='Learning rate.')
    parser.add_argument('--l1_weight', type=float, default=200, help='Weight of l1 loss.')
    parser.add_argument('--sampling_pattern', type=str, default='all', required=False,help='Whether to use \"all\" images or whether to skip n images (\"skip_1\" picks every 2nd image.')

    parser.add_argument('--img_sidelength', type=int, default=512,help='Sidelength of generated images. Default 512. Only less than native resolution of images is recommended.')

    parser.add_argument('--no_occlusion_net', action='store_true', default=False,help='Disables occlusion net and replaces it with a fully convolutional 2d net.')
    parser.add_argument('--num_trgt', type=int, default=2, required=False, help='How many novel views will be generated at training time.')

    parser.add_argument('--checkpoint', default='',help='Path to a checkpoint to load model weights from.')
    parser.add_argument('--start_epoch', type=int, default=0,help='Start epoch')

    parser.add_argument('--grid_dim', type=int, default=32,help='Grid sidelength. Default 32.')
    parser.add_argument('--num_grid_feats', type=int, default=64,help='Number of features stored in each voxel.')
    parser.add_argument('--nf0', type=int, default=64,help='Number of features in outermost layer of U-Net architectures.')
    parser.add_argument('--near_plane', type=float, default=np.sqrt(3)/2,help='Position of the near plane.')


    ### deep-transfer specific arguments


    #parser.add_argument('--content', help='Path of the content image (or a directory containing images) to be trasformed')
    parser.add_argument('--style', help='Path of the style image (or a directory containing images) to use')
    parser.add_argument('--synthesis', default=False, action='store_true', help='Flag to syntesize a new texture. Must provide a texture style image')
    parser.add_argument('--stylePair', help='Path of two style images (separated by ",") to use in combination')
    parser.add_argument('--mask', help='Path of the binary mask image (white on black) to trasfer the style pair in the corrisponding areas')

    parser.add_argument('--contentSize', type=int, help='Reshape content image to have the new specified maximum size (keeping aspect ratio)') # default=768 in the paper
    parser.add_argument('--styleSize', type=int, help='Reshape style image to have the new specified maximum size (keeping aspect ratio)')

    parser.add_argument('--outDir', default='outputs', help='Path of the directory where stylized results will be saved')
    parser.add_argument('--outPrefix', help='Name prefixed in the saved stylized images')

    parser.add_argument('--alpha', type=float, default=0.2, help='Hyperparameter balancing the blending between original content features and WCT-transformed features')
    parser.add_argument('--beta', type=float, default=0.5, help='Hyperparameter balancing the interpolation between the two images in the stylePair')
    parser.add_argument('--no-cuda', default=False, action='store_true', help='Flag to enables GPU (CUDA) accelerated computations')

    return parser.parse_args()


def save_image(img, content_name, style_name, out_ext, args):
    torchvision.utils.save_image(img.cpu().detach().squeeze(0),
     os.path.join(args.outDir,
      (args.outPrefix + '_' if args.outPrefix else '') + content_name + '_stylized_by_' + style_name + '_alpha_' + str(int(args.alpha*100)) + '.' + out_ext))




def validate_args(args):
    supported_img_formats = ('.png', '.jpg', '.jpeg')

    # assert that we have a combinations of cli args meaningful to perform some task
    assert((args.content and args.style)   or (args.content and args.stylePair) or (args.style and args.synthesis) or (args.stylePair and args.synthesis) or (args.mask and args.content and args.stylePair))

    if args.content:
        if os.path.isfile(args.content) and os.path.splitext(args.content)[-1].lower().endswith(supported_img_formats):
            pass
        elif os.path.isdir(args.content) and any([os.path.splitext(file)[-1].lower().endswith(supported_img_formats) for file in os.listdir(args.content)]):
            pass
        else:
            raise ValueError("--content '" + args.content + "' must be an existing image file or a directory containing at least one supported image")

    if args.style:
        if os.path.isfile(args.style) and os.path.splitext(args.style)[-1].lower().endswith(supported_img_formats):
            pass
        elif os.path.isdir(args.style) and any([os.path.splitext(file)[-1].lower().endswith(supported_img_formats) for file in os.listdir(args.style)]):
            pass
        else:
            raise ValueError("--style '" + args.style + "' must be an existing image file or a directory containing at least one supported image")

    if args.stylePair:
        if len(args.stylePair.split(',')) == 2:
            args.style0 = args.stylePair.split(',')[0]
            args.style1 = args.stylePair.split(',')[1]
            if os.path.isfile(args.style0) and os.path.splitext(args.style0)[-1].lower().endswith(supported_img_formats) and \
                    os.path.isfile(args.style1) and os.path.splitext(args.style1)[-1].lower().endswith(supported_img_formats):
                pass
            else:
                raise ValueError("--stylePair '" + args.stylePair + "' must be an existing and supported image file paths pair")
            pass
        else:
            raise ValueError('--stylePair must be a comma separeted pair of image file paths')

    if args.mask:
        if os.path.isfile(args.mask) and os.path.splitext(args.mask)[-1].lower().endswith(supported_img_formats):
            pass
        else:
            raise ValueError("--mask '" + args.mask + "' must be an existing and supported image file path")

    if args.outDir != './outputs':
        args.outDir = os.path.normpath(args.outDir)
        if re.search(r'[^A-Za-z0-9- :_\\\/]', args.outDir):
            raise ValueError("--outDir '" + args.outDir + "' contains illegal characters")

    if args.outPrefix:
        args.outPrefix = os.path.normpath(args.outPrefix)
        if re.search(r'[^A-Za-z0-9-_\\\/]', args.outPrefix):
            raise ValueError("--outPrefix '" + args.outPrefix + "' contains illegal characters")

    if args.contentSize and (args.contentSize <= 0 or args.contentSize > 3840):
        raise ValueError("--contentSize '" + args.contentSize + "' have an invalid value (must be between 0 and 3840)")

    if args.styleSize and (args.styleSize <= 0 or args.styleSize > 3840):
        raise ValueError("--styleSize '" + args.styleSize + "' have an invalid value (must be between 0 and 3840)")

    if not 0. <= args.alpha <= 1.:
        raise ValueError("--alpha '" + args.alpha + "' have an invalid value (must be between 0 and 1)")

    if not 0. <= args.beta <= 1.:
        raise ValueError("--beta '" + args.beta + "' have an invalid value (must be between 0 and 1)")

    return args




### from deep-transfer
def transferStyle(args):
    
    args = validate_args(args)
    dataset = PairDataset.ContentStylePairDataset(args)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    if not args.no_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda:0')
    else:
        args.device = torch.device('cpu')

    ## this has to be fixed
    for i, sample in enumerate(dataloader):
        content = sample['content'].to(device=args.device)
        style = sample['style'].to(device=args.device)
    
        c_basename = str(os.path.basename(sample['contentPath'][0]).split('.')[0])
        c_ext = str(os.path.basename(sample['contentPath'][0]).split('.')[-1])
    
        s_basename = str(os.path.basename(sample['stylePath'][0]).split('.')[0])
        s_ext = str(os.path.basename(sample['stylePath'][0]).split('.')[-1])
    
        
        style_model = autoencoder.SingleLevelWCT(args).to(args.device)
    
        start = timer()
        out = style_model(content, style)
        end = timer()
    
        save_image(out, c_basename, s_basename, c_ext, args)

        print('Wall-clock time took for stylization: ' + str(end - start) + 's')





opt = parse_args()
print('\n'.join(["%s: %s" % (key, value) for key, value in vars(opt).items()]))

device = get_device()

input_image_dims = [opt.img_sidelength, opt.img_sidelength]
proj_image_dims = [64, 64] # Height, width of 2d feature map used for lifting and rendering.

# Read origin of grid, scale of each voxel, and near plane
_, grid_barycenter, scale, near_plane, _ = \
    util.parse_intrinsics(os.path.join(opt.data_root, 'intrinsics.txt'), trgt_sidelength=input_image_dims[0])

if near_plane == 0.0:
    near_plane = opt.near_plane

# Read intrinsic matrix for lifting and projection
lift_intrinsic = util.parse_intrinsics(os.path.join(opt.data_root, 'intrinsics.txt'),trgt_sidelength=proj_image_dims[0])[0]
proj_intrinsic = lift_intrinsic

# Set up scale and world coordinates of voxel grid
voxel_size = (1. / opt.grid_dim) * scale
grid_origin = torch.tensor(np.eye(4)).float().to(device).squeeze()
grid_origin[:3,3] = grid_barycenter

# Minimum and maximum depth used for rejecting voxels outside of the cmaera frustrum
depth_min = 0.
depth_max = opt.grid_dim * voxel_size + near_plane
grid_dims = 3 * [opt.grid_dim]

# Resolution of canonical viewing volume in the depth dimension, in number of voxels.
frustrum_depth = 2 * grid_dims[-1]

model = DeepVoxels(lifting_img_dims=proj_image_dims,
                   frustrum_img_dims=proj_image_dims,
                   grid_dims=grid_dims,
                   use_occlusion_net=not opt.no_occlusion_net,
                   num_grid_feats=opt.num_grid_feats,
                   nf0=opt.nf0,
                   img_sidelength=input_image_dims[0])
model.to(device)

# Projection module
projection = ProjectionHelper(projection_intrinsic=proj_intrinsic,
                              lifting_intrinsic=lift_intrinsic,
                              depth_min=depth_min,
                              depth_max=depth_max,
                              projection_image_dims=proj_image_dims,
                              lifting_image_dims=proj_image_dims,
                              grid_dims=grid_dims,
                              voxel_size=voxel_size,
                              device=device,
                              frustrum_depth=frustrum_depth,
                              near_plane=near_plane)




def test():
    # Create the training dataset loader
    dataset = TestDataset(pose_dir=os.path.join(opt.data_root, 'pose'))

    util.custom_load(model, opt.checkpoint)
    model.eval()

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    dir_name = os.path.join(datetime.datetime.now().strftime('%m_%d'),
                            datetime.datetime.now().strftime('%H-%M-%S_') +
                            '_'.join(opt.checkpoint.strip('/').split('/')[-2:]) + '_'
                            + opt.data_root.strip('/').split('/')[-1])

    traj_dir = os.path.join(opt.logging_root, 'test_traj', dir_name)
    depth_dir = os.path.join(traj_dir, 'depth')

    data_util.cond_mkdir(traj_dir)
    data_util.cond_mkdir(depth_dir)

    forward_time = 0.

    print('starting testing...')
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

            # style here : can also directly call the auto-encoder on the output
            opt.content = os.path.join(traj_dir, "img_%05d.png" % iter)
            transferStyle(opt)
            iter += 1


        depth_imgs = np.stack(depth_imgs, axis=0)
        depth_imgs = (depth_imgs - np.amin(depth_imgs)) / (np.amax(depth_imgs) - np.amin(depth_imgs))
        depth_imgs *= 2**16 - 1
        depth_imgs = depth_imgs.round()

        for i in range(len(depth_imgs)):
            cv2.imwrite(os.path.join(depth_dir, "img_%05d.png" % i), depth_imgs[i].astype(np.uint16))

    print("Average forward pass time over %d examples is %f"%(iter, forward_time/iter))


def main():
    test()


if __name__ == '__main__':
    main()
