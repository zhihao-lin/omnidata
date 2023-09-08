import torch
import torch.nn.functional as F
from torchvision import transforms

import PIL
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2 

import argparse
import os
from pathlib import Path
import glob
import sys
import pdb
from tqdm import tqdm
from .modules.unet import UNet
from .modules.midas.dpt_depth import DPTDepthModel
from .data.transforms import get_transform


parser = argparse.ArgumentParser(description='Visualize output for depth or surface normals')

parser.add_argument('--task', dest='task', help="normal or depth", default='normal')
parser.add_argument('--source_dir', help="path to rgb image")
parser.add_argument('--output_dir', help="path to where output image should be stored")
parser.add_argument('--start', type=int, help='starting frame index')
parser.add_argument('--end', type=int, help='ending frame index')
parser.add_argument('--img_size', type=int, default=384)
args = parser.parse_args()

root_dir = '/home/zhi-hao/Desktop/omnidata/omnidata_tools/torch/pretrained_models/'

trans_topil = transforms.ToPILImage()
map_location = (lambda storage, loc: storage.cuda()) if torch.cuda.is_available() else torch.device('cpu')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# get target task and model
if args.task == 'normal':
    pretrained_weights_path = root_dir + 'omnidata_dpt_normal_v2.ckpt'
    model = DPTDepthModel(backbone='vitb_rn50_384', num_channels=3) # DPT Hybrid
    checkpoint = torch.load(pretrained_weights_path, map_location=map_location)
    if 'state_dict' in checkpoint:
        state_dict = {}
        for k, v in checkpoint['state_dict'].items():
            state_dict[k[6:]] = v
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.to(device)
    trans_totensor = transforms.Compose([get_transform('rgb', image_size=None)])

elif args.task == 'depth':
    pretrained_weights_path = root_dir + 'omnidata_dpt_depth_v2.ckpt'  # 'omnidata_dpt_depth_v1.ckpt'
    # model = DPTDepthModel(backbone='vitl16_384') # DPT Large
    model = DPTDepthModel(backbone='vitb_rn50_384') # DPT Hybrid
    checkpoint = torch.load(pretrained_weights_path, map_location=map_location)
    if 'state_dict' in checkpoint:
        state_dict = {}
        for k, v in checkpoint['state_dict'].items():
            state_dict[k[6:]] = v
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    trans_totensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5, std=0.5)])

else:
    print("task should be one of the following: normal, depth")
    sys.exit()

trans_rgb = transforms.Compose([transforms.Resize(512, interpolation=PIL.Image.BILINEAR),
                                transforms.CenterCrop(512)])

def depth2img(depth, scale=1):
    depth = depth/scale
    depth = np.clip(depth, a_min=0., a_max=1.)
    depth_img = cv2.applyColorMap((depth*255).astype(np.uint8),
                                  cv2.COLORMAP_MAGMA)
    return depth_img

def nearest_multiple(x, base=128):
    s = np.round(x/base)
    multiple = int(base * s)
    return multiple

def standardize_depth_map(img, mask_valid=None, trunc_value=0.1):
    if mask_valid is not None:
        img[~mask_valid] = torch.nan
    sorted_img = torch.sort(torch.flatten(img))[0]
    # Remove nan, nan at the end of sort
    num_nan = sorted_img.isnan().sum()
    if num_nan > 0:
        sorted_img = sorted_img[:-num_nan]
    # Remove outliers
    trunc_img = sorted_img[int(trunc_value * len(sorted_img)): int((1 - trunc_value) * len(sorted_img))]
    trunc_mean = trunc_img.mean()
    trunc_var = trunc_img.var()
    eps = 1e-6
    # Replace nan by mean
    img = torch.nan_to_num(img, nan=trunc_mean)
    # Standardize
    img = (img - trunc_mean) / torch.sqrt(trunc_var + eps)
    return img


def save_outputs_patch(img_path, save_path):
    patch = args.img_size
    with torch.no_grad():
        img = np.array(Image.open(img_path))
        h, w, c = np.array(img).shape
        start_ind = list(range(0, w - h // 4 * 3 , h // 4))
        start_ind[-1] = w - h

        final = torch.zeros((1, 3, h, w))
        weights = torch.ones((1, 3, h, w)) * 1e-6
        for ind in start_ind:
            img_patch = Image.fromarray(img[:, ind:ind+h])
            img_tensor = trans_totensor(img_patch)[:3].unsqueeze(0).to(device)
            if img_tensor.shape[1] == 1:
                img_tensor = img_tensor.repeat_interleave(3,1)
            output = model(img_tensor).clamp(min=0, max=1)
            import torchvision.transforms.functional as F
            final[:, :, :, ind:ind+h] = final[:, :, :, ind:ind+h] + F.resize(output, h).detach().cpu()
            weights[:, :, :, ind:ind+h] = weights[:, :, :, ind:ind+h] + 1.0
        final = final / weights

        if args.task == 'depth':
            output = output.clamp(0,1)
            output = 1 - output
            # output = standardize_depth_map(output)
            plt.imsave(save_path, output.detach().cpu().squeeze(),cmap='viridis')

        else:
            trans_topil(final[0]).save(save_path)

def save_output_whole(img_path, save_path):
    img = np.array(Image.open(img_path))
    h, w = img.shape[:2]
    base = 128
    h_t, w_t = nearest_multiple(h, base), nearest_multiple(w, base)
    img_t = cv2.resize(img, (w_t, h_t), interpolation=cv2.INTER_LINEAR)
    img_tensor = trans_totensor(img_t).unsqueeze(0).to(device) # (1, 3, h, w)
    if img_tensor.shape[1] == 1:
        img_tensor = img_tensor.repeat_interleave(3, 1)

    output = model(img_tensor).clamp(min=0, max=1)
    img_out = output[0].cpu().detach().numpy()
    if args.task == 'normal':
        img_out = cv2.resize(img_out.transpose(1, 2, 0), (w, h), interpolation=cv2.INTER_LINEAR)
        img_out = img_out.transpose(2, 0, 1)
    else:
        img_out = cv2.resize(img_out, (w, h), interpolation=cv2.INTER_LINEAR)
    np.save(save_path.replace(".png", ".npy"), img_out)
    if args.task == 'normal':
        img_out = cv2.cvtColor(img_out.transpose(1, 2, 0), cv2.COLOR_RGB2BGR)
        img_out = (img_out*255).astype(np.uint8)
        cv2.imwrite(save_path, img_out)
    else:
        img_out = depth2img(img_out, img_out.max())
        cv2.imwrite(save_path.replace(".npy", ".png"), img_out)

names = ['{:0>10d}.png'.format(i) for i in range(args.start, args.end+1)]
source_paths = [os.path.join(args.source_dir, name) for name in names]
os.makedirs(args.output_dir, exist_ok=True)
output_paths = [os.path.join(args.output_dir, name) for name in names]
n_path = len(output_paths)

for i in tqdm(range(n_path)):
    save_output_whole(source_paths[i], output_paths[i])
