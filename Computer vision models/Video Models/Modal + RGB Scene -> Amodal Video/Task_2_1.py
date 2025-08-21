#pip install torch torchvision matplotlib av pytorch_msssim

# PyTorch, Torchvision
import torch
from torch import nn
import torchvision
from torchvision.transforms import ToPILImage, ToTensor
from torchvision.utils import make_grid
from torchvision.io import write_video
from torch.utils.data import Dataset
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, utils
from PIL import Image

# Common
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import random
import json
from IPython.display import Video
import tarfile
import glob
from tqdm import tqdm
from PIL import Image
import io
import cv2

# Utils from Torchvision
tensor_to_image = ToPILImage()
image_to_tensor = ToTensor()

def get_img_dict(img_dir):
    img_files = [x for x in img_dir.iterdir() if x.name.endswith('.png') or x.name.endswith('.tiff')]
    img_files.sort()

    img_dict = {}
    for img_file in img_files:
        img_type = img_file.name.split('_')[0]
        if img_type not in img_dict:
            img_dict[img_type] = []
        img_dict[img_type].append(img_file)
    return img_dict


def get_sample_dict(sample_dir):

    camera_dirs = [x for x in sample_dir.iterdir() if 'camera' in x.name]
    camera_dirs.sort()

    sample_dict = {}

    for cam_dir in camera_dirs:
        cam_dict = {}
        cam_dict['scene'] = get_img_dict(cam_dir)

        obj_dirs = [x for x in cam_dir.iterdir() if 'obj_' in x.name]
        obj_dirs.sort()

        for obj_dir in obj_dirs:
            cam_dict[obj_dir.name] = get_img_dict(obj_dir)

        sample_dict[cam_dir.name] = cam_dict

    return sample_dict


def make_obj_viz(cam_dict, cam_num=0):

    n_frames = 24
    n_cols = 6

    all_obj_ids = [x for x in sample_dict['camera_0000'].keys() if 'obj_' in x]
    obj_id_str = random.sample(all_obj_ids, k=1)[0]
    obj_id_int = int(obj_id_str.split('_')[1])

    grid_tensors = []
    for i in range(n_frames):
        grid = []
        scene_rgb_tensor = image_to_tensor(Image.open(cam_dict['scene']['rgba'][i]).convert('RGB'))
        grid.append(scene_rgb_tensor)
        scene_masks_tensor = image_to_tensor(Image.open(cam_dict['scene']['segmentation'][i]).convert('RGB'))
        grid.append(scene_masks_tensor)

        scene_masks_p = Image.open(cam_dict['scene']['segmentation'][i])
        scene_masks_p_tensor = torch.tensor(np.array(scene_masks_p))
        obj_modal_tensor = (scene_masks_p_tensor==obj_id_int)
        blended_obj_modal_tensor = scene_masks_tensor*obj_modal_tensor
        grid.append(blended_obj_modal_tensor)

        obj_amodal_tensor = image_to_tensor(Image.open(cam_dict[obj_id_str]['segmentation'][i]).convert('RGB'))
        blended_obj_amodal_tensor = blended_obj_modal_tensor + (obj_amodal_tensor != obj_modal_tensor)
        grid.append(blended_obj_amodal_tensor)

        obj_rgb_tensor = image_to_tensor(Image.open(cam_dict[obj_id_str]['rgba'][i]).convert('RGB'))
        grid.append(obj_rgb_tensor)

        blended_scene_obj_tensor = (scene_rgb_tensor/3 + 2*blended_obj_amodal_tensor/3)
        grid.append(blended_scene_obj_tensor)

        grid_tensors.append(make_grid(grid, nrow=n_cols, padding=2, pad_value=127))

    return grid_tensors


def make_vid(grid_tensors, save_path):
    vid_tensor = torch.stack(grid_tensors, dim=1).permute(1, 2, 3, 0)
    vid_tensor = (vid_tensor*255).long()
    write_video(save_path, vid_tensor, fps=5, options={'crf':'20'})


''' Code to download files from the MOVi-MC-AC Dataset
!wget https://huggingface.co/datasets/Amar-S/MOVi-MC-AC/resolve/main/test_obj_descriptors.json
#Download Descriptors, Readme, etc.
!wget https://huggingface.co/datasets/Amar-S/MOVi-MC-AC/resolve/main/train_obj_descriptors.json
!wget https://huggingface.co/datasets/Amar-S/MOVi-MC-AC/resolve/main/ex_vis.mp4
!wget https://huggingface.co/datasets/Amar-S/MOVi-MC-AC/resolve/main/README.md
!wget "https://huggingface.co/datasets/Amar-S/MOVi-MC-AC/resolve/main/Notice%201%20-%20Unlimited_datasets.pdf"
!wget https://huggingface.co/datasets/Amar-S/MOVi-MC-AC/resolve/main/.gitattributes
#Test to see if you are on the right huggingface repo
from huggingface_hub import HfApi, hf_hub_download
import random, os
api = HfApi()
repo_id = "Amar-S/MOVi-MC-AC"
# # List all files in the repo
files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
# # Separate train and test files
train_files = [f for f in files if f.startswith("train/") and not f.endswith(".json")]
test_files = [f for f in files if f.startswith("test/") and not f.endswith(".json")]
print(f"Found {len(train_files)} train files and {len(test_files)} test files.")
#Download 4% of Train/Test files
import os
import random
import shutil
from huggingface_hub import hf_hub_download
os.makedirs("/content/data/train", exist_ok=True)
os.makedirs("/content/data/test", exist_ok=True)
# # Sample 4% of each split (as you were doing)
subset_train = random.sample(train_files, int(len(train_files) * 0.015))
subset_test = random.sample(test_files, int(len(test_files) * 0.015))
# # Download the training files (uncomment and fix)
for file in subset_train:
    out_path = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=file)
    dest_path = f"/content/data/train/{os.path.basename(file)}"
    shutil.copyfile(out_path, dest_path)  # COPY the actual file content instead of renaming symlink
# # Download the test files
for file in subset_test:
    out_path = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=file)
    dest_path = f"/content/data/test/{os.path.basename(file)}"
    shutil.copyfile(out_path, dest_path)  # COPY the actual file content here as well



def extract_files(path=''):
    root = Path(path)
    for archive in root.rglob("*.tar.gz"):
        extract_path = archive.parent / archive.stem.replace(".tar", "")
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(path=extract_path)
'''


def get_all_samples(root_dir):
    root = Path(root_dir)
    sample_dict = {}

    for cam_dir in root.rglob("camera_*"):
        if cam_dir.is_dir():
            rgba_imgs = sorted(cam_dir.glob("rgba_*.png"))
            segm_imgs = sorted(cam_dir.glob("segmentation_*.png"))

            if len(rgba_imgs) == 0 or len(segm_imgs) == 0:
                print(f"Skipping {cam_dir} — missing or empty rgba/segmentation folders")
                continue

            scene_id = cam_dir.parents[1].name  # e.g. data/train/<scene>/<scene>/camera_xxxx
            cam_id = cam_dir.name

            if scene_id not in sample_dict:
                sample_dict[scene_id] = {}

            cam_dict = {
                'rgba': rgba_imgs,
                'segmentation': segm_imgs,
            }

            # Add all obj_XXXX folders
            for obj_dir in sorted(cam_dir.glob("obj_*")):
                cam_dict[obj_dir.name] = {
                    'segmentation': sorted((obj_dir).glob("segmentation*.png")),
                    'rgba': sorted((obj_dir).glob("rgba*.png"))
                }

            sample_dict[scene_id][cam_id] = cam_dict

    print(f"Loaded {len(sample_dict)} scenes from {root_dir}")
    return sample_dict


class WindowedModalMaskDataset(Dataset):
    def __init__(self, root_dir, window_size=5):
        self.sample_dict = get_all_samples(root_dir)
        self.entries = []
        self.window_size = window_size

        for scene_id, cams in self.sample_dict.items():
            for cam_id, data in cams.items():
                num_frames = len(data['rgba'])
                for start_idx in range(num_frames - window_size + 1):
                    self.entries.append((scene_id, cam_id, start_idx))

        print(f"Total dataset size (windows): {len(self.entries)} samples")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        scene_id, cam_id, start_idx = self.entries[idx]
        paths = self.sample_dict[scene_id][cam_id]
        T = self.window_size

        scene_frames = []
        amodal_frames = []

        rand_obj_id = None  # will be selected on first frame

        for t in range(T):
            frame_idx = start_idx + t

            # Load RGB image
            scene_img = Image.open(paths['rgba'][frame_idx]).convert('RGB')
            scene_tensor = image_to_tensor(scene_img)  # [3, H, W]

            # Load object segmentation mask (same as used for modal)
            obj_mask = np.array(Image.open(paths['segmentation'][frame_idx]))
            obj_mask_tensor = torch.tensor(obj_mask, dtype=torch.int64)  # [H, W]

            # Choose the object once, from first frame
            if t == 0:
                unique_ids = torch.unique(obj_mask_tensor)
                unique_objects = unique_ids[unique_ids != 0]  # exclude background
                if len(unique_objects) == 0:
                    raise ValueError(f"No objects in segmentation mask at frame {frame_idx}!")
                rand_obj_id = random.choice(unique_objects.tolist())

            # Compute modal mask for selected object
            modal_mask = (obj_mask_tensor == rand_obj_id).float().unsqueeze(0)  # [1, H, W]

            # Combine RGB + modal mask
            scene_with_mask = torch.cat((scene_tensor, modal_mask), dim=0)  # [4, H, W]
            scene_frames.append(scene_with_mask)  # list of [4, H, W]

            # Load amodal mask (per-object segmentation from separate path, assumed same format)
            amodal_mask = Image.open(paths[f'obj_{rand_obj_id:04d}']['segmentation'][frame_idx])  # reuse same mask
            #amodal_mask_tensor = (torch.tensor(amodal_mask_arr, dtype=torch.float32) == rand_obj_id).float().unsqueeze(0)  # [1, H, W]
            amodal_mask_tensor = image_to_tensor(amodal_mask)  # [1, H, W]
            amodal_frames.append(amodal_mask_tensor)

        # Stack all frames: output [T, 4, H, W] and [T, 1, H, W]
        modal = torch.stack(scene_frames, dim=0)    # [T, 4, H, W]
        amodal = torch.stack(amodal_frames, dim=0)  # [T, 1, H, W]

        return modal, amodal


class conv2d_inplace_spatial(nn.Module):
    def __init__(self, in_channels, out_channels, pooling_function, activation = nn.GELU()):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(out_channels),
            activation,
            pooling_function,
        )

    def forward(self, x):
        return self.double_conv(x)


class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, 1),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, 1),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, g):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class Upscale(nn.Module):
    def __init__(self, scale_factor=(2, 2), mode='bilinear', align_corners=False):
        super(Upscale, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners)


class Unet_Image(nn.Module):
    def __init__(self, in_channels = 4, mask_content_preds = False):
        super().__init__()

        self.mpool_2 = nn.MaxPool2d((2, 2))

        self.down1 = conv2d_inplace_spatial(in_channels, 32, self.mpool_2)
        self.down2 = conv2d_inplace_spatial(32, 64, self.mpool_2)
        self.down3 = conv2d_inplace_spatial(64, 128, self.mpool_2)
        self.down4 = conv2d_inplace_spatial(128, 256, self.mpool_2)

        self.upscale_2 = Upscale(scale_factor=(2, 2), mode='bilinear', align_corners=False)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 64, 1), nn.ReLU()
        )
        self.lstm = nn.LSTM(input_size=64*16*16, hidden_size=512, batch_first=True, bidirectional=True)
        self.lstm_proj = nn.Linear(1024, 256 * 16 * 16)

        self.up1 = conv2d_inplace_spatial(256, 128, self.upscale_2)
        self.up2 = conv2d_inplace_spatial(256, 64, self.upscale_2)
        self.up3 = conv2d_inplace_spatial(128, 32, self.upscale_2)

        self.atten_gate2 = AttentionGate(128, 128, 64)
        self.atten_gate1 = AttentionGate(64, 64, 32)
        self.atten_gate0 = AttentionGate(32, 32, 16)

        self.up4_amodal_content = conv2d_inplace_spatial(64, 1, self.upscale_2, activation = nn.Identity())

    def encode_frame(self, x):
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x4_bottleneck = self.bottleneck(x4)
        return x1, x2, x3, x4_bottleneck# [B, T, 3, H, W]

    def decode(self, h1, h2, h3, h4):
        h4 = self.up1(h4) # 6, 256, 1, 16, 16 -> 6, 128, 1, 32, 32 (double spatial, then conv-in-place channels to half)
        h3 = self.atten_gate2(h3, h4)
        h34 = torch.cat((h3, h4), dim = 1) # (6, 2*128, 1, 32, 32)

        h34 = self.up2(h34) # 6, 256, 1, 32, 32 -> 6, 128, 2, 64, 64
        h34 = self.atten_gate1(h2, h34)
        h234 = torch.cat((h2, h34), dim = 1) # (6, 2*128, )

        h234 = self.up3(h234)
        h234 = self.atten_gate0(h1, h234)
        h1234 = torch.cat((h1, h234), dim = 1)

        #logits_amodal_mask = self.up4_amodal_mask(h1234)
        logits_amodal_content = self.up4_amodal_content(h1234)
        return logits_amodal_content

    def forward(self, x):  # x: [B, T, C, H, W]
        B, T, C, H, W = x.shape
        lstm_inputs = []
        skip_connections = []

        for t in range(T):
            x1, x2, x3, x4 = self.encode_frame(x[:, t])
            skip_connections.append((x1, x2, x3))
            lstm_inputs.append(x4.flatten(1))  # [B, 64*16*16]

        lstm_in = torch.stack(lstm_inputs, dim=1)  # [B, T, feat_dim]
        lstm_out, _ = self.lstm(lstm_in)  # [B, T, 1024]
        lstm_out = self.lstm_proj(lstm_out).view(B, T, 256, 16, 16)

        # Decode for each frame
        output_frames = []
        for t in range(T):
            x1, x2, x3 = skip_connections[t]
            decoded = self.decode(x1, x2, x3, lstm_out[:, t])
            output_frames.append(decoded)

        return torch.stack(output_frames, dim=1)


def draw_amodal_boundary(rgb_image, amodal_mask, color=(255, 0, 255)):
    """
    Draws an outline of the amodal mask on top of the RGB image.
    Assumes rgb_image is in [H, W, 3] and amodal_mask is [H, W].
    """
    contours, _ = cv2.findContours(amodal_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    outlined = cv2.drawContours(rgb_image.copy(), contours, -1, color, thickness=2)
    return outlined


def save_sequence_gif(output, input, target, gif_path='output.gif', fps=2, frame_idx=0):
    output = output.detach().cpu()
    input = input.detach().cpu()
    target = target.detach().cpu()

    _, T, _, H, W = output.shape
    frames = []

    for t in range(T):
        fig, axs = plt.subplots(1, 4, figsize=(12, 3))

        # Get RGB image
        rgb = input[frame_idx, t, :3].permute(1, 2, 0).numpy()  # shape: [H, W, 3]
        rgb = (rgb * 255).astype(np.uint8) if rgb.max() <= 1.0 else rgb.astype(np.uint8)

        # Get GT amodal mask
        gt_mask = target[frame_idx, t, 0].numpy()
        gt_mask = (gt_mask > 0.5).astype(np.uint8)  # Binarize

        # Draw outline
        rgb_outlined = draw_amodal_boundary(rgb, gt_mask)

        axs[0].imshow(rgb_outlined)
        axs[0].set_title("RGB + GT Amodal Outline")

        # Modal mask
        modal = input[frame_idx, t, 3]
        axs[1].imshow(modal, cmap='gray')
        axs[1].set_title("Modal Mask")

        # Predicted amodal
        pred = output[frame_idx, t, 0]
        axs[2].imshow(pred, cmap='gray')
        axs[2].set_title("Predicted Amodal")

        # Ground truth amodal
        axs[3].imshow(gt_mask, cmap='gray')
        axs[3].set_title("GT Amodal")

        for ax in axs:
            ax.axis('off')
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        frame = Image.open(buf)
        frames.append(frame)

    frames[0].save(
        gif_path, save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0
    )
    print(f"Saved GIF to {gif_path}")


train_dataset = WindowedModalMaskDataset('data/train', window_size=10)
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle = True)
val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

loss_fn = nn.BCEWithLogitsLoss()

model = Unet_Image(4).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
step = 0
for epoch in range(30):
    model.train()
    step += 1
    for data, target in train_dataloader:
        data, target = data.to(device), target.to(device)

        fake_video = model(data)

        recon_loss = loss_fn(fake_video, target)

        optimizer.zero_grad()
        recon_loss.backward()
        optimizer.step()

    #Show predictions from last trianing run
    #fake_video = torch.sigmoid(fake_video).round()
    #gif = f'train_epoch_{step:03d}.gif'
    #save_sequence_gif(fake_video, data, target, gif_path=gif, fps=6, frame_idx=0)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
      for data,target in val_dataloader:
        data, target = data.to(device), target.to(device)

        fake_video = model(data)

        recon_loss = loss_fn(fake_video, target)
        val_loss += recon_loss.item()

    #fake_video = torch.sigmoid(fake_video).round()
    #gif = f'train_epoch_{step:03d}.gif'
    #save_sequence_gif(fake_video, data, target, gif_path=gif, fps=6, frame_idx=0)
    print(f"Epoch {epoch+1} - Val Loss: {val_loss/len(val_dataloader):.4f}")

#Testing
test_dataset = WindowedModalMaskDataset('data/test', window_size=15)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

model.eval()
step = 0
with torch.no_grad():
    for data, target in test_dataloader:
        step += 1
        data, target = data.to(device), target.to(device)
        fake_video = model(data)
        fake_video = torch.sigmoid(fake_video).round()
        gif = f'output{step:04d}.gif'
        #if step % 50 == 0:
        save_sequence_gif(fake_video, data, target, gif_path=gif, fps=6, frame_idx=0)
        #break