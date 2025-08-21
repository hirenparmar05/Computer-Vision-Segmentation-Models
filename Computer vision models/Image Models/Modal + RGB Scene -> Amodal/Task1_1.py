# Standard Library Imports
import os #
import shutil # provides high level file operations (copying, deleting, moving files/directories)
import tarfile # allows working with tar archives (compressed or uncompressed)
import random # provides functions for generating random numbers, shuffling squences, etc.
import cv2 # imports OpenCV Library functions used to read, display, or capture video (Face detection, object tracking, image processing)
import glob  # Finds files/paths matching specified patterns (like *.jpg)
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Hugging Face Hub
from huggingface_hub import HfApi, hf_hub_download # the api allows allows interaction with hugging face hub (upload/download models, datasets)
# the hub_download lets you download files from Hugging Face hub (model weights, datasets)
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
# PyTorch Ecosystem
import torch # Core PyTorch Library for tensor operations and nueral networks
from torch import nn # contains nueral network layers, loss functions, and utilities (nn.Linear, nn.ReLU)
from torch.utils.data import Dataset, DataLoader # Dataset: Abstract class for Custom Datasets // DataLoader: Efficient data loading/batching(supports multiprocessing)
import torchvision.transforms as transforms # preprocessing utilities for images (resizing, normalization, augmentation)
from torchvision.utils import make_grid # creates a grid of images (useful for visualing grids)
import torch.nn.functional as F # PyTorches functional interface for neural network operations
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Image and Numerical Processing
from PIL import Image # Python Imaging Library (Pillow) for image manipulation (open, save, resize, etc.)
import numpy as np # NumPy for numerical operations (arrays, linear algebra, etc.)
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Visualization
import matplotlib.pyplot as plt # Matplotlib for plotting graphs and displaying Images
from tabulate import tabulate # Pretty-print tabular data
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------

def compute_occlusion(rgba_path, seg_path):
    """Calculate precise occlusion percentage (0-1) between modal and amodal masks"""
    try:
        # Load masks with validation
        seg = cv2.imread(seg_path, cv2.IMREAD_GRAYSCALE)
        rgba = cv2.imread(rgba_path, cv2.IMREAD_UNCHANGED)

        if seg is None or rgba is None or rgba.shape[2] != 4:
            return None

        # Create binary masks
        modal_mask = (seg > 0).astype(np.uint8)
        amodal_mask = (rgba[:,:,3] > 0).astype(np.uint8)

        # Calculate occlusion ratio with edge case handling
        visible_pixels = np.sum(modal_mask)
        total_pixels = np.sum(amodal_mask)

        if total_pixels == 0:  # Invalid case (no object)
            return None

        occlusion = 1 - (visible_pixels / total_pixels)

        # Special handling for boundary cases
        if visible_pixels == 0:
            return 1.0  # 100% occluded
        if visible_pixels == total_pixels:
            return 0.0  # 0% occluded

        return occlusion

    except Exception as e:
        print(f"Error processing {seg_path}: {str(e)}")
        return None

def filter_scenes(root_dir, min_occ=0.25, max_occ=0.75):
    """Strictly filter scenes to only keep 25%-75% occlusion"""
    kept = removed = invalid = empty = 0

    for scene_dir in glob.glob(os.path.join(root_dir, "*")):
        if not os.path.isdir(scene_dir):
            continue

        scene_valid = True
        camera_dirs = list(glob.glob(os.path.join(scene_dir, "camera_*")))

        for cam_dir in camera_dirs:
            if not os.path.isdir(cam_dir):
                continue

            rgba_files = sorted(glob.glob(os.path.join(cam_dir, "rgba_*.png")))
            seg_files = sorted(glob.glob(os.path.join(cam_dir, "segmentation_*.png")))

            if not rgba_files or not seg_files:
                invalid += 1
                scene_valid = False
                break

            # Check multiple frames
            valid_frames = 0
            for rgba, seg in zip(rgba_files[:3], seg_files[:3]):
                seg_img = cv2.imread(seg, cv2.IMREAD_GRAYSCALE)
                rgba_img = cv2.imread(rgba, cv2.IMREAD_UNCHANGED)

                # Skip empty masks
                if np.sum(seg_img > 0) == 0 or np.sum(rgba_img[..., 3] > 0) == 0:
                    empty += 1
                    continue

                occ = compute_occlusion(rgba, seg)
                if occ is None or not (min_occ <= occ <= max_occ):
                    continue

                valid_frames += 1

            # Require at least 2 valid frames per camera
            if valid_frames < 2:
                scene_valid = False
                break

        if scene_valid:
            kept += len(camera_dirs)
        else:
            removed += len(camera_dirs)
            shutil.rmtree(scene_dir)

    print(f"\n=== Filter Results ===")
    print(f"Kept: {kept} cameras in valid scenes")
    print(f"Removed: {removed} cameras")
    print(f"Invalid: {invalid} (missing files)")
    print(f"Empty: {empty} (zero-pixel masks)")

def download_and_process(sample_pct=0.0125, min_occ=0.25, max_occ=0.75):
    """Download and process dataset with strict occlusion filtering"""
    api = HfApi()
    repo_id = "Amar-S/MOVi-MC-AC"
    os.makedirs("/content/data/train", exist_ok=True)
    os.makedirs("/content/data/test", exist_ok=True)

    def process_files(files, dest):
        for f in random.sample(files, max(1, int(len(files) * sample_pct))):
            try:
                path = hf_hub_download(repo_id=repo_id, filename=f, repo_type="dataset")
                dest_path = os.path.join(dest, os.path.basename(f))
                shutil.copy(path, dest_path)
                with tarfile.open(dest_path, 'r:gz') as tar:
                    tar.extractall(dest)
                os.remove(dest_path)
            except Exception as e:
                print(f"Error processing {f}: {e}")

    files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    train_files = [f for f in files if f.startswith("train/") and f.endswith(".tar.gz")]
    test_files = [f for f in files if f.startswith("test/") and f.endswith(".tar.gz")]

    process_files(train_files, "/content/data/train")
    process_files(test_files, "/content/data/test")

    print("Filtering training data...")
    filter_scenes("/content/data/train", min_occ, max_occ)

    print("\nFiltering test data...")
    filter_scenes("/content/data/test", min_occ, max_occ)

download_and_process(min_occ=0.25, max_occ=0.75)

# Load checkpoint
def load_checkpoint(filename, model, optimizer=None, device='cuda'):
    """Load model checkpoint"""
    checkpoint = torch.load(filename, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    epoch = checkpoint['epoch']
    train_metrics = checkpoint['train_metrics']
    val_metrics = checkpoint['val_metrics']

    print(f"Loaded checkpoint from epoch {epoch + 1}")
    return model, optimizer, epoch, train_metrics, val_metrics

# get_img_dict
# 1. Takes a list of images
# 2. Groups them by the first part of their filename (before the first underscore)
# 3. Stores these groups in a dictionary where:
#    - Keys are the image types (prefixes)
#    - Values are lists of all files sharing that prefix
# essentially sorting them alphebetically

def get_img_dict(img_dir): # Function call
    img_files = [x for x in img_dir.iterdir() if x.name.endswith('.png') or x.name.endswith('.tiff')] # img_files is the goes through the image directory and adds any .png or .tiff files into the img_files variables
    img_files.sort() # sorts to ensure consistent ordering

    img_dict = {} # dictionary to store grouped images by prefix

    for img_file in img_files:
        img_type = img_file.name.split('_')[0] # splits file names from cat_123 to cat 123 and takes the first index = cat and assigns it to img_type, in all cat_123.jpg becomes cat
        if img_type not in img_dict: # checks the dictionary to see if it is already in the dictionary
            img_dict[img_type] = [] # if not, it initializes it with an empty list as its value
        img_dict[img_type].append(img_file) # if it is, it adds to that associated list with its img_type in the dictionary

    return img_dict # returns the dictionary as output

# get_sample_dict

def get_sample_dict(sample_dir): # Function call


    camera_dirs = [x for x in sample_dir.iterdir() if 'camera' in x.name] # get all directories with camera in their name only (camera1, camera2, ...)
    camera_dirs.sort() # again, sorts for consistent ordering

    sample_dict = {} # Top level dictionary to story camera-wise data

    for cam_dir in camera_dirs: # for each cam_directory in camera directories
        cam_dict = {} # Dictionary for this specific camera
        cam_dict['scene'] = get_img_dict(cam_dir) # groups scene images by prefix

        obj_dirs = [x for x in cam_dir.iterdir() if 'obj_' in x.name] # get all object directories (obj_0001, obj_0002, ...)
        obj_dirs.sort() # sorts for consistent ordering

        for obj_dir in obj_dirs: # for each object directory in object directories
            cam_dict[obj_dir.name] = get_img_dict(obj_dir) # group images in this object directory by prefix and store under the objects name

        sample_dict[cam_dir.name] = cam_dict # add this cameras data to the sample_dict

    return sample_dict # returns a nested dictionary like: {'camera': {'scene': {...}, 'obj_1': ...}}

# make_obj_viz --> Video
# make_vid --> Video
class ModalAmodalDataset(Dataset):
    @staticmethod
    def get_default_transform(img_size):
        return transforms.Compose([transforms.Resize(img_size), transforms.ToTensor(),])

    def __init__(self, root_dir, split, transform=None, img_size=(256, 256)):
        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size
        self.transform = transform or self.get_default_transform(img_size)
        self.samples = self._build_sample_index()

    def _build_sample_index(self):
            samples = []
            split_dir = os.path.join(self.root_dir, self.split)

            with os.scandir(split_dir) as scene_entries:
                for scene_entry in scene_entries:
                    if not scene_entry.is_dir():
                        continue

                    with os.scandir(scene_entry.path) as camera_entries:
                        for camera_entry in camera_entries:
                            if not camera_entry.is_dir() or not camera_entry.name.startswith('camera_'):
                                continue

                            # Get all RGBA images
                            rgba_files = [f.path for f in os.scandir(camera_entry.path)
                                        if f.name.startswith('rgba_') and f.name.endswith('.png')]

                            for obj_entry in os.scandir(camera_entry.path):
                                if not obj_entry.is_dir() or not obj_entry.name.startswith('obj_'):
                                    continue

                                try:
                                    obj_id = int(obj_entry.name.split('_')[1])
                                except:
                                    continue

                                for rgba_file in rgba_files:
                                    frame_name = os.path.basename(rgba_file)[5:-4]  # removes 'rgba_' and '.png'
                                    seg_file = os.path.join(camera_entry.path, f'segmentation_{frame_name}.png')
                                    amodal_file = os.path.join(obj_entry.path, f'segmentation_{frame_name}.png')

                                    if os.path.exists(seg_file) and os.path.exists(amodal_file):
                                        samples.append({
                                            'rgb_path': rgba_file,
                                            'segmentation_path': seg_file,
                                            'amodal_path': amodal_file,
                                            'object_id': obj_id,
                                            'frame_id': frame_name,
                                            'scene': scene_entry.name,
                                            'camera': camera_entry.name
                                        })
            return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        max_attempts = 5  # Maximum tries to find valid sample
        attempt = 0

        while attempt < max_attempts:
            sample = self.samples[idx]

            # Load images
            rgb_image = Image.open(sample['rgb_path']).convert('RGB')
            panoptic_seg = Image.open(sample['segmentation_path'])

            # Create modal mask
            modal_mask = (np.array(panoptic_seg) == sample['object_id']).astype(np.uint8) * 255
            modal_mask = Image.fromarray(modal_mask)

            # Load amodal mask
            amodal_mask = Image.open(sample['amodal_path']).convert('L')
            amodal_mask = amodal_mask.point(lambda x: 255 if x > 128 else 0)

            # Apply transforms
            rgb_tensor = self.transform(rgb_image)
            modal_tensor = self.transform(modal_mask)[:1]
            amodal_tensor = self.transform(amodal_mask)[:1]

            # Check for empty masks
            modal_pixels = torch.sum(modal_tensor > 0.5)
            amodal_pixels = torch.sum(amodal_tensor > 0.5)

            if modal_pixels == 0 and amodal_pixels == 0:
                # Skip this sample and try another
                idx = random.randint(0, len(self)-1)
                attempt += 1
                continue

            return {
                'rgb': rgb_tensor,
                'modal_mask': modal_tensor,
                'amodal_mask': amodal_tensor,
                'object_id': sample['object_id'],
                'frame_id': sample['frame_id'],
                'scene': sample['scene'],
                'camera': sample['camera'],
                'amodal_path': sample['amodal_path']
            }

        # If all attempts fail, return first sample
        return self.__getitem__(0)

def create_dataloader(root_dir, split, batch_size=4, shuffle=True, num_workers=4, img_size=(224, 224)):
    dataset = ModalAmodalDataset(root_dir=root_dir, split=split, img_size=img_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=torch.cuda.is_available())
class conv2d_inplace_spatial(nn.Module):
    """Double convolution block with optional pooling"""
    def __init__(self, in_channels, out_channels, pooling_function=None, activation=nn.GELU(), kernel_size=3, padding=1):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            activation,
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            activation,
        )
        self.pooling = pooling_function if isinstance(pooling_function, nn.Module) else None

    def forward(self, x):
        x = self.double_conv(x)
        if self.pooling is not None:
            x = self.pooling(x)
        return x

class Upscale(nn.Module):
    def __init__(self, scale_factor=2, mode='bilinear', align_corners=False):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor,
                           mode=self.mode, align_corners=self.align_corners)

class Unet_Image(nn.Module):
    def __init__(self, in_channels=4):
        super().__init__()
        # Encoder Path
        self.mpool_2 = nn.MaxPool2d(2)
        self.down1 = conv2d_inplace_spatial(in_channels, 32, self.mpool_2)
        self.down2 = conv2d_inplace_spatial(32, 64, self.mpool_2)
        self.down3 = conv2d_inplace_spatial(64, 128, self.mpool_2)
        self.down4 = conv2d_inplace_spatial(128, 256, self.mpool_2)
        self.down5 = conv2d_inplace_spatial(256, 512)  # Bottleneck

        # Decoder Path with Upscale
        self.upscale_2 = Upscale(scale_factor=2)
        self.up1 = conv2d_inplace_spatial(512 + 256, 256, self.upscale_2)
        self.up2 = conv2d_inplace_spatial(256 + 128, 128, self.upscale_2)
        self.up3 = conv2d_inplace_spatial(128 + 64, 64, self.upscale_2)
        self.up4 = conv2d_inplace_spatial(64 + 32, 32, self.upscale_2)

        # Final output
        self.final_conv = nn.Sequential(nn.Conv2d(32, 1, kernel_size=1), nn.Sigmoid())

    def encode(self, x):
        """Encoder with skip connections"""
        x1 = self.down1(x)  # 32
        x2 = self.down2(x1) # 64
        x3 = self.down3(x2) # 128
        x4 = self.down4(x3) # 256
        x5 = self.down5(x4) # 512
        return x1, x2, x3, x4, x5

    def decode(self, x1, x2, x3, x4, x5):
        """Decoder using Upscale module"""
        x = self.up1(torch.cat([x5, x4], dim=1))  # 512+256 -> 256
        x = self.up2(torch.cat([x, x3], dim=1))   # 256+128 -> 128
        x = self.up3(torch.cat([x, x2], dim=1))    # 128+64 -> 64
        x = self.up4(torch.cat([x, x1], dim=1))    # 64+32 -> 32
        return self.final_conv(x)

    def forward(self, batch, bce_weight=0.5, dice_weight=0.5):
        """Forward pass with input validation and weighted losses"""
        # Input validation
        assert isinstance(batch, dict), "Input must be a dictionary"
        assert all(k in batch for k in ['rgb', 'modal_mask', 'amodal_mask']), "Missing required keys"
        assert batch['rgb'].shape[1] == 3, "RGB input must have 3 channels"
        assert batch['modal_mask'].shape[1] == 1, "Modal mask must be single channel"

        # Model forward pass
        modal_input = torch.cat((batch['rgb'], batch['modal_mask']), dim=1)
        amodal_mask_labels = batch['amodal_mask'].float()
        pred_mask = self.decode(*self.encode(modal_input))

        # Loss calculation
        bce_loss = F.binary_cross_entropy(pred_mask, amodal_mask_labels)

        # Dice loss (direct calculation)
        smooth = 1.0
        pred_flat = pred_mask.view(-1)
        target_flat = amodal_mask_labels.view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)

        # Weighted total loss
        total_loss = bce_weight * bce_loss + dice_weight * dice_loss

        # Metrics
        metrics = {
            'loss': total_loss.item(),
            'bce': bce_loss.item(),
            'dice': 1 - dice_loss.item(),
            'iou': (intersection + smooth) / ((pred_flat + target_flat).sum() - intersection + smooth).item()
        }

        return total_loss, metrics, batch

def batch_to_device(batch, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

def aggregate_metrics(metrics_list):
    return {k: sum(m[k] for m in metrics_list) / len(metrics_list) for k in metrics_list[0]}

def train_step(batch, model, optimizer, bce_weight=0.5, dice_weight=0.5):
    model.train()
    optimizer.zero_grad()
    total_loss, metrics, _ = model(batch, bce_weight=bce_weight, dice_weight=dice_weight)  # Updated
    total_loss.backward()
    optimizer.step()
    return total_loss, metrics

def val_step(batch, model, bce_weight=0.5, dice_weight=0.5):
    model.eval()
    with torch.no_grad():
        total_loss, metrics, batch = model(batch, bce_weight=bce_weight, dice_weight=dice_weight)  # Updated
    return total_loss, metrics, batch

def run_epoch(model, dataloader, device, optimizer=None, bce_weight=0.5, dice_weight=0.5):  # Added params
    metrics_list = []
    sample_batch = None

    for i, batch in enumerate(dataloader):
        batch = batch_to_device(batch, device)

        if optimizer is not None:
            loss, metrics = train_step(batch, model, optimizer, bce_weight, dice_weight)  # Updated
        else:
            loss, metrics, batch = val_step(batch, model, bce_weight, dice_weight)  # Updated
            if i == 0:
                sample_batch = batch

        metrics_list.append(metrics)

    return aggregate_metrics(metrics_list), sample_batch

def visualize_results(sample, model, epoch):
    model.eval()
    with torch.no_grad():
        # Prepare sample batch as dictionary (consistent with forward())
        sample_dict = {
            'rgb': sample['rgb'][0].unsqueeze(0).to(device),
            'modal_mask': sample['modal_mask'][0].unsqueeze(0).to(device),
            'amodal_mask': sample['amodal_mask'][0].unsqueeze(0).to(device)
        }

        # Create 4-channel input (RGB + modal mask)
        model_input = torch.cat([sample_dict['rgb'], sample_dict['modal_mask']], dim=1)

        # Get encoder features (x1-x5)
        x1, x2, x3, x4, x5 = model.encode(model_input)

        # Decode with skip connections
        pred_mask = model.decode(x1, x2, x3, x4, x5)

        # Prepare visualization
        rgb = sample_dict['rgb'].squeeze().permute(1,2,0).cpu().numpy()
        modal = sample_dict['modal_mask'].squeeze().cpu().numpy()
        pred = pred_mask.squeeze().cpu().numpy() > 0.5  # Apply threshold
        gt = sample_dict['amodal_mask'].squeeze().cpu().numpy()

        # Visualization
        fig, ax = plt.subplots(2, 2, figsize=(5, 5))
        titles = ['RGB Input', 'Modal Mask', 'Predicted Amodal', 'Ground Truth']
        images = [rgb, modal, pred, gt]

        for i, (ax, title, img) in enumerate(zip(ax.flat, titles, images)):
            ax.imshow(img, cmap='gray' if i > 0 else None)
            ax.set_title(title)
            ax.axis('off')

        plt.suptitle(f'Epoch {epoch+1} Results')
        plt.tight_layout()
        plt.show()

def train(model, optimizer, train_loader, val_loader, epochs, device, bce_weight=0.5, dice_weight=0.5, save_path='model_checkpoint.pth'):
    train_metrics = {'loss': [], 'iou': [], 'dice': []}
    val_metrics = {'loss': [], 'iou': [], 'dice': []}

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1} / {epochs}")

        # Training
        model.train()
        train_epoch_metrics, _ = run_epoch(model, train_loader, device, optimizer, bce_weight, dice_weight)

        # Validation
        model.eval()
        val_epoch_metrics, sample_batch = run_epoch(model, val_loader, device, None, bce_weight, dice_weight)

        # Store metrics
        for k in train_metrics:
            train_metrics[k].append(train_epoch_metrics[k])
            val_metrics[k].append(val_epoch_metrics[k])

        print(f"Train Loss: {train_epoch_metrics['loss']:.4f} | Val Loss: {val_epoch_metrics['loss']:.4f}")
        print(f"Train IOU: {train_epoch_metrics['iou']:.4f} | Val IOU: {val_epoch_metrics['iou']:.4f}")
        print(f"Train Dice: {train_epoch_metrics['dice']:.4f} | Val Dice: {val_epoch_metrics['dice']:.4f}")

        if epoch % 1 == 0:
            visualize_results(sample_batch, model, epoch)

        # Save checkpoint every epoch (or adjust frequency as needed)
        save_checkpoint(model, optimizer, epoch, train_metrics, val_metrics, save_path)

    return train_metrics, val_metrics

# Arguments
learning_rate = 3e-4
batch_size = 64
n_workers = 2
n_epochs = 20
img_size = (256, 256)
bce_weight = 0.5
dice_weight = 0.5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Data
train_loader = create_dataloader(root_dir='/content/data', split='train', batch_size=batch_size, num_workers=n_workers, img_size=img_size)
val_loader = create_dataloader(root_dir='/content/data', split='test', batch_size=batch_size, num_workers=n_workers, img_size=img_size)

# save function to .pth file
def save_checkpoint(model, optimizer, epoch, train_metrics, val_metrics, filename):
    """Save model checkpoint with all relevant information"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'model_config': {'in_channels': model.in_channels if hasattr(model, 'in_channels') else 4,}
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved to {filename}")

# Model
model = Unet_Image(in_channels=4).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train
train_metrics, val_metrics = train(
    model=model,
    optimizer=optimizer,
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=n_epochs,
    device=device,
    bce_weight=bce_weight,
    dice_weight=dice_weight,
    save_path='best_model.pth'
)