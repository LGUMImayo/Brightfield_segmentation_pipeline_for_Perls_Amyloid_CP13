# --- CRITICAL FIX: MUST BE FIRST ---
# Force safe integer limits before any library loads MONAI
# -----------------------------------

import os
import json
import csv
import sys
import numpy as np
import torch
import glob
import pytorch_lightning as pl
import torchvision.utils
from torchvision.transforms import ToPILImage

from torchviz import make_dot
import neptune
from neptune.types import File

import torchmetrics
from monai.data import list_data_collate
from monai.networks.nets import UNet, SwinUNETR, FlexibleUNet
from monai.networks.layers import Norm
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss, FocalLoss, TverskyLoss  # Import FocalLoss and TverskyLoss

from skimage import io, color, exposure
from skimage.color import rgb2gray, rgb2lab
from skimage.morphology import disk, dilation

from PIL import Image
import torch
import pytorch_lightning as pl

from torch.utils.data import Dataset
import torchmetrics

from monai.transforms import (
    MapTransform,
    # AddChanneld, # deprecated in latest version of monai 
    # AddChannel,
    # AsChannelFirstd,
    # AsChannelLastd,
    Compose,
    CastToTyped,
    MapLabelValued,
    SqueezeDimd,
    Resized,
    CenterSpatialCropd,
    RandFlipd,
    RandAffined,
    ScaleIntensityRanged,
    RandGaussianNoised,
    RandAdjustContrastd,
    RandGaussianSmoothd,
    RandStdShiftIntensityd,
    RandScaleIntensityd,  # <-- ADD THIS
    RandRotate90d,        # <-- AND THIS
    EnsureTyped,
    ScaleIntensityRange,
    EnsureType,
    SpatialPadd,
    Lambda,
    RandBiasFieldd,       # <--- CHANGED: Added dictionary version
    RandShiftIntensityd,  # <--- CHANGED: Added dictionary version
    RandHistogramShiftd,  # <--- CHANGED: Added dictionary version
    # Removed array versions to prevent confusion
    # RandBiasField,
    # RandAdjustContrast,
    # RandShiftIntensity,
    # RandHistogramShift,
    # RandGaussianNoise
)

# Import parent modules
try:
    from .utils import get_weights, extract_largest_component_bbox_image, MapImage
    from .utils import get_image_paths, contrast_img
except ImportError:
    from utils import get_weights, extract_largest_component_bbox_image, MapImage
    from utils import get_image_paths, contrast_img


def dynamic_scale(image: np.ndarray) -> np.ndarray:
    """Scales an input image by determining the maximum and minimum pixel values using monai's transform method ScaleIntensityRange.

    Args:
        image (np.ndarray): Input image to be scaled 

    Returns:
        np.ndarray: Scaled image.
    """
    a_min, a_max = image.min(), image.max()
    transform = ScaleIntensityRange(
        a_min=a_min,
        a_max=a_max,
        b_min=0.0,
        b_max=1.0,
        clip=True,
    )
    if image.max() > 255:
        print(image.max())
    return transform(image)

def fixed_scale(image: np.ndarray) -> np.ndarray:
    """Scales an input image by dividing by 255.0 to preserve absolute intensity.
    
    Args:
        image (np.ndarray): Input image to be scaled 

    Returns:
        np.ndarray: Scaled image.
    """
    return image.astype(np.float32) / 255.0


class tiff_reader(MapTransform):
    """
    Custom TIFF reader to preprocess and apply optional contrast or bounding box extraction.
    """
    def __init__(self, image_col=None, boundingbox=True, dilation=True, disk_dilation=3, scaling_method="dynamic", contrast_enhance=True, keys=["image", "label"], *args, **kwargs):
        super().__init__(keys, *args, **kwargs)
        self.keys = keys
        self.image_col = image_col
        self.boundingbox = boundingbox
        self.dilation = dilation
        self.disk_dilation = disk_dilation
        self.scaling_method = scaling_method
        self.contrast_enhance = contrast_enhance

    def __call__(self, data_dict):
        
        # Select scaling function
        scale_func = fixed_scale if self.scaling_method == "fixed" else dynamic_scale

        d = dict(data_dict)
        
        # 1. Load Image
        # Assuming the image path is in the dictionary under the first key (usually 'image')
        image_path = d[self.keys[0]]
        
        # Handle different input types (path string or already loaded array)
        if isinstance(image_path, str):
            try:
                img = io.imread(image_path)
            except Exception as e:
                print(f"Error reading image {image_path}: {e}")
                # Return a dummy image (black) of the correct size to prevent crashing
                # Assuming standard patch size 256x256x3
                img = np.zeros((256, 256, 3), dtype=np.uint8)
        else:
            img = image_path

        # 2. Bounding Box Extraction (Crucial for removing empty background)
        # We do this BEFORE scaling to ensure thresholding works on raw data
        if self.boundingbox:
            # This function crops the image to the largest tissue component
            # It returns the cropped image and the bbox coordinates
            # We only need the image here for training
            img, bbox = extract_largest_component_bbox_image(img)
            
            # If we have a label, we must crop it using the SAME bbox
            if len(self.keys) > 1 and self.keys[1] in d:
                label_path = d[self.keys[1]]
                if isinstance(label_path, str):
                    lbl = io.imread(label_path)
                else:
                    lbl = label_path
                
                # Apply the same crop to the label
                if bbox is not None:
                    minr, minc, maxr, maxc = bbox
                    lbl = lbl[minr:maxr, minc:maxc]
                
                d[self.keys[1]] = lbl

        # 3. Contrast Enhancement (CLAHE)
        # This normalizes the histogram locally, making the model robust to exposure differences
        if self.contrast_enhance:
            # Normalize to 0-1 for CLAHE
            if img.dtype != np.uint8:
                if img.max() > img.min():
                    img = (img - img.min()) / (img.max() - img.min())
                else:
                    img = np.zeros_like(img)
            
            # Apply CLAHE (clip_limit controls contrast, 0.03 is standard)
            # CHANGE: Reduced from 0.03 to 0.01 to avoid over-enhancing high-contrast bubble artifacts
            img = exposure.equalize_adapthist(img, clip_limit=0.01)
        else:
            # 4. Scaling (Only if CLAHE is NOT applied, because CLAHE returns [0, 1])
            img = scale_func(img)

        # 5. Dimension adjustments (Add channel dim if needed: HxW -> 1xHxW)
        # And ensure Channel First (C, H, W)
        if img.ndim == 2:
            img = img[np.newaxis, ...]
        elif img.ndim == 3:
            # Check if channels are last (H, W, C) -> (C, H, W)
            # Heuristic: if last dim is small (<=4) and first dim is large
            if img.shape[-1] <= 4 and img.shape[0] > 4:
                img = np.transpose(img, (2, 0, 1))
        
        d[self.keys[0]] = img.astype(np.float32)

        # Process Label if it exists
        if len(self.keys) > 1 and self.keys[1] in d:
            lbl = d[self.keys[1]]
            # Ensure label is loaded if we didn't do it in the bbox step
            if isinstance(lbl, str):
                lbl = io.imread(lbl)
            
            # Binarize label if needed (assuming 0 is bg, >0 is fg)
            lbl = (lbl > 0).astype(np.float32)
            
            if lbl.ndim == 2:
                lbl = lbl[np.newaxis, ...]
            
            d[self.keys[1]] = lbl

        # Add logic for Weight Dictionary
        if "weight" in self.keys and "weight" in d:
             # Load weight map similar to label
             w_path = d["weight"]
             if isinstance(w_path, str):
                 try:
                     weight_map = io.imread(w_path)
                 except (FileNotFoundError, OSError):
                     # FALLBACK: If map is missing, generate a dummy map (all ones)
                     # This prevents crashing for 1-2 missing files
                     print(f"WARNING: Weight map missing: {w_path}. Using default weights.")
                     # Use the spatial dimensions of the loaded image
                     # img is (C, H, W) at this point
                     current_img = d[self.keys[0]]
                     spatial_shape = current_img.shape[-2:] # H, W
                     weight_map = np.ones(spatial_shape, dtype=np.float32)
                 
             else:
                 weight_map = w_path
                 
             # Normalize or ensure float32
             weight_map = weight_map.astype(np.float32)
             
             # Add channel dim
             if weight_map.ndim == 2:
                 weight_map = weight_map[np.newaxis, ...]
                 
             d["weight"] = weight_map

        return d


class ImageDataset(Dataset):
    def __init__(self, data_fnames, label_fnames, args, training=False, prediction=False):
        # ...existing code...
        self.training = training
        self.prediction = prediction
        self.data_fnames = data_fnames
        self.label_fnames = label_fnames
        
        # FIXED: Only replace the directory name, as the filename still contains '_img_'
        self.weight_fnames = [f.replace('/images/', '/weights/') for f in self.data_fnames]
        
        self.args = args
        self.Nsamples = len(self.data_fnames)
        
        # ... params setup ...
        self.patch_size = args['patch_size']
        self.target_size = (self.patch_size[0], self.patch_size[1])

        self.input_col = args.get('input_channels', 3)
        self.image_col = args.get('image_col', 'image')
        self.boundingbox = args.get('boundingbox', False)
        self.dilation = args.get('dilation', False)
        self.disk_dilation = args.get('disk_dilation', False)
        self.scaling_method = args.get('scaling_method', 'dynamic')
        self.class_values = args.get('class_values', [0, 1, 2])

        self.rotate_range = args.get('rotate_range', 0.0)
        self.translate_range = args.get('translate_range', 0.0)
        self.shear_range = args.get('shear_range', 0.0)
        self.scale_range = args.get('scale_range', 0.0)

        self.contrast_enhance = args.get('contrast_enhance', True)

        # REMOVE OR COMMENT OUT THIS LINE - It overwrites the correct logic from line 204
        # self.weight_fnames = [f.replace('images', 'weights').replace('_img_', '_weight_') for f in self.data_fnames]
        
        # Add to dictionary (Keep this)
        self.data_dicts = [
            {
                "image": self.data_fnames[i], 
                "label": self.label_fnames[i],
                "weight": self.weight_fnames[i]
            } 
            for i in range(self.Nsamples)
        ]
        
        self.transform = self.get_data_transforms(self.training, self.boundingbox, self.dilation, self.disk_dilation, self.scaling_method, self.contrast_enhance)

    def __len__(self):
        return self.Nsamples

    def __getitem__(self, idx):
        # ... (Your existing getitem is good) ...
        if self.training:
            for _ in range(10):
                try:
                    data = self.transform(self.data_dicts[idx])
                    if data['image'].max() > 0:
                        return data
                except Exception:
                    pass
                idx = torch.randint(0, len(self.data_dicts), (1,)).item()
        return self.transform(self.data_dicts[idx])

    def get_data_transforms(self, training, boundingbox, dilation, disk_dilation, scaling_method, contrast_enhance):
        # FIX: Add "weight" to keys list for ALL spatial transforms
        keys_list = ["image", "label", "weight"] 
        
        if not training:  
            transform = Compose(
                [
                    tiff_reader(keys=keys_list, image_col=self.image_col, boundingbox=boundingbox, dilation=dilation, disk_dilation=disk_dilation, scaling_method=scaling_method, contrast_enhance=contrast_enhance),
                    Resized(keys=keys_list, spatial_size=self.target_size, mode=['area', 'nearest', 'nearest']), # 'nearest' for weight
                    MapLabelValued(["label"], self.class_values, list(range(len(self.class_values)))),
                    CastToTyped(keys=["label"], dtype=torch.long),
                    EnsureTyped(keys=keys_list)
                ]
            )
            
        else:  
            transform = Compose(
                [
                    tiff_reader(keys=keys_list, image_col=self.image_col, boundingbox=boundingbox, dilation=dilation, disk_dilation=disk_dilation, scaling_method=scaling_method, contrast_enhance=contrast_enhance),
                    Resized(keys=keys_list, spatial_size=self.target_size, mode=['area', 'nearest', 'nearest']),
                    
                    # FIX: Add 'weight' here
                    RandFlipd(keys=keys_list, prob=0.5, spatial_axis=(0, 1)),
                    
                    # Intensity transforms only apply to image (Correct)
                    RandAdjustContrastd(keys=["image"], prob=0.5),
                    RandGaussianNoised(keys=["image"], prob=0.5, std=0.05),
                    RandGaussianSmoothd(keys=["image"], prob=0.5, sigma_x=(0.5, 1.5), sigma_y=(0.5, 1.5)),
                    RandStdShiftIntensityd(keys=["image"], factors=0.2, prob=0.5),
                    # --- Add new augmentations ---
                    RandScaleIntensityd(keys=["image"], factors=0.2, prob=0.5),
                    RandRotate90d(keys=keys_list, prob=0.5, max_k=3),
                    # ----------------------------
                    RandAffined(
                        keys=keys_list,
                        mode=['bilinear', 'nearest', 'nearest'], # 'nearest' is important for weights/labels
                        padding_mode='zeros', 
                        prob=1.0, 
                        spatial_size=self.patch_size,
                        rotate_range=self.rotate_range * np.ones(1),
                        translate_range=self.translate_range * np.asarray(self.patch_size),
                        shear_range=self.shear_range * np.ones(2),
                        scale_range=self.scale_range * np.ones(2)
                    ),
                    # --- SIMPLIFIED AUGMENTATIONS (Reduced to prevent over-corruption) ---
                    # 1. Mimic Edge Accumulation / Uneven Staining (reduced probability)
                    RandBiasFieldd(keys=["image"], coeff_range=(0.1, 0.2), prob=0.3),

                    # 2. Mimic Stain Intensity Variation (single contrast aug, not duplicate)
                    # RandAdjustContrastd already applied above - removed duplicate
                    
                    # 3. Randomly shifts intensity (reduced offset)
                    RandShiftIntensityd(keys=["image"], offsets=0.05, prob=0.3),
                    
                    # 4. Histogram shift (reduced probability to prevent over-augmentation)
                    RandHistogramShiftd(keys=["image"], num_control_points=5, prob=0.2),

                    # NOTE: Removed duplicate RandGaussianNoised - already applied above
                    
                    MapLabelValued(["label"], self.class_values, list(range(len(self.class_values)))),
                    CastToTyped(keys=["label"], dtype=torch.long),
                    EnsureTyped(keys=keys_list)
                ]
            )
           
        return transform


class PredDataset2D(Dataset):
    def __init__(self, pred_data_dir, args):
        self.pred_data_dir = pred_data_dir
        self.data_file = get_image_paths(pred_data_dir)
        # self.data_file = pred_data_lst
        self.input_col = args['input_channels']
        self.image_col = args['image_col']
        self.boundingbox = args['boundingbox']

    def __len__(self):
        return len(self.data_file)

    def __getitem__(self, idx):
        
        img_name = self.data_file[idx]
        img_path = os.path.join(self.pred_data_dir, img_name)
        # img_path = self.data_file[idx]
        img = np.array(Image.open(img_path))
        if img.ndim == 4 and img.shape[-1] < 4:  # If shape is (h, w, d, c) assuming there are maximum 3 channels or modalities 
            img = np.transpose(img[...,:3], (3, 0, 1, 2))  # Move channel to the first position
            img = dynamic_scale(img)
        elif img.ndim == 3 and img.shape[-1] <= 4:  # If shape is (h, w, c)
            img = np.transpose(img[...,:3], (2, 0, 1))  # Move channel to the first position
            img = dynamic_scale(img)
        elif self.input_col == 1: # grayscale image 
            img = dynamic_scale(img)
            img = np.expand_dims(img, axis=0)

        else:
            raise ValueError(f"Unexpected image shape: {img.shape}, channel dimension should be last and image should be either 2D or 3D")
        
        transform = Compose(
            [
                EnsureType()
            ]
        )
        img = transform(img)
        if self.boundingbox:
            img = extract_largest_component_bbox_image(img)
        
        # Changed from tuple to dictionary to match Unet2D.predict_step expectation
        return {"image": img, "image_meta_dict": {"filename_or_obj": img_path}}


class Unet2D(pl.LightningModule):
    def __init__(self, train_ds, val_ds, **kwargs):
        """    
        PyTorch Lightning Module for training and evaluating 2D and 3D UNet-based models.
        This module supports both the traditional UNet and SwinUNETR architectures 
        for image segmentation tasks.

        Args:
            train_ds (Dataset): Training dataset
            val_ds (Dataset): Validation dataset

        Example: 

        >>> from monai.networks.nets import UNet, SwinUNETR

        >>> # 5 layers each down/up sampling their inputs by a factor 2 with residual blocks
        >>> model = UNet(
        >>>     spatial_dims=2,
        >>>     in_channels=3,
        >>>     out_channels=3,
        >>>     channels=(32, 64, 128, 256, 512),
        >>>     strides=(2, 2, 2, 2),
        >>>     kernel_size=3,
        >>>     up_kernel_size=3,
        >>>     num_res_units=2,
        >>>     dropout=0.4,
        >>>     norm=Norm.BATCH,
        >>>     )

        >>> # SwinUNETR model with 3D inputs and batch normalization
        >>> model = SwinUNETR(
        >>>     spatial_dims=3,
        >>>     img_size=(256, 256, 256),
        >>>     in_channels=3,
        >>>     out_channels=3,
        >>>     feature_size=48,
        >>>     num_heads=[3, 6, 12, 24, 12] ,
        >>>     depths=[2, 4, 8, 16, 24] ,
        >>>     drop_rate=0.4,
        >>>     attn_drop_rate=0.2,
        >>>     norm_name='batch'
        >>>     )
        """
        super(Unet2D, self).__init__()

        self.save_hyperparameters(ignore=['train_ds', 'val_ds'])
        self.hparams.output_channels = self.hparams.num_classes
        self.hparams.pred_patch_size = self.hparams.pred_patch_size
        self.hparams.model = self.hparams.model
        self.hparams.spatial_dims = self.hparams.spatial_dims
        self.hparams.class_values = self.hparams.class_values
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.hparams.task = 'multiclass'

        self.cnfmat = torchmetrics.ConfusionMatrix(num_classes=self.hparams.num_classes,
                                                   task=self.hparams.task,
                                                   normalize=None)
        if self.hparams.model == 'resnet':
            self.model = UNet(
                spatial_dims=self.hparams.spatial_dims,
                in_channels=self.hparams.input_channels,
                out_channels=self.hparams.output_channels,
                channels=(32, 64, 128, 256, 512),
                strides=(2, 2, 2, 2),
                kernel_size=3,
                up_kernel_size=3,
                num_res_units=2,
                dropout=0.4,
                norm=Norm.BATCH,
            )

        
        elif self.hparams.model == "swin":
            img_size = self.hparams.pred_patch_size
            feature_size = 48        
            use_checkpoint = True   
            dropout_rate = 0.4
            attention_dropout_rate = 0.2
            depths = [2, 4, 8, 16, 24]  
            num_heads = [3, 6, 12, 24, 12] 

            # Instantiate the model
            self.model = SwinUNETR(
                spatial_dims=self.hparams.spatial_dims,
                img_size=img_size,
                in_channels=self.hparams.input_channels,
                out_channels=self.hparams.output_channels,
                feature_size=feature_size,
                num_heads=num_heads,
                depths=depths,
                use_checkpoint=use_checkpoint,
                drop_rate=dropout_rate,
                attn_drop_rate=attention_dropout_rate,
                norm_name='batch'
                # norm_name = ('group', {"num_groups": 3, "affine": True})
            )
        
        elif self.hparams.model == "efficientnet":
            # EfficientNet-B4 encoder with UNet decoder
            # Benefits:
            # 1. ImageNet pretrained weights - better feature extraction
            # 2. Compound scaling - efficient depth/width/resolution balance
            # 3. Better texture discrimination (iron vs bubbles)
            self.model = FlexibleUNet(
                in_channels=self.hparams.input_channels,
                out_channels=self.hparams.output_channels,
                backbone="efficientnet-b4",  # Can also try b0-b7
                pretrained=True,  # Use ImageNet pretrained weights
                decoder_channels=(256, 128, 64, 32, 16),
                spatial_dims=self.hparams.spatial_dims,
                norm=("batch", {"eps": 1e-3, "momentum": 0.01}),
                act=("relu", {"inplace": True}),
                dropout=0.3,
                decoder_bias=True,
                upsample="nontrainable",  # Use bilinear upsampling
            )
            print(f"Loaded EfficientNet-B4 encoder with ImageNet pretrained weights")
            print(f"Model params: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")

        else:
            raise AttributeError ("Only 3 possibles models can be applied: resnet, swin, or efficientnet")
        self.has_executed = False

    @staticmethod
    def add_args(parser):
        parser.add_argument("--num_classes", type=int, default=3, help="number of segmentation classes to predict")
        parser.add_argument("--input_channels", type=int, default=3,
                            help="number of input channels (1 for grayscale, 3 for RGB)")
        parser.add_argument("--background_index", type=int, default=0, help="background index")
        parser.add_argument("--pred_patch_size", type=int, default=(64, 64),
                            help="spatial size of rolling window prediction patch")
        parser.add_argument("--batch_size", type=int, default=1, help="dataloader batch size")
        parser.add_argument("--lr", type=int, default=3e-4, help="Adam learning rate")
        parser.add_argument("--num_workers", type=int, default=4, help="number of dataloader workers (cpus)")
        return parser

    def forward(self, x):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return self.model(x.to(device))

    def train_dataloader(self):
        """
        Creates and returns the DataLoader for the training dataset.

        Returns:
            DataLoader: A PyTorch DataLoader for the training dataset.
        """
        train_loader = torch.utils.data.DataLoader(
            self.train_ds, batch_size=self.hparams.batch_size, shuffle=True,
            collate_fn=list_data_collate, num_workers=self.hparams.num_workers,
            persistent_workers=True, pin_memory=torch.cuda.is_available())
        return train_loader

    def val_dataloader(self):
        """
        Creates and returns the DataLoader for the validation dataset.

        Returns:
            DataLoader: A PyTorch DataLoader for the validation dataset.
        """
        val_loader = torch.utils.data.DataLoader(
            self.val_ds, batch_size=self.hparams.batch_size, shuffle=False,
            collate_fn=list_data_collate, num_workers=self.hparams.num_workers,
            persistent_workers=True, pin_memory=torch.cuda.is_available())
        return val_loader

    def configure_optimizers(self):
        # Use AdamW with proper weight decay for better regularization
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.hparams.lr, weight_decay=0.01)
        
        # Option 1: Cosine Annealing with Warm Restarts (helps escape local minima)
        # T_0 = restart every 30 epochs, T_mult = 2 doubles period each restart
        # Restarts at epochs: 30, 90, 210 (good coverage for 300 epochs)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=30,
            T_mult=2,
            eta_min=1e-6
        )
        lr_scheduler = {"scheduler": scheduler,
                        "interval": "epoch"}
        
        # Alternative Option 2 (if cosine doesn't work): Keep ReduceLROnPlateau
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
        #                                                        mode='min',
        #                                                        factor=0.5,  # Less aggressive reduction
        #                                                        patience=15,
        #                                                        min_lr=1e-6)
        # lr_scheduler = {"scheduler": scheduler,
        #                 "monitor": "val_loss",
        #                 "interval": "epoch"}

        return [optimizer], [lr_scheduler]

    def loss_function(self, logits, labels, weight_map=None, class_weights=None):
        """
        V2 LOSS: Focal Loss + Tversky Loss (balanced)
        Focal: handles class imbalance, focuses on hard examples
        Tversky (α=0.6, β=0.4): slightly more balanced, still favors precision
        
        v2 Changes:
        - Tversky: α=0.7→0.6, β=0.3→0.4 (give more weight to recall)
        - Iron weight maps: 4.0→5.0, boundaries: 6.0→7.0
        """
        
        if labels.ndim == 3:
            labels = labels.unsqueeze(1)
        
        # 1. Focal Loss - handles class imbalance
        focal_fn = FocalLoss(
            to_onehot_y=True,
            gamma=2.0,
            weight=class_weights,
            reduction='none'
        )
        pixel_loss = focal_fn(logits, labels)
        
        # Apply weight map if provided
        if weight_map is not None:
            if weight_map.ndim == 3:
                weight_map = weight_map.unsqueeze(1)
            weight_map = weight_map.to(pixel_loss.device).type_as(pixel_loss)
            focal_term = (pixel_loss * weight_map).mean()
        else:
            focal_term = pixel_loss.mean()

        # 2. Tversky Loss - α=0.6, β=0.4 (more balanced, slightly favors precision)
        tversky_fn = TverskyLoss(
            to_onehot_y=True,
            softmax=True,
            include_background=False,
            alpha=0.6,  # Weight for false positives (was 0.7)
            beta=0.4,   # Weight for false negatives (was 0.3)
            smooth_nr=1e-5,
            smooth_dr=1e-5,
        )
        tversky_term = tversky_fn(logits, labels)

        # Combined Loss: 50% Focal + 50% Tversky
        return 0.5 * focal_term + 0.5 * tversky_term

    def training_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        
        # CHECK IF WEIGHT MAP EXISTS IN BATCH
        weight_map = batch.get("weight", None) 
        
        logits = self.forward(images)
        
        classes = list(range(self.hparams.num_classes))
        # torch.save(images, 'train_tensor.pt')

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        class_weights = get_weights(labels, classes, device, include_background=True)

        # Pass weight_map to loss
        loss = self.loss_function(logits, labels, weight_map=weight_map, class_weights=class_weights)
        self.loss = loss.item()
        self.log("train_loss", loss.item(), on_step=False, on_epoch=True, prog_bar=True)

        # if not self.has_executed:
        #     dummy_input = torch.rand((1, self.hparams.input_channels,) + self.hparams.pred_patch_size) 
        #     make_dot(self(dummy_input), params=dict(self.named_parameters())).render('network_graph', format='png')
        #     self.logger.log_image(key="model_visualization", images=["network_graph.png"]) # Wandb Logger
        #     self.has_executed = True
            
        # if os.path.exists("network_graph.png"):
        #     os.remove("network_graph.png")
        # if os.path.exists("train_tensor.pt"):
        #     os.remove("train_tensor.pt")

        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        
        # FIX 1: Retrieve weight map in validation too
        weight_map = batch.get("weight", None)
        
        logits = self.forward(images)
        classes = list(range(self.hparams.num_classes))
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        class_weights = get_weights(labels, classes, device, include_background=True)
        
        # FIX 2: Use keyword arguments so class_weights is not confused with weight_map
        loss = self.loss_function(logits, labels, weight_map=weight_map, class_weights=class_weights)
        
        labeled_ind = (labels != -1)
        mask = labeled_ind.squeeze(1) if labeled_ind.ndim == 4 else labeled_ind
        labeled_logits = torch.cat([logits[i, :, mask[i]] for i in range(len(logits))], dim=-1).transpose(
            0, 1)

        batch_logs = {"loss": loss,
                      "logits": logits,
                      "labels": labels}
        if not self.has_executed:
            to_pil = ToPILImage()
            if batch_idx % 100 == 0:
                x, y = images[:20], logits[:20]
                gridx = torchvision.utils.make_grid(x, nrow=5)

                # Convert the grid to a PIL image
                # --- FIXED: Comment out WandB logging in training_step ---
                # self.logger.log_image(key="training_img", images=[to_pil(gridx)]) 
                # ---------------------------------------------------------
                
                preds = torch.argmax(y, dim=1).byte().squeeze(1)
                preds = (preds * 255).byte()

                y = MapImage(preds, self.hparams.class_values, reverse=False)
                gridy = torchvision.utils.make_grid(y.view(y.shape[0], 1, y.shape[1], y.shape[2]), nrow=5)
                
                # --- FIXED: Comment out WandB logging in training_step ---
                # self.logger.log_image(key="prediction_imgs", images=[to_pil(gridy)])
                # ---------------------------------------------------------

        if torch.cuda.is_available():
            self.cnfmat(labeled_logits, labels[labeled_ind])
        else:
            self.cnfmat(labeled_logits, torch.tensor(labels)[torch.tensor(labeled_ind)])
        self.log("val_loss", loss.item(), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return batch_logs

    def on_validation_epoch_end(self):
        acc, prec, recall, iou, dice = self._compute_cnf_stats()

        # CRITICAL FIX: Print detailed stats to SLURM log so you see them even if CSV fails
        print(f"\n[Epoch {self.current_epoch}] Val Metrics -> Acc: {acc:.4f} | Prec: {prec:.4f} | Recall: {recall:.4f} | IoU: {iou:.4f} | Dice: {dice:.4f}")

        self.log('val_acc', acc, prog_bar=True, sync_dist=True)
        self.log('val_recall', recall, prog_bar=False, sync_dist=True)
        self.log('val_precision', prec, prog_bar=False, sync_dist=True)
        self.log('val_iou', iou, prog_bar=True, sync_dist=True)
        self.log('val_dice', dice, prog_bar=True, sync_dist=True)

        val_logs = {'log':
                        {'val_acc': acc,
                         'val_recall': recall,
                         'val_precision': prec,
                         'val_iou': iou,
                         'val_dice': dice}
                    }

        return val_logs

    def test_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        logits = self.forward(images)
        labeled_ind = (labels != -1)
        mask = labeled_ind.squeeze(1) if labeled_ind.ndim == 4 else labeled_ind
        labeled_logits = torch.cat([logits[i, :, mask[i]] for i in range(len(logits))], dim=-1).transpose(
            0, 1)

        if torch.cuda.is_available():
            self.cnfmat(labeled_logits, labels[labeled_ind])
        else:
            self.cnfmat(labeled_logits, torch.tensor(labels.numpy())[torch.tensor(labeled_ind.numpy())])

    def on_test_epoch_end(self):
        acc, prec, recall, iou, dice = self._compute_cnf_stats()

        print(f"test performance: acc={acc:.02f}, precision={prec:.02f}, recall={recall:.02f}, iou={iou:.02f}, dice={dice:.02f}")
        self.log('test_acc', acc, sync_dist=True)
        self.log('test_recall', recall, sync_dist=True)
        self.log('test_precision', prec, sync_dist=True)
        self.log('test_iou', iou, sync_dist=True)
        self.log('test_dice', dice, sync_dist=True)

        with open(os.path.join(self.hparams.log_dir, 'test_stats.csv'), 'w') as f:
            writer = csv.writer(f)
            # writer.writerow(["acc, prec, recall, iou, dice"])
            # writer.writerow([f"{acc:.02f}", f"{prec:.02f}", f"{recall:.02f}", f"{iou:.02f}", f"{dice:.02f}"])
            writer.writerow(["acc", "prec", "recall", "iou", "dice"])
            writer.writerow([acc, prec, recall, iou, dice])


    def pred_function(self, image):
        return sliding_window_inference(image, self.hparams.pred_patch_size, 1, self.forward)

    def predict_step(self, batch, batch_idx, use_tta=True):
        """
        Prediction with optional Test-Time Augmentation (TTA).
        TTA applies geometric transforms, averages predictions, and improves robustness.
        """
        images = batch['image']
        
        if use_tta:
            # Collect predictions from multiple augmented views
            all_probs = []
            
            # 1. Original
            logits = self(images)
            all_probs.append(torch.softmax(logits, dim=1))
            
            # 2. Horizontal flip
            flipped_h = torch.flip(images, dims=[3])  # flip width
            logits_h = self(flipped_h)
            probs_h = torch.softmax(logits_h, dim=1)
            all_probs.append(torch.flip(probs_h, dims=[3]))  # flip back
            
            # 3. Vertical flip
            flipped_v = torch.flip(images, dims=[2])  # flip height
            logits_v = self(flipped_v)
            probs_v = torch.softmax(logits_v, dim=1)
            all_probs.append(torch.flip(probs_v, dims=[2]))  # flip back
            
            # 4. 90-degree rotation
            rotated_90 = torch.rot90(images, k=1, dims=[2, 3])
            logits_90 = self(rotated_90)
            probs_90 = torch.softmax(logits_90, dim=1)
            all_probs.append(torch.rot90(probs_90, k=-1, dims=[2, 3]))  # rotate back
            
            # 5. 180-degree rotation
            rotated_180 = torch.rot90(images, k=2, dims=[2, 3])
            logits_180 = self(rotated_180)
            probs_180 = torch.softmax(logits_180, dim=1)
            all_probs.append(torch.rot90(probs_180, k=-2, dims=[2, 3]))  # rotate back
            
            # 6. 270-degree rotation
            rotated_270 = torch.rot90(images, k=3, dims=[2, 3])
            logits_270 = self(rotated_270)
            probs_270 = torch.softmax(logits_270, dim=1)
            all_probs.append(torch.rot90(probs_270, k=-3, dims=[2, 3]))  # rotate back
            
            # Average all probability maps
            probs = torch.stack(all_probs, dim=0).mean(dim=0)
        else:
            # No TTA - original behavior
            logits = self(images)
            probs = torch.softmax(logits, dim=1)
        
        preds = torch.argmax(probs, dim=1)
        
        # Return tuple: (Probabilities, Predictions, Filenames)
        # We use this in train.py to save results
        return probs, preds, batch['image_meta_dict']['filename_or_obj']


# The following code assign compute true negative s.t. true positives of the background as the true negatives for all other classes
# This works only in binary settings where the background class is the only negative class 
# For general multiclass tasks, we need a more robust calculation that sums over all non-class rows and columns to avoid misrepresenting the true negatives.
    def _compute_cnf_stats(self):
        cnfmat = self.cnfmat.compute()
        true = torch.diag(cnfmat).as_tensor() 
        tn = true[self.hparams.background_index]
        tp = torch.cat([true[:self.hparams.background_index], true[self.hparams.background_index + 1:]])

        fn = (cnfmat.sum(1) - true)[torch.arange(cnfmat.size(0)) != self.hparams.background_index]
        fp = (cnfmat.sum(0) - true)[torch.arange(cnfmat.size(1)) != self.hparams.background_index]

        acc = torch.sum(true) / torch.sum(cnfmat)
        precision = torch.sum(tp) / torch.sum(tp + fp)
        recall = torch.sum(tp) / torch.sum(tp + fn)
        iou = torch.sum(tp) / (torch.sum(cnfmat) - tn)
        # dice = 2*torch.sum(2*tp) / (torch.sum(cnfmat) + tp - tn)
        dice = 2 * torch.sum(tp) / (2 * torch.sum(tp) + torch.sum(fp) + torch.sum(fn))

        iou_per_class = tp / (tp + fp + fn)

        self.cnfmat.reset()

        return acc.mean().item(), precision.mean().item(), recall.mean().item(), iou.mean().item(), dice.mean().item()


    # Compute metrics without excluding background
    # def _compute_cnf_stats(self):
    #     cnfmat = self.cnfmat.compute()  # Confusion matrix: shape (num_classes, num_classes)
    #     true = torch.diag(cnfmat)  # True positives for each class

    #     # Total true positives, false negatives, and false positives for each class
    #     tp = true
    #     fn = cnfmat.sum(dim=1) - true  # Row sum minus true positives
    #     fp = cnfmat.sum(dim=0) - true  # Column sum minus true positives
    #     tn = cnfmat.sum() - (tp + fn + fp)  # Total - (tp, fp, fn)

    #     # Metrics
    #     acc = torch.sum(tp) / torch.sum(cnfmat)  # Overall accuracy
    #     precision = torch.mean(tp / (tp + fp + 1e-8))  # Macro-averaged precision
    #     recall = torch.mean(tp / (tp + fn + 1e-8))  # Macro-averaged recall
    #     iou = torch.mean(tp / (tp + fp + fn + 1e-8))  # Macro-averaged IoU
    #     iou_per_class = tp / (tp + fp + fn + 1e-8)  # Per-class IoU

    #     self.cnfmat.reset()  # Reset confusion matrix for next evaluation cycle

    #     return acc.item(), precision.item(), recall.item(), iou.item()
