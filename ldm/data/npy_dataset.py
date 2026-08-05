import os
import glob
import cv2
import albumentations
import numpy as np
from torch.utils.data import Dataset
from ldm.util.normalization import NormalizationManager


class NpyAutoencoderDataset(Dataset):
    """
    
    - 
    """
    
    def __init__(self, 
                 data_root, 
                 size=256,
                 in_channels=3,
                 min_crop_f=0.5, 
                 max_crop_f=1.0,
                 random_crop=True,
                 center_crop=False,
                 degradation=None,
                 normalization='auto',
                 data_min=None,
                 data_max=None,
                 normalize_per_channel=False,
                 stats_file=None,
                 compute_stats=False):
        """
        :
        :param max_crop_f: 
        :param normalization: 
            
        :param normalize_per_channel: 
        :param compute_stats: 
        """
        assert size is not None, " 'size'"
        assert os.path.exists(data_root), f": {data_root}"
        
        self.data_root = data_root
        self.size = size
        self.in_channels = in_channels
        self.normalization = normalization
        self.normalize_per_channel = normalize_per_channel
        self.stats_file = stats_file
        
        if normalization == 'auto':
        
        self.npy_paths = sorted(glob.glob(os.path.join(self.data_root, "**", "*.npy"), recursive=True))
        
        if not self.npy_paths:
        
        print(f" {len(self.npy_paths)}  .npy  '{data_root}'")
        
        self.norm_manager = NormalizationManager(
            normalization=normalization if normalization != 'auto' else 'min_max',
            data_min=data_min,
            data_max=data_max,
            normalize_per_channel=normalize_per_channel,
            stats_file=stats_file
        )
        
        if compute_stats or (normalization == 'dataset_stats' and data_min is None and not stats_file):
            print("...")
            self._compute_dataset_stats()
            if stats_file:
                self.norm_manager.save_stats(stats_file)
        
        print(f": ={size}, ={in_channels}, ={normalization}")
        
        self.min_crop_f = min_crop_f
        self.max_crop_f = max_crop_f
        assert max_crop_f <= 1.0, "max_crop_f  <= 1.0"
        
        if center_crop:
            self.random_crop = False
        else:
            self.random_crop = random_crop
        
        self.image_rescaler = albumentations.SmallestMaxSize(
            max_size=size, 
            interpolation=cv2.INTER_AREA
        )
        
        self.square_crop = albumentations.CenterCrop(height=size, width=size)
    
    def _compute_dataset_stats(self, num_samples=1000):
        """
        
        """
        
        all_values = []
        sample_count = 0
        
        for idx in range(min(num_samples, len(self.npy_paths))):
            npy_path = self.npy_paths[idx]
            image = np.load(npy_path)
            
            if image.ndim == 2:
                image = image[..., None]
            if image.ndim == 3 and image.shape[0] < image.shape[1]:
                image = np.transpose(image, (1, 2, 0))
            
            all_values.append(image)
            sample_count += 1
        
        all_data = np.concatenate([img.flatten() for img in all_values])
        
        data_min = float(np.min(all_data))
        data_max = float(np.max(all_data))
        
        if self.normalize_per_channel and len(all_values) > 0:
            first_img = all_values[0]
            if first_img.ndim == 3:
                channel_mins = []
                channel_maxs = []
                for c in range(first_img.shape[-1]):
                    channel_data = np.concatenate([img[:, :, c].flatten() for img in all_values])
                    channel_mins.append(float(np.min(channel_data)))
                    channel_maxs.append(float(np.max(channel_data)))
                data_min = channel_mins
                data_max = channel_maxs
        
        self.norm_manager.data_min = data_min
        self.norm_manager.data_max = data_max
        self.norm_manager.normalization = 'dataset_stats'
        
        print(f":")
        if isinstance(data_min, list):
            print(f"  Min: {data_min}")
            print(f"  Max: {data_max}")
        else:
            print(f"  Min: {data_min}")
            print(f"  Max: {data_max}")
        
        if hasattr(self, 'stats_file') and self.stats_file:
            self.norm_manager.save_stats(self.stats_file)
    
    def normalize_data(self, image):
        """
        
        
        Args:
            
        Returns:
        """
        if self.normalization == 'auto':
            image = image.astype(np.float32)
            if self.normalize_per_channel and image.ndim == 3:
                normalized = np.zeros_like(image)
                for c in range(image.shape[-1]):
                    channel_data = image[:, :, c]
                    min_val = np.min(channel_data)
                    max_val = np.max(channel_data)
                    if max_val - min_val > 1e-8:
                        normalized[:, :, c] = 2.0 * (channel_data - min_val) / (max_val - min_val) - 1.0
                    else:
                        normalized[:, :, c] = 0.0
                return normalized
            else:
                min_val = np.min(image)
                max_val = np.max(image)
                if max_val - min_val > 1e-8:
                    return 2.0 * (image.astype(np.float32) - min_val) / (max_val - min_val) - 1.0
                else:
                    return np.zeros_like(image, dtype=np.float32)
        else:
            return self.norm_manager.normalize(image)
    
    def denormalize_data(self, image):
        """
        
        Args:
            
        Returns:
        """
        if self.normalization == 'auto':
        return self.norm_manager.denormalize(image)
    
    def __len__(self):
        return len(self.npy_paths)
    
    def __getitem__(self, idx):
        """
        .npy
        
        :
            dict: 
        """
        npy_path = self.npy_paths[idx]
        
        image = np.load(npy_path)
        
        if image.ndim == 2:
            image = image[..., None]  # (H, W) -> (H, W, 1)
        
        if image.ndim == 3 and image.shape[0] < image.shape[1] and image.shape[0] < image.shape[2]:
            image = np.transpose(image, (1, 2, 0))
        
        if image.ndim == 3:
            current_channels = image.shape[-1]
            
            if current_channels != self.in_channels:
                if current_channels < self.in_channels:
                    channels_to_add = self.in_channels - current_channels
                    image = np.concatenate([image] + [image[:, :, -1:]] * channels_to_add, axis=-1)
                else:
                    image = image[:, :, :self.in_channels]
        else:
        
        needs_albumentations = False
        min_side_len = min(image.shape[:2])
        if min_side_len > self.size or min(image.shape[:2]) != self.size:
            needs_albumentations = True
        
        if needs_albumentations and image.dtype == np.uint16:
            image = image.astype(np.float32)
        
        # if min_side_len > self.size:
        #     crop_side_len = int(min_side_len * np.random.uniform(self.min_crop_f, self.max_crop_f))
            
        #     if self.random_crop:
        #         cropper = albumentations.RandomCrop(height=crop_side_len, width=crop_side_len)
        #     else:
        #         cropper = albumentations.CenterCrop(height=crop_side_len, width=crop_side_len)
            
        #     image = cropper(image=image)["image"]
        
        if min(image.shape[:2]) != self.size:
            image = self.image_rescaler(image=image)["image"]
            
            image = self.square_crop(image=image)["image"]
        
        image = self.normalize_data(image)
        
        assert image.ndim == 3 and image.shape[-1] == self.in_channels, \
        
        return {
            "image": image  # (H, W, C) float32, range [-1, 1]
        }


class NpySRTrain(NpyAutoencoderDataset):
    """
    """
    def __init__(self, **kwargs):
        super().__init__(random_crop=True, **kwargs)


class NpySRValidation(NpyAutoencoderDataset):
    """
    """
    def __init__(self, **kwargs):
        super().__init__(random_crop=False, center_crop=True, **kwargs)