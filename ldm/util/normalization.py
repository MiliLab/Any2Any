"""


"""

import numpy as np
import torch
import pickle
import os
from pathlib import Path


class NormalizationManager:
    """
    
    2. 
    3. 
    4. 
    """
    
    def __init__(self, 
                 normalization='uint8',
                 data_min=None,
                 data_max=None,
                 normalize_per_channel=False,
                 stats_file=None):
        """
        :
        :param normalization: 
            - 'uint8': [0, 255] -> [-1, 1]
            - 'zero_one': [0, 1] -> [-1, 1]
            - 'min_max': min/max
        :param normalize_per_channel: 
        """
        self.normalization = normalization
        self.data_min = data_min
        self.data_max = data_max
        self.normalize_per_channel = normalize_per_channel
        self.stats_file = stats_file
        
        if stats_file:
            original_stats_file = stats_file
            if not os.path.isabs(stats_file):
                current_dir = os.getcwd()
                project_root = current_dir
                if 'uniTransfer' in current_dir:
                    parts = current_dir.split(os.sep)
                    if 'uniTransfer' in parts:
                        idx = parts.index('uniTransfer')
                        project_root = os.sep.join(parts[:idx+1])
                
                possible_paths = [
                    os.path.join(current_dir, stats_file),  # 
                    os.path.join(project_root, stats_file),  # 
                    os.path.join(project_root, 'uniTransfer', stats_file),  # uniTransfer
                ]
                
                found = False
                for path in possible_paths:
                    if os.path.exists(path):
                        stats_file = os.path.abspath(path)
                        found = True
                        break
                
                if not found:
                    if os.path.exists(stats_file):
                        stats_file = os.path.abspath(stats_file)
                    else:
                        stats_dir_path = os.path.join(current_dir, 'stats', os.path.basename(stats_file))
                        if os.path.exists(stats_dir_path):
                            stats_file = os.path.abspath(stats_dir_path)
                        else:
            
            if os.path.exists(stats_file):
                try:
                    self.load_stats(stats_file)
                except Exception as e:
                    if self.normalization == 'dataset_stats':
                        raise ValueError(f" {stats_file}: {e}")
            else:
                if self.normalization == 'dataset_stats':
                    print(f": {os.getcwd()}")
                    print(f":")
                    if not os.path.isabs(original_stats_file):
                        current_dir = os.getcwd()
                        project_root = current_dir
                        if 'uniTransfer' in current_dir:
                            parts = current_dir.split(os.sep)
                            if 'uniTransfer' in parts:
                                idx = parts.index('uniTransfer')
                                project_root = os.sep.join(parts[:idx+1])
                        print(f"  - {os.path.join(current_dir, original_stats_file)}")
                        print(f"  - {os.path.join(project_root, original_stats_file)}")
                        print(f"  - {os.path.join(project_root, 'uniTransfer', original_stats_file)}")
        
        if self.normalization == 'dataset_stats' and (self.data_min is None or self.data_max is None):
            raise ValueError(
                f"  1. stats_file: {self.stats_file}\n"
                f"  2. \n"
                f"  3. data_mindata_max"
            )
    
    def normalize(self, image):
        """
        [-1, 1]
        
        Args:
            
        Returns:
        """
        is_torch = isinstance(image, torch.Tensor)
        if is_torch:
            image_np = image.cpu().numpy()
        else:
            image_np = np.array(image)
        
        original_shape = image_np.shape
        is_batch = len(original_shape) == 4  # (B, C, H, W) or (B, H, W, C)
        
        if len(original_shape) == 4:
            if original_shape[1] < original_shape[2]:  # (B, C, H, W)
                image_np = np.transpose(image_np, (0, 2, 3, 1))  # -> (B, H, W, C)
        elif len(original_shape) == 3:
            if original_shape[0] < original_shape[1]:  # (C, H, W)
                image_np = np.transpose(image_np, (1, 2, 0))  # -> (H, W, C)
        elif len(original_shape) == 2:
            image_np = image_np[..., None]  # -> (H, W, 1)
        
        if self.normalization == 'uint8':
            if np.issubdtype(image_np.dtype, np.floating):
                if not hasattr(self, '_validated_uint8_float_range'):
                    self._validated_uint8_float_range = False
                if not self._validated_uint8_float_range:
                    try:
                        mx = float(np.nanmax(image_np))
                        mn = float(np.nanmin(image_np))
                        looks_like_zero_one = (0.0 <= mn <= 0.05) and (0.95 <= mx <= 1.05)
                        looks_like_minus1_one = (-1.05 <= mn <= -0.95) and (0.95 <= mx <= 1.05)
                        if looks_like_zero_one or looks_like_minus1_one:
                            raise ValueError(
                            )
                    except Exception as e:
                        if isinstance(e, ValueError):
                            raise
                    self._validated_uint8_float_range = True
            normalized = (image_np.astype(np.float32) / 127.5) - 1.0
            
        elif self.normalization == 'zero_one':
            normalized = image_np.astype(np.float32) * 2.0 - 1.0
            
        elif self.normalization in ['min_max', 'dataset_stats']:
            if self.data_min is None or self.data_max is None:
                raise ValueError(f"{self.normalization}data_mindata_max")
            
            normalized = self._normalize_min_max(image_np)
        else:
            raise ValueError(f": {self.normalization}")
        
        if len(original_shape) == 4:
            if original_shape[1] < original_shape[2]:  #  (B, C, H, W)
                normalized = np.transpose(normalized, (0, 3, 1, 2))  # -> (B, C, H, W)
        elif len(original_shape) == 3:
            if original_shape[0] < original_shape[1]:  #  (C, H, W)
                normalized = np.transpose(normalized, (2, 0, 1))  # -> (C, H, W)
        elif len(original_shape) == 2:
            normalized = normalized[..., 0]  # -> (H, W)
        
        if is_torch:
            return torch.from_numpy(normalized).to(image.device).to(image.dtype)
        else:
            return normalized.astype(np.float32)
    
    def _normalize_min_max(self, image):
        image = image.astype(np.float32)
        
        if self.normalize_per_channel and image.ndim == 4:  # (B, H, W, C)
            normalized = np.zeros_like(image)
            min_vals = np.array(self.data_min) if isinstance(self.data_min, (list, tuple, np.ndarray)) else [self.data_min] * image.shape[-1]
            max_vals = np.array(self.data_max) if isinstance(self.data_max, (list, tuple, np.ndarray)) else [self.data_max] * image.shape[-1]
            
            for c in range(image.shape[-1]):
                min_val = min_vals[c] if c < len(min_vals) else min_vals[-1]
                max_val = max_vals[c] if c < len(max_vals) else max_vals[-1]
                
                if max_val - min_val > 1e-8:
                    normalized[:, :, :, c] = 2.0 * (image[:, :, :, c] - min_val) / (max_val - min_val) - 1.0
                else:
                    normalized[:, :, :, c] = 0.0
            return normalized
        elif self.normalize_per_channel and image.ndim == 3:  # (H, W, C)
            normalized = np.zeros_like(image)
            min_vals = np.array(self.data_min) if isinstance(self.data_min, (list, tuple, np.ndarray)) else [self.data_min] * image.shape[-1]
            max_vals = np.array(self.data_max) if isinstance(self.data_max, (list, tuple, np.ndarray)) else [self.data_max] * image.shape[-1]
            
            for c in range(image.shape[-1]):
                min_val = min_vals[c] if c < len(min_vals) else min_vals[-1]
                max_val = max_vals[c] if c < len(max_vals) else max_vals[-1]
                
                if max_val - min_val > 1e-8:
                    normalized[:, :, c] = 2.0 * (image[:, :, c] - min_val) / (max_val - min_val) - 1.0
                else:
                    normalized[:, :, c] = 0.0
            return normalized
        else:
            min_val = self.data_min if isinstance(self.data_min, (int, float)) else np.array(self.data_min).min()
            max_val = self.data_max if isinstance(self.data_max, (int, float)) else np.array(self.data_max).max()
            
            if max_val - min_val > 1e-8:
                return 2.0 * (image - min_val) / (max_val - min_val) - 1.0
            else:
                return np.zeros_like(image)
    
    def denormalize(self, image):
        """
        
        Args:
            
        Returns:
        """
        is_torch = isinstance(image, torch.Tensor)
        if is_torch:
            image_np = image.cpu().numpy()
        else:
            image_np = np.array(image)
        
        original_shape = image_np.shape
        
        if len(original_shape) == 4:
            if original_shape[1] < original_shape[2]:  # (B, C, H, W)
                image_np = np.transpose(image_np, (0, 2, 3, 1))  # -> (B, H, W, C)
        elif len(original_shape) == 3:
            if original_shape[0] < original_shape[1]:  # (C, H, W)
                image_np = np.transpose(image_np, (1, 2, 0))  # -> (H, W, C)
        elif len(original_shape) == 2:
            image_np = image_np[..., None]  # -> (H, W, 1)
        
        if self.normalization == 'uint8':
            # [-1, 1] -> [0, 255]
            denormalized = ((image_np + 1.0) * 127.5).clip(0, 255)
            denormalized = denormalized.astype(np.uint8)
            
        elif self.normalization == 'zero_one':
            # [-1, 1] -> [0, 1]
            denormalized = ((image_np + 1.0) / 2.0).clip(0, 1)
            denormalized = denormalized.astype(np.float32)
            
        elif self.normalization in ['min_max', 'dataset_stats']:
            if self.data_min is None or self.data_max is None:
                raise ValueError(f"data_mindata_max")
            denormalized = self._denormalize_min_max(image_np)
        else:
            raise ValueError(f": {self.normalization}")
        
        if len(original_shape) == 4:
            if original_shape[1] < original_shape[2]:  #  (B, C, H, W)
                denormalized = np.transpose(denormalized, (0, 3, 1, 2))  # -> (B, C, H, W)
        elif len(original_shape) == 3:
            if original_shape[0] < original_shape[1]:  #  (C, H, W)
                denormalized = np.transpose(denormalized, (2, 0, 1))  # -> (C, H, W)
        elif len(original_shape) == 2:
            denormalized = denormalized[..., 0]  # -> (H, W)
        
        if is_torch:
            return torch.from_numpy(denormalized).to(image.device)
        else:
            return denormalized
    
    def _denormalize_min_max(self, image):
        if self.normalize_per_channel and image.ndim == 4:  # (B, H, W, C)
            denormalized = np.zeros_like(image, dtype=np.float32)
            min_vals = np.array(self.data_min) if isinstance(self.data_min, (list, tuple, np.ndarray)) else [self.data_min] * image.shape[-1]
            max_vals = np.array(self.data_max) if isinstance(self.data_max, (list, tuple, np.ndarray)) else [self.data_max] * image.shape[-1]
            
            for c in range(image.shape[-1]):
                min_val = min_vals[c] if c < len(min_vals) else min_vals[-1]
                max_val = max_vals[c] if c < len(max_vals) else max_vals[-1]
                
                if max_val - min_val > 1e-8:
                    denormalized[:, :, :, c] = ((image[:, :, :, c] + 1.0) / 2.0) * (max_val - min_val) + min_val
                else:
                    denormalized[:, :, :, c] = min_val
            return denormalized
        elif self.normalize_per_channel and image.ndim == 3:  # (H, W, C)
            denormalized = np.zeros_like(image, dtype=np.float32)
            min_vals = np.array(self.data_min) if isinstance(self.data_min, (list, tuple, np.ndarray)) else [self.data_min] * image.shape[-1]
            max_vals = np.array(self.data_max) if isinstance(self.data_max, (list, tuple, np.ndarray)) else [self.data_max] * image.shape[-1]
            
            for c in range(image.shape[-1]):
                min_val = min_vals[c] if c < len(min_vals) else min_vals[-1]
                max_val = max_vals[c] if c < len(max_vals) else max_vals[-1]
                
                if max_val - min_val > 1e-8:
                    denormalized[:, :, c] = ((image[:, :, c] + 1.0) / 2.0) * (max_val - min_val) + min_val
                else:
                    denormalized[:, :, c] = min_val
            return denormalized
        else:
            min_val = self.data_min if isinstance(self.data_min, (int, float)) else np.array(self.data_min).min()
            max_val = self.data_max if isinstance(self.data_max, (int, float)) else np.array(self.data_max).max()
            
            if max_val - min_val > 1e-8:
                return ((image + 1.0) / 2.0) * (max_val - min_val) + min_val
            else:
                return np.full_like(image, min_val, dtype=np.float32)
    
    def compute_dataset_stats(self, data_loader, num_samples=1000):
        """
        
        Args:
            data_loader: 
            num_samples: 
            
        Returns:
            (data_min, data_max): 
        """
        
        all_mins = []
        all_maxs = []
        sample_count = 0
        
        for batch in data_loader:
            if isinstance(batch, dict):
                images = batch.get('image', batch.get('input'))
            else:
                images = batch
            
            if isinstance(images, torch.Tensor):
                images_np = images.cpu().numpy()
            else:
                images_np = np.array(images)
            
            if len(images_np.shape) == 4 and images_np.shape[1] < images_np.shape[2]:
                images_np = np.transpose(images_np, (0, 2, 3, 1))
            
            if self.normalize_per_channel and images_np.ndim == 4:
                for c in range(images_np.shape[-1]):
                    all_mins.append(images_np[:, :, :, c].min())
                    all_maxs.append(images_np[:, :, :, c].max())
            else:
                all_mins.append(images_np.min())
                all_maxs.append(images_np.max())
            
            sample_count += images_np.shape[0]
            if sample_count >= num_samples:
                break
        
        if self.normalize_per_channel:
            data_min = np.array(all_mins).min()
            data_max = np.array(all_maxs).max()
        else:
            data_min = min(all_mins)
            data_max = max(all_maxs)
        
        self.data_min = data_min
        self.data_max = data_max
        self.normalization = 'dataset_stats'
        
        print(f":")
        print(f"  Min: {data_min}")
        print(f"  Max: {data_max}")
        
        return data_min, data_max
    
    def save_stats(self, filepath):
        stats = {
            'normalization': self.normalization,
            'data_min': self.data_min,
            'data_max': self.data_max,
            'normalize_per_channel': self.normalize_per_channel
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(stats, f)
        print(f": {filepath}")
    
    def load_stats(self, filepath):
        try:
            with open(filepath, 'rb') as f:
                stats = pickle.load(f)
            
            if 'normalization' not in stats:
            if 'data_min' not in stats:
            if 'data_max' not in stats:
            
            self.normalization = stats['normalization']
            self.data_min = stats['data_min']
            self.data_max = stats['data_max']
            self.normalize_per_channel = stats.get('normalize_per_channel', False)
            print(f" {filepath} ")
            print(f"  - normalization: {self.normalization}")
            print(f"  - data_min: {self.data_min}")
            print(f"  - data_max: {self.data_max}")
            print(f"  - normalize_per_channel: {self.normalize_per_channel}")
        except FileNotFoundError:
            raise FileNotFoundError(f": {filepath}")
        except Exception as e:
            raise ValueError(f" {filepath}: {e}")
