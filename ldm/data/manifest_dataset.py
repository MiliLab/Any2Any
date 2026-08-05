"""
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset

from ldm.util.normalization import NormalizationManager


class ManifestMultiModalDataset(Dataset):
    """
    manifest.json
    
    - manifest.json
    """
    
    def __init__(self,
                 manifest_file,
                 normalization_configs=None,  # 
                 augment=True,
                 **kwargs):
        """
        :
        :param manifest_file: manifest.json
        :param enable_any_to_all: any-to-all
            - False: manifest.json
            - OptSar: [("Opt", "Sar"), ("Sar", "Opt")]
            - OptSar: [("Opt", "Sar")]
            - : [("Opt", "Sar"), ("Opt", "Pan"), ("Sar", "Pan")]
        :param normalization_configs: 
            {
                'Opt': {'normalization': 'uint8', ...},
                'Sar': {'normalization': 'dataset_stats', 'stats_file': '...', ...},
                ...
            }
        
        """
        self.manifest_file = manifest_file
        self.modalities = modalities
        self.enable_any_to_all = enable_any_to_all
        self.allowed_pairs = allowed_pairs
        self.augment = augment
        
        if allowed_pairs is not None:
            self.allowed_pairs_set = self._parse_allowed_pairs(allowed_pairs)
            for src, tgt in sorted(self.allowed_pairs_set):
                print(f"  - {src} -> {tgt}")
        else:
            self.allowed_pairs_set = None
        
        with open(manifest_file, 'r') as f:
            self.manifest = json.load(f)
        
        self.manifest_list = list(self.manifest.values())
        print(f" {manifest_file}  {len(self.manifest_list)} ")
        
        available_modalities = set()
        for sample in self.manifest_list[:10]:  # 10
            available_modalities.update(sample.keys())
        print(f"Manifest: {sorted(available_modalities)}")
        
        self.norm_managers = {}
        if normalization_configs is None:
            normalization_configs = {}
        
        for mod in modalities:
            norm_config = normalization_configs.get(mod_key, {})
            
            self.norm_managers[mod_key] = NormalizationManager(
                normalization=norm_config.get('normalization', 'dataset_stats'),
                data_min=norm_config.get('data_min', None),
                data_max=norm_config.get('data_max', None),
                normalize_per_channel=norm_config.get('normalize_per_channel', False),
                stats_file=norm_config.get('stats_file', None)
            )
        
        if enable_any_to_all:
            self._build_any_to_all_pairs()
        else:
            self.pairs = []
            for idx, sample in enumerate(self.manifest_list):
                available_mods = [mod for mod in modalities if mod in sample]
                if len(available_mods) >= 2:
                    for i, src_mod in enumerate(available_mods):
                        for tgt_mod in available_mods[i+1:]:
                            if self._is_pair_allowed(src_mod, tgt_mod):
                                self.pairs.append({
                                    'sample_idx': idx,
                                    'source_modality': src_mod,
                                    'target_modality': tgt_mod
                                })
                            if self._is_pair_allowed(tgt_mod, src_mod):
                                self.pairs.append({
                                    'sample_idx': idx,
                                    'source_modality': tgt_mod,
                                    'target_modality': src_mod
                                })
            
        
        if augment:
            try:
                import albumentations as A
                self.transform = A.Compose([
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                ], additional_targets={'image1': 'image'})
            except ImportError:
                self.transform = None
        else:
            self.transform = None
    
    def _parse_allowed_pairs(self, allowed_pairs):
        """
        
        Args:
            allowed_pairs: 
                - : [("Opt", "Sar"), ("Sar", "Opt")]
                - : ["Opt->Sar", "Sar->Opt"]
        
        Returns:
            set: {(, ), ...}
        """
        pairs_set = set()
        
        for pair in allowed_pairs:
            if isinstance(pair, tuple) and len(pair) == 2:
                src, tgt = pair
                pairs_set.add((str(src), str(tgt)))
            elif isinstance(pair, str):
                if '->' in pair:
                    src, tgt = pair.split('->', 1)
                    pairs_set.add((src.strip(), tgt.strip()))
                elif '->' in pair:
                    src, tgt = pair.split('->', 1)
                    pairs_set.add((src.strip(), tgt.strip()))
                else:
            else:
        
        return pairs_set
    
    def _is_pair_allowed(self, src_mod, tgt_mod):
        """
        
        
        Args:
            src_mod: 
            tgt_mod: 
        
        Returns:
        """
        if self.allowed_pairs_set is None:
            return True
        return (src_mod, tgt_mod) in self.allowed_pairs_set
    
    def _build_any_to_all_pairs(self):
        """
        any-to-all
        
        """
        self.pairs = []
        
        for idx, sample in enumerate(self.manifest_list):
            available_mods = [mod for mod in self.modalities if mod in sample]
            
            if len(available_mods) >= 2:
                for src_mod in available_mods:
                    for tgt_mod in available_mods:
                        if src_mod != tgt_mod:  # 
                            if self._is_pair_allowed(src_mod, tgt_mod):
                                self.pairs.append({
                                    'sample_idx': idx,
                                    'source_modality': src_mod,
                                    'target_modality': tgt_mod
                                })
        
        if len(self.manifest_list) > 0:
            print(f"   {len(self.pairs) / len(self.manifest_list):.1f} ")
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        if not hasattr(self, '_getitem_counter'):
            self._getitem_counter = 0
        self._getitem_counter += 1
        should_debug = self._getitem_counter <= 3
        
        if should_debug:
            print(f"[DEBUG] __getitem__  (idx={idx}, counter={self._getitem_counter})")
        
        try:
            pair = self.pairs[idx]
            sample = self.manifest_list[pair['sample_idx']]
            src_mod = pair['source_modality']
            tgt_mod = pair['target_modality']
            
            if should_debug:
                print(f"[DEBUG] : {src_mod} -> {tgt_mod}, sample_idx={pair['sample_idx']}")
            
            src_path = sample[src_mod]
            if should_debug:
                print(f"[DEBUG] : {src_path}")
            if not os.path.exists(src_path):
                raise FileNotFoundError(f": {src_path}")
            src_image = np.load(src_path)
            if should_debug:
                print(f"[DEBUG] , shape: {src_image.shape}, dtype: {src_image.dtype}")
                try:
                    print(f"[DEBUG] : range=[{src_image.min():.3f}, {src_image.max():.3f}], mean={src_image.mean():.3f}, std={src_image.std():.3f}")
                except Exception as _e:
                    print(f"[DEBUG] : {_e}")
            
            tgt_path = sample[tgt_mod]
            if should_debug:
                print(f"[DEBUG] : {tgt_path}")
            if not os.path.exists(tgt_path):
                raise FileNotFoundError(f": {tgt_path}")
            tgt_image = np.load(tgt_path)
            if should_debug:
                print(f"[DEBUG] , shape: {tgt_image.shape}, dtype: {tgt_image.dtype}")
                try:
                    print(f"[DEBUG] : range=[{tgt_image.min():.3f}, {tgt_image.max():.3f}], mean={tgt_image.mean():.3f}, std={tgt_image.std():.3f}")
                except Exception as _e:
                    print(f"[DEBUG] : {_e}")
            
            if should_debug:
                print(f"[DEBUG] ...")
            src_image = self._process_image_shape(src_image)
            tgt_image = self._process_image_shape(tgt_image)
            if should_debug:
                print(f"[DEBUG] , src: {src_image.shape}, tgt: {tgt_image.shape}")
            
            if self.transform is not None:
                if should_debug:
                    print(f"[DEBUG] ...")
                src_hwc = src_image if len(src_image.shape) == 3 and src_image.shape[2] <= src_image.shape[0] else src_image.transpose(1, 2, 0)
                tgt_hwc = tgt_image if len(tgt_image.shape) == 3 and tgt_image.shape[2] <= tgt_image.shape[0] else tgt_image.transpose(1, 2, 0)
                
                transformed = self.transform(image=src_hwc, image1=tgt_hwc)
                src_hwc = transformed['image']
                tgt_hwc = transformed['image1']
                
                if len(src_hwc.shape) == 3:
                    src_image = src_hwc.transpose(2, 0, 1)
                if len(tgt_hwc.shape) == 3:
                    tgt_image = tgt_hwc.transpose(2, 0, 1)
                if should_debug:
                    print(f"[DEBUG] ")
            
            if should_debug:
                print(f"[DEBUG] ...")
            src_norm_manager = self.norm_managers[src_mod]
            tgt_norm_manager = self.norm_managers[tgt_mod]
            
            src_image = src_norm_manager.normalize(src_image)
            tgt_image = tgt_norm_manager.normalize(tgt_image)
            if should_debug:
                print(f"[DEBUG] , src range: [{src_image.min():.3f}, {src_image.max():.3f}], tgt range: [{tgt_image.min():.3f}, {tgt_image.max():.3f}]")
            
            if should_debug:
                print(f"[DEBUG] torch tensor...")
            if isinstance(src_image, np.ndarray):
                src_image = torch.from_numpy(src_image).float()
            if isinstance(tgt_image, np.ndarray):
                tgt_image = torch.from_numpy(tgt_image).float()
            
            if len(src_image.shape) == 2:
                src_image = src_image.unsqueeze(0)
            if len(tgt_image.shape) == 2:
                tgt_image = tgt_image.unsqueeze(0)
            
            if should_debug:
            
            return {
                "source_image": src_image,
                "target_image": tgt_image,
                "target_modality": tgt_mod.lower(),
                "source_path": src_path,
                "target_path": tgt_path,
            }
        except Exception as e:
            print(f"[ERROR] __getitem__  (idx={idx}, counter={getattr(self, '_getitem_counter', 'N/A')}): {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _process_image_shape(self, image):
        """
        
        - 2D: (H, W) -> (1, H, W)
        - 3D: (H, W, C) -> (C, H, W)  (C, H, W) -> (C, H, W)
        """
        if len(image.shape) == 2:
            # 2D: (H, W) -> (1, H, W)
            return image[np.newaxis, :, :]
        elif len(image.shape) == 3:
                return image.transpose(2, 0, 1)  # HWC -> CHW
            else:  # CHW
                return image
        else:
            raise ValueError(f": {image.shape}")
    
