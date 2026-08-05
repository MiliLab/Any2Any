"""
"""

import torch
import numpy as np
import pytorch_lightning as pl
from omegaconf import OmegaConf
from contextlib import contextmanager
from einops import rearrange, repeat
from torchvision.utils import make_grid

from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.util import instantiate_from_config, default, log_txt_as_img, ismap, isimage
from ldm.modules.adapter import MultiModalAdapterManager
from ldm.modules.conditioner import FlexibleConditioner
from ldm.modules.ema import LitEma
from ldm.models.autoencoder import VQModelInterface, AutoencoderKL, IdentityFirstStage


class MultiModalTranslator(LatentDiffusion):
    """
    
    
    """
    
    def __init__(self,
                 
                 unet_config,
                 
                 conditioning_mode='concat',  # 'concat', 'crossattn', 'hybrid', 'none'
                 condition_channels=3,
                 context_dim=512,
                 use_modality_embedding=False,
                 
                 adapter_type='simple',  # 'simple', 'residual', 'attention'
                 
                 modalities=['opt', 'sar', 'pan', 'nir', 'ms'],
                 
                 **kwargs):
        """
        :
        :param unet_config: UNet
        :param conditioning_mode: 
        :param condition_channels: 
        :param context_dim: cross-attentioncontext
        :param use_modality_embedding: 
        :param enable_adapters: adapter
        :param adapter_type: adapter
        :param adapter_configs: adapter
        :param modalities: 
        :param **kwargs: LatentDiffusion
        """
        
        self.modalities = modalities
        self.num_modalities = len(modalities)
        self.enable_adapters = enable_adapters
        self.conditioning_mode = conditioning_mode
        self.condition_channels = condition_channels
        
        self.modality_to_id = {mod: i for i, mod in enumerate(modalities)}
        
        cond_stage_config = kwargs.pop('cond_stage_config', None)
        sd15_ckpt_path = kwargs.pop('sd15_ckpt_path', None)
        freeze_unet_encoder = kwargs.pop('freeze_unet_encoder', False)
        
        first_modality = modalities[0]
        temp_first_stage_config = first_stage_configs[first_modality]
        
        preserve_spatial = kwargs.get('preserve_spatial', True)
        conditioner_config = {
            'target': 'ldm.modules.conditioner.FlexibleConditioner',
            'params': {
                'conditioning_mode': conditioning_mode,
                'condition_channels': condition_channels,
                'context_dim': context_dim,
                'use_modality_embedding': use_modality_embedding,
                'num_modalities': self.num_modalities,
                'preserve_spatial': preserve_spatial
            }
        }
        
        if cond_stage_config is None:
            cond_stage_config = conditioner_config
        
        # ========================================================================
        # ========================================================================
        latent_channels = condition_channels 
        
        if conditioning_mode == 'concat':
            target_channels = latent_channels
            condition_input_channels = condition_channels
            if use_modality_embedding:
                condition_input_channels += condition_channels * 2
            unet_in_channels = target_channels + condition_input_channels
            
        elif conditioning_mode == 'crossattn':
            unet_in_channels = latent_channels
            
        elif conditioning_mode == 'hybrid':
            target_channels = latent_channels
            condition_input_channels = condition_channels
            if use_modality_embedding:
                condition_input_channels += condition_channels * 2
            unet_in_channels = target_channels + condition_input_channels
            
        elif conditioning_mode == 'none':
            unet_in_channels = latent_channels
            
        else:
            raise ValueError(f": {conditioning_mode}")
        
        unet_config = unet_config.copy()
        if 'params' not in unet_config:
            unet_config['params'] = {}
        unet_config['params']['in_channels'] = unet_in_channels
        
        # ========================================================================
        # ========================================================================
        if conditioning_mode == 'concat':
            unet_config['params']['use_spatial_transformer'] = False
            unet_config['params']['context_dim'] = None
            unet_config['params']['legacy'] = False
        
        self.use_spatial_transformer = unet_config['params'].get('use_spatial_transformer', True)

        super().__init__(
            first_stage_config=temp_first_stage_config,
            cond_stage_config=cond_stage_config,
            unet_config=unet_config,
            **kwargs
        )
        
        self._sd15_ckpt_path = sd15_ckpt_path
        self._freeze_unet_encoder = freeze_unet_encoder
        
        # ========================================================================
        # ========================================================================
        self.first_stage_models = torch.nn.ModuleDict()
        for mod in modalities:
            if mod not in first_stage_configs:
                raise ValueError(f" '{mod}' AE")
            config = first_stage_configs[mod]
            self.first_stage_models[mod] = instantiate_from_config(config)
            for param in self.first_stage_models[mod].parameters():
                param.requires_grad = False
            print(f" '{mod}' AE")
        
        self.first_stage_model = self.first_stage_models[modalities[0]]
        
        # ========================================================================
        # ========================================================================
        self.adapter_manager = MultiModalAdapterManager(
            modalities=modalities,
            latent_channels=latent_channels,
            enable_adapters=enable_adapters,
            adapter_type=adapter_type,
            adapter_configs=adapter_configs
        )
        
        if conditioning_mode == 'concat':
            self.model.conditioning_key = 'concat'
        elif conditioning_mode == 'crossattn':
            self.model.conditioning_key = 'crossattn'
        elif conditioning_mode == 'hybrid':
            self.model.conditioning_key = 'hybrid'
        elif conditioning_mode == 'none':
            self.model.conditioning_key = None
        
        print(f": {conditioning_mode}")
        print(f"Adapter: { if enable_adapters else }")
        print(f"UNet: {unet_in_channels}")
        print(f"UNet: {self.model.conditioning_key}")
        print(f"Spatial Transformer: { if self.use_spatial_transformer else }")
        
        # ========================================================================
        # ========================================================================
        if self._sd15_ckpt_path is not None:
            self.load_sd15_weights(self._sd15_ckpt_path)
        
        # ========================================================================
        # ========================================================================
        if self._freeze_unet_encoder:
            self.freeze_unet_encoder()
    
    def get_ae(self, modality):
        """
        AE
        
        Args:
            modality: 
            
        Returns:
            AE
        """
        if modality not in self.first_stage_models:
            raise ValueError(f": {modality}")
        return self.first_stage_models[modality]
    
    def encode_to_latent(self, x, modality):
        """
        AE
        
        Args:
            x:  [B, C, H, W]
            modality: 
            
        Returns:
             [B, C_latent, H_latent, W_latent]
        """
        # ========================================================================
        # ========================================================================
        if not hasattr(self, '_input_check_counter'):
            self._input_check_counter = 0
        self._input_check_counter += 1
        
        should_check_input = (self._input_check_counter <= 10) or (self._input_check_counter % 200 == 0)
        
        try:
            is_training = getattr(self, 'training', False)
        except:
            is_training = False
        
        if should_check_input and is_training:
            try:
                with torch.no_grad():
                    input_stats = {
                        'min': x.min().item(),
                        'max': x.max().item(),
                        'mean': x.mean().item(),
                        'std': x.std().item(),
                        'abs_max': x.abs().max().item(),
                    }
                    
                    if self._input_check_counter <= 10 or self._input_check_counter % 500 == 0:
                        print(f"\n{'='*80}")
                        print(f"{'='*80}")
                        print(f"   : [{input_stats['min']:.6f}, {input_stats['max']:.6f}]")
                        print(f"   : {input_stats['mean']:.6f}, : {input_stats['std']:.6f}")
                        print(f"   : {input_stats['abs_max']:.6f}")
                        print(f"   : {list(x.shape)}")
                        
                        if input_stats['min'] < -1.1 or input_stats['max'] > 1.1:
                            print(f"      : [-1.0, 1.0]")
                            print(f"      : [{input_stats['min']:.6f}, {input_stats['max']:.6f}]")
                            print(f"      ")
                        elif abs(input_stats['min'] + 1.0) > 0.1 or abs(input_stats['max'] - 1.0) > 0.1:
                        else:
                        print()
            except Exception as e:
                if self._input_check_counter <= 1:
                pass
        
        ae = self.get_ae(modality)
        with torch.no_grad():
            posterior = ae.encode(x)
            
            # ========================================================================
            # ========================================================================
            if should_check_input and is_training:
                with torch.no_grad():
                    z_raw_stats = {
                        'min': z_raw.min().item(),
                        'max': z_raw.max().item(),
                        'mean': z_raw.mean().item(),
                        'std': z_raw.std().item(),
                        'abs_max': z_raw.abs().max().item(),
                    }
                    
                    if self._input_check_counter <= 10 or self._input_check_counter % 500 == 0:
                        print(f"   : [{z_raw_stats['min']:.6f}, {z_raw_stats['max']:.6f}]")
                        print(f"   : {z_raw_stats['mean']:.6f}, : {z_raw_stats['std']:.6f}")
                        print(f"   : {z_raw_stats['abs_max']:.6f}")
                        print(f"   : {list(z_raw.shape)}")
                        
                        if z_raw_stats['abs_max'] > 10.0:
                            print(f"      ")
                        print()
            
            try:
                scale_factor_val = self.scale_factor.item() if isinstance(self.scale_factor, torch.Tensor) else self.scale_factor
            except (AttributeError, RuntimeError):
                scale_factor_val = getattr(self, 'scale_factor', 1.0)
                if isinstance(scale_factor_val, torch.Tensor):
                    scale_factor_val = scale_factor_val.item()
            z_scaled = scale_factor_val * z_raw
            
            if should_check_input and is_training:
                with torch.no_grad():
                    z_scaled_stats = {
                        'min': z_scaled.min().item(),
                        'max': z_scaled.max().item(),
                        'mean': z_scaled.mean().item(),
                        'std': z_scaled.std().item(),
                        'abs_max': z_scaled.abs().max().item(),
                    }
                    
                    if self._input_check_counter <= 10 or self._input_check_counter % 500 == 0:
                        print(f"   : [{z_scaled_stats['min']:.6f}, {z_scaled_stats['max']:.6f}]")
                        print(f"   : {z_scaled_stats['mean']:.6f}, : {z_scaled_stats['std']:.6f}")
                        print(f"   : {z_scaled_stats['abs_max']:.6f}")
                        
                        if z_scaled_stats['abs_max'] > 1.0:
                            print(f"      ")
                        elif z_scaled_stats['abs_max'] < 0.1:
                            print(f"      ")
                        else:
                        
                        print(f"{'='*80}\n")
        
        return z_raw
    
    def decode_from_latent(self, z, modality, is_scaled=False):
        """
        AE
        
        Args:
            z:  [B, C_latent, H_latent, W_latent]
            modality: 
            
        Returns:
             [B, C, H, W]
        
        """
        ae = self.get_ae(modality)
        with torch.no_grad():
            if is_scaled:
                try:
                    scale_factor_val = self.scale_factor.item() if isinstance(self.scale_factor, torch.Tensor) else self.scale_factor
                except (AttributeError, RuntimeError):
                    scale_factor_val = getattr(self, 'scale_factor', 1.0)
                    if isinstance(scale_factor_val, torch.Tensor):
                        scale_factor_val = scale_factor_val.item()
            
            dec = ae.decode(z)
        return dec
    
    def decode_first_stage(self, z, predict_cids=False, force_not_quantize=False, modality=None):
        """
        
        Args:
            force_not_quantize: 
            
        Returns:
             [B, C, H, W]
        """
        if predict_cids:
            if z.dim() == 4:
                z = torch.argmax(z.exp(), dim=1).long()
            if modality is not None:
                ae = self.get_ae(modality)
            else:
                ae = self.first_stage_model
            z = ae.quantize.get_codebook_entry(z, shape=None)
            z = rearrange(z, 'b h w c -> b c h w').contiguous()

        z = 1. / self.scale_factor * z

        if modality is not None:
            ae = self.get_ae(modality)
        else:
            ae = self.first_stage_model

        if hasattr(self, "split_input_params"):
            if self.split_input_params["patch_distributed_vq"]:
                ks = self.split_input_params["ks"]  # eg. (128, 128)
                stride = self.split_input_params["stride"]  # eg. (64, 64)
                uf = self.split_input_params["vqf"]
                bs, nc, h, w = z.shape
                if ks[0] > h or ks[1] > w:
                    ks = (min(ks[0], h), min(ks[1], w))
                    print("reducing Kernel")

                if stride[0] > h or stride[1] > w:
                    stride = (min(stride[0], h), min(stride[1], w))
                    print("reducing stride")

                fold, unfold, normalization, weighting = self.get_fold_unfold(z, ks, stride, uf=uf)

                z = unfold(z)  # (bn, nc * prod(**ks), L)
                # 1. Reshape to img shape
                z = z.view((z.shape[0], -1, ks[0], ks[1], z.shape[-1]))  # (bn, nc, ks[0], ks[1], L )

                # 2. apply model loop over last dim
                if isinstance(ae, VQModelInterface):
                    output_list = [ae.decode(z[:, :, :, :, i],
                                             force_not_quantize=predict_cids or force_not_quantize)
                                   for i in range(z.shape[-1])]
                else:
                    output_list = [ae.decode(z[:, :, :, :, i])
                                   for i in range(z.shape[-1])]

                o = torch.stack(output_list, axis=-1)  # # (bn, nc, ks[0], ks[1], L)
                o = o * weighting
                # Reverse 1. reshape to img shape
                o = o.view((o.shape[0], -1, o.shape[-1]))  # (bn, nc * ks[0] * ks[1], L)
                # stitch crops together
                decoded = fold(o)
                decoded = decoded / normalization  # norm is shape (1, 1, h, w)
                return decoded
            else:
                if isinstance(ae, VQModelInterface):
                    return ae.decode(z, force_not_quantize=predict_cids or force_not_quantize)
                else:
                    return ae.decode(z)
        else:
            if isinstance(ae, VQModelInterface):
                return ae.decode(z, force_not_quantize=predict_cids or force_not_quantize)
            else:
                return ae.decode(z)
    
    # def get_input(self, batch, k=None, return_first_stage_outputs=False, **kwargs):
    #     """
        
    #     Args:
            
    #     Returns:
    #     """
    #     if not hasattr(self, '_get_input_counter'):
    #         self._get_input_counter = 0
    #     self._get_input_counter += 1
        
    #     if self._get_input_counter <= 3:
    #         print(f"[DEBUG] batch keys: {list(batch.keys()) if isinstance(batch, dict) else 'Not a dict'}")
        
    #     try:
    #         target_images = batch.get("target_image", batch.get("image"))
    #         target_modalities = batch.get("target_modality", ["sar"])
            
    #         target_images_processed = None
            
    #         if self._get_input_counter <= 3:
    #             print(f"[DEBUG] target_images type: {type(target_images)}")
    #             if isinstance(target_images, torch.Tensor):
    #                 print(f"[DEBUG] target_images shape: {target_images.shape}")
    #             elif isinstance(target_images, list):
    #                 print(f"[DEBUG] target_images list length: {len(target_images)}")
    #                 if len(target_images) > 0:
    #                     print(f"[DEBUG] first item shape: {target_images[0].shape if hasattr(target_images[0], 'shape') else 'N/A'}")
            
    #         if isinstance(target_images, torch.Tensor):
    #             if len(target_images.shape) == 3:
    #                 target_images = target_images.unsqueeze(0)
    #             target_images_processed = target_images
    #             if self._get_input_counter <= 3:
    #             target_latent = self.encode_to_latent(target_images, target_modalities[0] if isinstance(target_modalities, list) else target_modalities)
    #             if self._get_input_counter <= 3:
    #             if len(target_latent.shape) == 3:
    #                 target_latent = target_latent.unsqueeze(0)
    #         elif isinstance(target_images, list):
    #             if isinstance(target_modalities, str):
    #                 target_modalities = [target_modalities]
    #             if len(target_images) != len(target_modalities):
    #                 if len(target_modalities) == 1:
    #                     target_modalities = target_modalities * len(target_images)
    #                 else:
                
    #             target_latents = []
    #             for tgt_img, tgt_mod in zip(target_images, target_modalities):
    #                 if isinstance(tgt_img, torch.Tensor) and len(tgt_img.shape) == 3:
    #                     tgt_img = tgt_img.unsqueeze(0)
    #                 tgt_z = self.encode_to_latent(tgt_img, tgt_mod)
    #                 if len(tgt_z.shape) == 3:
    #                     tgt_z = tgt_z.unsqueeze(0)
    #                 if len(tgt_z.shape) == 4 and tgt_z.shape[0] == 1:
    #                     tgt_z = tgt_z.squeeze(0)  # [1, C, H, W] -> [C, H, W]
    #                 target_latents.append(tgt_z)
                
    #             if len(target_latents) == 1:
    #                 target_latent = target_latents[0]
    #                 if len(target_latent.shape) == 3:
    #                     target_latent = target_latent.unsqueeze(0)  # [C, H, W] -> [1, C, H, W]
    #             else:
    #                 first_shape = target_latents[0].shape  # [C, H, W]
    #                 for i, tgt_z in enumerate(target_latents[1:], 1):
    #                     if tgt_z.shape != first_shape:
    #                         raise ValueError(
    #                             f"target_latents[0].shape={target_latents[0].shape}, "
    #                             f"target_latents[{i}].shape={tgt_z.shape}"
    #                         )
    #                 # Stack: [C, H, W] * N -> [N, C, H, W]
    #                 target_latent = torch.stack(target_latents, dim=0)
                
    #             if len(target_images) > 0:
    #                 first_img = target_images[0]
    #                 if isinstance(first_img, torch.Tensor):
    #                     processed_imgs = []
    #                     for img in target_images:
    #                         if isinstance(img, torch.Tensor):
    #                             if len(img.shape) == 3:
    #                                 processed_imgs.append(img)
    #                             elif len(img.shape) == 4:
    #                                 # [1, C, H, W] -> [C, H, W]
    #                                 processed_imgs.append(img.squeeze(0))
    #                             else:
    #                         else:
    #                     # Stack: [C, H, W] * N -> [N, C, H, W]
    #                     target_images_processed = torch.stack(processed_imgs, dim=0)
    #                 else:
    #         else:
            
    #         if len(target_latent.shape) != 4:
            
    #         if self.conditioning_mode != 'none':
    #             source_images = batch.get("source_image", batch.get("condition_image"))
                
    #             if source_images is None:
                
    #             if isinstance(source_images, torch.Tensor):
    #                 if len(source_images.shape) == 3:
    #                     source_images = source_images.unsqueeze(0)
    #                 if isinstance(source_modalities, str):
    #                     source_modalities = [source_modalities]
    #                 if self._get_input_counter <= 3:
    #                 src_z = self.encode_to_latent(source_images, source_modalities[0])
    #                 if self._get_input_counter <= 3:
    #                 if len(src_z.shape) == 3:
    #                     src_z = src_z.unsqueeze(0)
    #                 if self._get_input_counter <= 3:
    #                 source_latent = self.adapter_manager.forward_before(src_z, source_modalities[0])
    #                 if self._get_input_counter <= 3:
    #             elif isinstance(source_images, list):
    #                 if len(source_images) == 0:
    #                 if isinstance(source_modalities, str):
    #                     source_modalities = [source_modalities]
    #                 if len(source_images) != len(source_modalities):
    #                     if len(source_modalities) == 1:
    #                         source_modalities = source_modalities * len(source_images)
    #                     else:
                    
    #                 source_latents = []
    #                 for src_img, src_mod in zip(source_images, source_modalities):
    #                     if isinstance(src_img, torch.Tensor) and len(src_img.shape) == 3:
    #                         src_img = src_img.unsqueeze(0)
    #                     src_z = self.encode_to_latent(src_img, src_mod)
    #                     if len(src_z.shape) == 3:
    #                         src_z = src_z.unsqueeze(0)
    #                     src_z = self.encode_to_latent(src_img, src_mod)
    #                     if len(src_z.shape) == 3:
    #                         src_z = src_z.unsqueeze(0)
    #                     if len(src_z.shape) == 4 and src_z.shape[0] == 1:
    #                         src_z = src_z.squeeze(0)  # [1, C, H, W] -> [C, H, W]
    #                     if len(src_z.shape) == 3:
    #                         src_z = src_z.unsqueeze(0)  # [C, H, W] -> [1, C, H, W]
    #                     src_z = self.adapter_manager.forward_before(src_z, src_mod)
    #                     if len(src_z.shape) == 4 and src_z.shape[0] == 1:
    #                         src_z = src_z.squeeze(0)  # [1, C, H, W] -> [C, H, W]
    #                     source_latents.append(src_z)
                    
    #                 if len(source_latents) == 1:
    #                     source_latent = source_latents[0]
    #                     if len(source_latent.shape) == 3:
    #                         source_latent = source_latent.unsqueeze(0)  # [C, H, W] -> [1, C, H, W]
    #                 else:
    #                     first_shape = source_latents[0].shape  # [C, H, W]
    #                     for i, src_z in enumerate(source_latents[1:], 1):
    #                         if src_z.shape != first_shape:
    #                             raise ValueError(
    #                                 f"source_latents[0].shape={source_latents[0].shape}, "
    #                                 f"source_latents[{i}].shape={src_z.shape}"
    #                             )
    #                     # Stack: [C, H, W] * N -> [N, C, H, W]
    #                     source_latent = torch.stack(source_latents, dim=0)
    #             else:
                
    #             if len(source_latent.shape) != 4:
    #         else:
    #             source_latent = None
    #             source_modalities = None
            
    #         if self.conditioning_mode == 'none':
    #             condition_dict = {}
    #         else:
    #             if source_latent is None:
                
    #             source_modality_ids = torch.tensor([self.modality_to_id[mod] for mod in source_modalities], 
    #                                           device=source_latent.device)
    #         target_modality_ids = torch.tensor([self.modality_to_id[mod] for mod in target_modalities],
    #                                           device=target_latent.device)
            
    #         if len(source_latent.shape) == 3:
    #             source_latent = source_latent.unsqueeze(0)
            
    #         if source_latent.shape[0] != target_latent.shape[0]:
    #             raise ValueError(
    #                 f"source_latent.shape={source_latent.shape}, "
    #                 f"target_latent.shape={target_latent.shape}"
    #             )
            
    #         if self._get_input_counter <= 3:
    #         condition_dict = self.cond_stage_model(
    #             source_latent,
    #             source_modality_id=source_modality_ids,
    #             target_modality_id=target_modality_ids
    #         )
    #         if self._get_input_counter <= 3:
            
    #         if self.conditioning_mode == 'concat':
    #             if "c_concat" not in condition_dict:
    #             if len(condition_dict["c_concat"]) == 0:
    #             for i, cond_tensor in enumerate(condition_dict["c_concat"]):
    #                 if len(cond_tensor.shape) != 4:
    #                     raise ValueError(
    #                     )
    #         elif self.conditioning_mode == 'crossattn':
    #             if "c_crossattn" not in condition_dict:
    #             if len(condition_dict["c_crossattn"]) == 0:
    #             for i, cond_tensor in enumerate(condition_dict["c_crossattn"]):
    #                 if len(cond_tensor.shape) != 3:
    #                     raise ValueError(
    #                     )
    #         elif self.conditioning_mode == 'hybrid':
    #             if "c_concat" not in condition_dict:
    #             if "c_crossattn" not in condition_dict:
    #             if len(condition_dict["c_concat"]) == 0:
    #             if len(condition_dict["c_crossattn"]) == 0:
            
    #         if source_modalities is not None and len(source_modalities) > 0:
    #             source_modality_ids_viz = torch.tensor([self.modality_to_id[mod] for mod in source_modalities],
    #                                                   device=target_latent.device)
    #         else:
    #             source_modality_ids_viz = None
            
    #         # ========================================================================
    #         # ========================================================================
    #         bs = kwargs.get('bs', None)
    #         if bs is not None:
    #             actual_bs = bs
    #             if source_latent is not None:
    #                 actual_bs = min(actual_bs, source_latent.shape[0])
    #             actual_bs = min(actual_bs, target_latent.shape[0])
                
    #             if source_latent is not None:
    #                 source_latent = source_latent[:actual_bs]
    #                 if source_modalities is not None and len(source_modalities) > 0:
    #                     source_modality_ids = torch.tensor([self.modality_to_id[mod] for mod in source_modalities[:actual_bs]], 
    #                                                       device=source_latent.device)
                
    #             target_latent = target_latent[:actual_bs]
                
    #             if target_images_processed is not None:
    #                 target_images_processed = target_images_processed[:actual_bs]
                
    #             if source_latent is not None and self.conditioning_mode != 'none':
    #                 target_modality_ids = torch.tensor([self.modality_to_id[mod] for mod in target_modalities[:actual_bs]],
    #                                                   device=target_latent.device)
    #                 condition_dict = self.cond_stage_model(
    #                     source_latent,
    #                     source_modality_id=source_modality_ids,
    #                     target_modality_id=target_modality_ids
    #                 )
    #             elif condition_dict is not None:
    #                 for key in condition_dict:
    #                     if isinstance(condition_dict[key], list):
    #                         condition_dict[key] = [tensor[:actual_bs] if isinstance(tensor, torch.Tensor) else tensor for tensor in condition_dict[key]]
    #                     elif isinstance(condition_dict[key], torch.Tensor):
    #                         condition_dict[key] = condition_dict[key][:actual_bs]
            
    #         if not hasattr(self, '_final_latent_check_counter'):
    #             self._final_latent_check_counter = 0
            
    #         try:
    #             has_global_step = hasattr(self, 'global_step')
    #             is_training = getattr(self, 'training', False)
    #             should_check_final = (has_global_step and is_training and 
    #                                 ((self._final_latent_check_counter <= 10) or (self._final_latent_check_counter % 200 == 0)))
    #         except:
    #             should_check_final = False
    #             is_training = False
            
    #         if should_check_final:
    #             self._final_latent_check_counter += 1
    #             with torch.no_grad():
    #                 final_target_stats = {
    #                 'min': target_latent.min().item(),
    #                 'max': target_latent.max().item(),
    #                 'mean': target_latent.mean().item(),
    #                 'std': target_latent.std().item(),
    #                 'abs_max': target_latent.abs().max().item(),
    #             }
                
    #             try:
    #                 scale_factor_val = self.scale_factor.item() if isinstance(self.scale_factor, torch.Tensor) else self.scale_factor
    #             except (AttributeError, RuntimeError):
    #                 scale_factor_val = getattr(self, 'scale_factor', 1.0)
    #                 if isinstance(scale_factor_val, torch.Tensor):
    #                     scale_factor_val = scale_factor_val.item()
    #             target_scaled = target_latent * scale_factor_val
    #             final_target_scaled_stats = {
    #                 'min': target_scaled.min().item(),
    #                 'max': target_scaled.max().item(),
    #                 'mean': target_scaled.mean().item(),
    #                 'std': target_scaled.std().item(),
    #                 'abs_max': target_scaled.abs().max().item(),
    #             }
                
    #             if self._final_latent_check_counter <= 10 or self._final_latent_check_counter % 500 == 0:
    #                 print(f"\n{'='*80}")
    #                 print(f"{'='*80}")
    #                 print(f"   Scale Factor: {scale_factor_val:.6f}")
                    
    #                 abs_max_scaled = final_target_scaled_stats['abs_max']
    #                 if abs_max_scaled > 1.0:
    #                 elif abs_max_scaled > 0.8:
    #                 elif abs_max_scaled < 0.1:
    #                 else:
                    
    #                 if scale_factor_val < 0.01:
    #                 elif scale_factor_val > 1.0:
                    
    #                 print(f"{'='*80}\n")
            
    #         if return_first_stage_outputs:
    #             target_rec = self.decode_from_latent(target_latent, target_modalities[0] if isinstance(target_modalities, list) else target_modalities)
    #             if source_latent is not None and source_modalities is not None:
    #                 source_rec = self.decode_from_latent(source_latent, source_modalities[0] if isinstance(source_modalities, list) else source_modalities)
    #                 if target_images_processed is not None:
    #                     return target_latent, condition_dict, target_images_processed, target_rec, source_rec
    #                 else:
    #                     if isinstance(target_images, torch.Tensor):
    #                         return target_latent, condition_dict, target_images[:target_latent.shape[0]], target_rec, source_rec
    #                     else:
    #                         return target_latent, condition_dict, target_images[0] if len(target_images) > 0 else target_images, target_rec, source_rec
    #             else:
    #                 if target_images_processed is not None:
    #                     return target_latent, condition_dict, target_images_processed, target_rec, None
    #                 else:
    #                     if isinstance(target_images, torch.Tensor):
    #                         return target_latent, condition_dict, target_images[:target_latent.shape[0]], target_rec, None
    #                     else:
    #                         return target_latent, condition_dict, target_images[0] if len(target_images) > 0 else target_images, target_rec, None
    #         else:
    #             if self._get_input_counter <= 3:
    #             return target_latent, condition_dict
    #     except Exception as e:
    #         import traceback
    #         traceback.print_exc()
    #         raise

    # def get_input(self, batch, k=None, return_first_stage_outputs=False, **kwargs):
    #         """
    #         """
    #         if not hasattr(self, '_get_input_counter'):
    #             self._get_input_counter = 0
    #         self._get_input_counter += 1
            
    #         try:
    #             # ============================================================
    #             # ============================================================
    #             target_images = batch.get("target_image", batch.get("image"))
                
    #             target_modalities = batch.get("target_modality")
    #             if target_modalities is None:

    #             if isinstance(target_images, list):
    #                 if len(target_images) > 0 and isinstance(target_images[0], torch.Tensor):
    #                     target_images = torch.stack(target_images, dim=0)
    #                     if len(target_images.shape) == 5: 
    #                         target_images = target_images.squeeze(1)
                
    #             if isinstance(target_images, torch.Tensor):
    #                 if len(target_images.shape) == 3: 
    #                     target_images = target_images.unsqueeze(0)
    #             else:

    #             tgt_mod = target_modalities[0] if isinstance(target_modalities, list) else target_modalities
                
    #             if isinstance(target_modalities, list):
    #                 self._current_batch_target_modality = target_modalities[0]
    #             else:
    #                 self._current_batch_target_modality = target_modalities

    #             target_latent = self.encode_to_latent(target_images, tgt_mod)
                
    #             if len(target_latent.shape) == 3: 
    #                 target_latent = target_latent.unsqueeze(0)

    #             # ============================================================
    #             # ============================================================
    #             condition_dict = {}
    #             source_rec = None
    #             source_latent = None
                
    #             if self.conditioning_mode != 'none':
    #                 source_images = batch.get("source_image", batch.get("condition_image"))
                    
    #                 source_modalities = batch.get("source_modality")
    #                 if source_modalities is None:

    #                 if source_images is None:

    #                 if isinstance(source_images, list):
    #                     if len(source_images) > 0 and isinstance(source_images[0], torch.Tensor):
    #                         source_images = torch.stack(source_images, dim=0)
    #                         if len(source_images.shape) == 5:
    #                             source_images = source_images.squeeze(1)
                    
    #                 if len(source_images.shape) == 3: 
    #                     source_images = source_images.unsqueeze(0)
                    
    #                 src_mod = source_modalities[0] if isinstance(source_modalities, list) else source_modalities
    #                 source_latent = self.encode_to_latent(source_images, src_mod)
                    
    #                 if len(source_latent.shape) == 3: 
    #                     source_latent = source_latent.unsqueeze(0)
                    
    #                 source_latent = self.adapter_manager.forward_before(source_latent, src_mod)


    #             # ============================================================
    #             # ============================================================
    #             if self.scale_factor is not None:
    #                 if isinstance(self.scale_factor, torch.Tensor):
    #                     sf = self.scale_factor.to(target_latent.device)
    #                 else:
    #                     sf = self.scale_factor
                    
    #                 if source_latent is not None:
    #                     source_latent = source_latent * sf
                    
    #                 target_latent = target_latent * sf

    #             if self.conditioning_mode != 'none' and source_latent is not None:
    #                 condition_dict = self.cond_stage_model(source_latent)

    #             # ============================================================
    #             # ============================================================
    #             if self._get_input_counter <= 50 or self._get_input_counter % 100 == 0:
    #                 tgt_mean = target_latent.mean().item()
    #                 tgt_std = target_latent.std().item()
    #                 tgt_max = target_latent.abs().max().item()
                    
    #                 src_std = 0.0
    #                 if source_latent is not None:
    #                     src_mean = source_latent.mean().item()
    #                     src_std = source_latent.std().item()
    #                     src_max = source_latent.abs().max().item()
    #                 else:
    #                     src_mean, src_max = 0, 0

    #                 print(f"   Target (Opt): Mean={tgt_mean:.4f}, Std={tgt_std:.4f}, Max=|{tgt_max:.4f}|")
    #                 if source_latent is not None:
    #                     print(f"   Source (SAR): Mean={src_mean:.4f}, Std={src_std:.4f}, Max=|{src_max:.4f}|")
                    
    #                 if tgt_std > 2.0 or (source_latent is not None and src_std > 2.0):
    #                     raise ValueError(
    #                     )

    #             # ============================================================
    #             # ============================================================
    #             if return_first_stage_outputs:
    #                 tgt_mod = target_modalities[0] if isinstance(target_modalities, list) else target_modalities
    #                 target_rec = self.decode_from_latent(target_latent, tgt_mod, is_scaled=True)
                    
    #                 if source_latent is not None:
    #                     src_mod = source_modalities[0] if isinstance(source_modalities, list) else source_modalities
    #                     source_rec = self.decode_from_latent(source_latent, src_mod, is_scaled=True)
                    
    #                 return target_latent, condition_dict, target_images, target_rec, source_rec
    #             else:
    #                 return target_latent, condition_dict

    #         except Exception as e:
    #             import traceback
    #             traceback.print_exc()
    #             raise
    def get_input(self, batch, k=None, return_first_stage_outputs=False, **kwargs):
        """
        []
        1.  scale_factor = 0.214
        2.  + 
        """
        bs = kwargs.get('bs', None)
        
        if not hasattr(self, '_get_input_counter'):
            self._get_input_counter = 0
        self._get_input_counter += 1
        
        try:
            # ============================================================
            # ============================================================
            target_images = batch.get("target_image", batch.get("image"))
            target_modalities = batch.get("target_modality")
            
            if target_modalities is None:

            if isinstance(target_images, list):
                if len(target_images) > 0 and isinstance(target_images[0], torch.Tensor):
                    target_images = torch.stack(target_images, dim=0)
                    if len(target_images.shape) == 5: target_images = target_images.squeeze(1)
            
            if isinstance(target_images, torch.Tensor):
                if len(target_images.shape) == 3: target_images = target_images.unsqueeze(0)
                if bs is not None and target_images.shape[0] > bs:
                    target_images = target_images[:bs]
            else:
                raise ValueError(f"target_images : {type(target_images)}")

            tgt_mod = target_modalities[0] if isinstance(target_modalities, list) else target_modalities
            
            if isinstance(target_modalities, list):
                self._current_batch_target_modality = target_modalities[0]
            else:
                self._current_batch_target_modality = target_modalities

            target_latent = self.encode_to_latent(target_images, tgt_mod)
            if len(target_latent.shape) == 3: target_latent = target_latent.unsqueeze(0)

            # ============================================================
            # ============================================================
            condition_dict = {}
            source_rec = None
            source_latent = None
            
            if self.conditioning_mode != 'none':
                source_images = batch.get("source_image", batch.get("condition_image"))
                source_modalities = batch.get("source_modality")
                
                if source_images is None: raise ValueError(f)

                if isinstance(source_images, list):
                    if len(source_images) > 0 and isinstance(source_images[0], torch.Tensor):
                        source_images = torch.stack(source_images, dim=0)
                        if len(source_images.shape) == 5: source_images = source_images.squeeze(1)
                
                if len(source_images.shape) == 3: source_images = source_images.unsqueeze(0)
                if bs is not None and source_images.shape[0] > bs:
                    source_images = source_images[:bs]
                
                src_mod = source_modalities[0] if isinstance(source_modalities, list) else source_modalities
                source_latent = self.encode_to_latent(source_images, src_mod)
                if len(source_latent.shape) == 3: source_latent = source_latent.unsqueeze(0)
                
                source_latent = self.adapter_manager.forward_before(source_latent, src_mod)

                # ============================================================
                # 3. Scale Factor (0.214)
                # ============================================================
                if self.scale_factor is not None:
                    if isinstance(self.scale_factor, torch.Tensor):
                        sf = self.scale_factor.to(target_latent.device)
                    else:
                        sf = self.scale_factor

                    source_latent = source_latent * sf
                    target_latent = target_latent * sf

                    if self._get_input_counter <= 1:

                condition_dict = self.cond_stage_model(source_latent)

            if self.conditioning_mode == 'none' and self.scale_factor is not None:
                 if isinstance(self.scale_factor, torch.Tensor):
                    sf = self.scale_factor.to(target_latent.device)
                 else:
                    sf = self.scale_factor
                 target_latent = target_latent * sf

            if return_first_stage_outputs:
                tgt_mod = target_modalities[0] if isinstance(target_modalities, list) else target_modalities
                target_rec = self.decode_from_latent(target_latent, tgt_mod, is_scaled=True)
                
                if source_latent is not None:
                    src_mod = source_modalities[0] if isinstance(source_modalities, list) else source_modalities
                    source_rec = self.decode_from_latent(source_latent, src_mod, is_scaled=True)
                
                return target_latent, condition_dict, target_images, target_rec, source_rec
            else:
                return target_latent, condition_dict

        except Exception as e:
            print(f"\n[ERROR] get_input : {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def on_fit_start(self):
        print(f"\n[DEBUG] on_fit_start ", flush=True)
        print(f"[DEBUG] global_rank: {getattr(self, 'global_rank', 'N/A')}", flush=True)
        print(f"[DEBUG] local_rank: {getattr(self, 'local_rank', 'N/A')}", flush=True)
        import sys
        sys.stdout.flush()
        sys.stderr.flush()
        super().on_fit_start()
    
    def on_train_start(self):
        print(f"\n[DEBUG] on_train_start ", flush=True)
        print(f"[DEBUG] current_epoch: {getattr(self, 'current_epoch', 'N/A')}", flush=True)
        import sys
        sys.stdout.flush()
        sys.stderr.flush()
        super().on_train_start()
    
    def on_train_epoch_start(self):
        if getattr(self, 'current_epoch', 0) == 0:
            print(f"\n[DEBUG] on_train_epoch_start  (epoch {getattr(self, 'current_epoch', 'N/A')})", flush=True)
            import sys
            sys.stdout.flush()
            sys.stderr.flush()
        super().on_train_epoch_start()
    
    def apply_model(self, x_noisy, t, cond, return_ids=False):
        """
        
        Args:
            x_noisy: 
            t: 
            cond: 
            return_ids: ID
            
        Returns:
            UNet
        """
        output = super().apply_model(x_noisy, t, cond, return_ids)
        
        return output
    
    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps, **kwargs):
        """
        
        -  torch.cat 
        """
        if isinstance(cond, dict):
            processed_cond = {}
            for k, v in cond.items():
                if isinstance(v, list) and len(v) > 0:
                    tensor = v[0] if isinstance(v[0], torch.Tensor) else v[0]
                    if isinstance(tensor, torch.Tensor) and tensor.shape[0] != batch_size:
                        if tensor.shape[0] > batch_size:
                            tensor = tensor[:batch_size]
                    processed_cond[k] = [tensor]  # 
                elif isinstance(v, torch.Tensor):
                    if v.shape[0] != batch_size:
                        if v.shape[0] > batch_size:
                            v = v[:batch_size]
                    processed_cond[k] = v
                else:
                    processed_cond[k] = v
            cond = processed_cond
        elif isinstance(cond, list):
            cond = cond[0] if len(cond) > 0 else cond
            if isinstance(cond, torch.Tensor) and cond.shape[0] != batch_size:
                if cond.shape[0] > batch_size:
                    cond = cond[:batch_size]
        
        return super().sample_log(cond, batch_size, ddim, ddim_steps, **kwargs)
    
    def p_losses(self, x_start, cond, t, noise=None):
        """
        
        - logvar  lvlb_weights 
        
        """
        # ========================================================================
        # ========================================================================
        if not hasattr(self, '_latent_check_counter'):
            self._latent_check_counter = 0
        
        try:
            has_global_step = hasattr(self, 'global_step')
            is_training = getattr(self, 'training', False)
            should_check = (has_global_step and is_training and 
                           ((self._latent_check_counter <= 10) or (self._latent_check_counter % 100 == 0)))
        except:
            should_check = False
            is_training = False
        
        if should_check:
            self._latent_check_counter += 1
            with torch.no_grad():
                x_start_stats = {
                    'min': x_start.min().item(),
                    'max': x_start.max().item(),
                    'mean': x_start.mean().item(),
                    'std': x_start.std().item(),
                    'abs_max': x_start.abs().max().item(),
                }
                
                noise_stats = {
                    'min': noise.min().item() if noise is not None else 0,
                    'max': noise.max().item() if noise is not None else 0,
                    'mean': noise.mean().item() if noise is not None else 0,
                    'std': noise.std().item() if noise is not None else 0,
                }
                
                try:
                    scale_factor_val = self.scale_factor.item() if isinstance(self.scale_factor, torch.Tensor) else self.scale_factor
                except (AttributeError, RuntimeError):
                    scale_factor_val = getattr(self, 'scale_factor', 1.0)
                    if isinstance(scale_factor_val, torch.Tensor):
                        scale_factor_val = scale_factor_val.item()
                
                if self._latent_check_counter <= 10 or self._latent_check_counter % 500 == 0:
                    print(f"\n{'='*80}")
                    print(f"{'='*80}")
                    print(f"   : [{x_start_stats['min']:.6f}, {x_start_stats['max']:.6f}]")
                    print(f"   : {x_start_stats['mean']:.6f}, : {x_start_stats['std']:.6f}")
                    print(f"   : {x_start_stats['abs_max']:.6f}")
                    print(f"   : {list(x_start.shape)}")
                    print(f"   : [{noise_stats['min']:.6f}, {noise_stats['max']:.6f}]")
                    print(f"   : {noise_stats['mean']:.6f}, : {noise_stats['std']:.6f}")
                    print(f"   : [{x_start_stats['min']/scale_factor_val:.6f}, {x_start_stats['max']/scale_factor_val:.6f}]")
                    print(f"   (: x_start)")
                    
                    abs_max = x_start_stats['abs_max']
                    if abs_max > 10.0:
                        print(f"   loss")
                    elif abs_max > 5.0:
                        print(f"   scale_factor")
                    elif abs_max < 0.1:
                        print(f"   ")
                    else:
                    
                    print(f"{'='*80}\n")
        
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_output = self.apply_model(x_noisy, t, cond)

        loss_dict = {}
        prefix = 'train' if self.training else 'val'

        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = noise
        else:
            raise NotImplementedError()

        loss_simple = self.get_loss(model_output, target, mean=False).mean([1, 2, 3])
        loss_dict.update({f'{prefix}/loss_simple': loss_simple.mean()})
        
        # ========================================================================
        # ========================================================================
        if should_check and is_training:
            with torch.no_grad():
                model_output_stats = {
                    'min': model_output.min().item(),
                    'max': model_output.max().item(),
                    'mean': model_output.mean().item(),
                    'std': model_output.std().item(),
                }
                target_stats = {
                    'min': target.min().item(),
                    'max': target.max().item(),
                    'mean': target.mean().item(),
                    'std': target.std().item(),
                }
                
                diff = (model_output - target).abs()
                diff_stats = {
                    'mean': diff.mean().item(),
                    'std': diff.std().item(),
                    'max': diff.max().item(),
                    'min': diff.min().item(),
                }
                
                target_range = target_stats['max'] - target_stats['min']
                relative_error = diff_stats['mean'] / (target_range + 1e-8) * 100
                
                if self._latent_check_counter <= 10 or self._latent_check_counter % 500 == 0:
                    print(f"   Model Output: [{model_output_stats['min']:.6f}, {model_output_stats['max']:.6f}], "
                          f"={model_output_stats['mean']:.6f}, ={model_output_stats['std']:.6f}")
                    print(f"   Target ({self.parameterization}): [{target_stats['min']:.6f}, {target_stats['max']:.6f}], "
                          f"={target_stats['mean']:.6f}, ={target_stats['std']:.6f}")
                    print(f"   : ={diff_stats['mean']:.6f}, ={diff_stats['std']:.6f}, "
                          f"={diff_stats['max']:.6f}, ={diff_stats['min']:.6f}")
                    print(f"   : {relative_error:.2f}% (target)")
                    print(f"   Loss Simple (batch): {loss_simple.mean().item():.6f}")
                    
                    if loss_simple.mean().item() > 1.0:
                        print(f"      - ")
                        print(f"      - ")
                        print(f"      - ")
                    elif loss_simple.mean().item() > 0.5:
                        print(f"      - ")
                        print(f"      - ")
                    print()

        logvar_t = self.logvar.to(self.device)[t]
        loss = loss_simple / torch.exp(logvar_t) + logvar_t
        if self.learn_logvar:
            loss_dict.update({f'{prefix}/loss_gamma': loss.mean()})
            loss_dict.update({'logvar': self.logvar.data.mean()})

        loss = self.l_simple_weight * loss.mean()

        loss_vlb = self.get_loss(model_output, target, mean=False).mean(dim=(1, 2, 3))
        loss_vlb = (self.lvlb_weights.to(self.device)[t] * loss_vlb).mean()
        loss_dict.update({f'{prefix}/loss_vlb': loss_vlb})
        loss += (self.original_elbo_weight * loss_vlb)
        loss_dict.update({f'{prefix}/loss': loss})
        
        # ========================================================================
        # ========================================================================
        if should_check and is_training and (self._latent_check_counter <= 10 or self._latent_check_counter % 500 == 0):
            print(f"   Loss Simple: {loss_dict[f'{prefix}/loss_simple']:.6f}")
            print(f"   Loss VLB: {loss_dict[f'{prefix}/loss_vlb']:.6f}")
            print(f"   VLB: {self.original_elbo_weight * loss_dict[f'{prefix}/loss_vlb']:.10f}")
            print(f"   Loss Total: {loss_dict[f'{prefix}/loss']:.6f}")
            print(f"    (Total - Simple): {loss_dict[f'{prefix}/loss'] - loss_dict[f'{prefix}/loss_simple']:.10f}")
            print()

        return loss, loss_dict
    
    def load_sd15_weights(self, sd15_ckpt_path):
        """
         ( Scale-Shift Norm  Transformer)
        """
        import os
        if not os.path.exists(sd15_ckpt_path):
            raise FileNotFoundError(f"SD1.5 checkpoint: {sd15_ckpt_path}")
        
        
        pl_sd = torch.load(sd15_ckpt_path, map_location="cpu")
        sd = pl_sd["state_dict"] if "state_dict" in pl_sd else pl_sd
        
        unet_sd = {}
        prefix = "model.diffusion_model."
        for k, v in sd.items():
            if k.startswith(prefix):
                unet_sd[k[len(prefix):]] = v
        
        if len(unet_sd) == 0:
            return
        
        input_key = "input_blocks.0.0.weight"
        
        if input_key in unet_sd:
            sd_weight = unet_sd[input_key]
            current_model_weight = self.model.diffusion_model.input_blocks[0][0].weight
            target_shape = current_model_weight.shape
            
            if sd_weight.shape != target_shape:
                
                if (sd_weight.shape[0] == target_shape[0] and 
                    sd_weight.shape[2:] == target_shape[2:] and 
                    target_shape[1] > sd_weight.shape[1]):
                    
                    
                    new_weight = torch.zeros_like(current_model_weight)
                    
                    src_channels = sd_weight.shape[1]
                    new_weight[:, :src_channels, :, :] = sd_weight
                    
                    unet_sd[input_key] = new_weight
                else:
                    del unet_sd[input_key]

        model_dict = self.model.diffusion_model.state_dict()
        filtered_sd = {}
        skipped_keys = []
        
        for k, v in unet_sd.items():
            if not self.use_spatial_transformer:
                if "transformer_blocks" in k or "attn1" in k or "attn2" in k:

            if k == "out.2.weight" or k == "out.2.bias":
                if k in model_dict and model_dict[k].shape != v.shape:
                continue
            
            if k in model_dict:
                if model_dict[k].shape == v.shape:
                    filtered_sd[k] = v
                else:
                    skipped_keys.append(f"{k}: Ckpt {list(v.shape)} vs Model {list(model_dict[k].shape)}")
            else:
                pass

        if len(skipped_keys) > 0:
            for msg in skipped_keys[:5]:
                print(f"  - {msg}")
            if len(skipped_keys) > 5: print(f"  - ... ( {len(skipped_keys)} )")

        m, u = self.model.diffusion_model.load_state_dict(filtered_sd, strict=False)
        
    
    def freeze_unet_encoder(self):
        """
        """
        unet = self.model.diffusion_model
        
        for param in unet.parameters():
            param.requires_grad = False
        
        for param in unet.output_blocks.parameters():
            param.requires_grad = True
            
        for param in unet.out.parameters():
            param.requires_grad = True
            
        for param in unet.middle_block.parameters():
            param.requires_grad = True
        
        trainable = sum(p.numel() for p in unet.parameters() if p.requires_grad)
        total = sum(p.numel() for p in unet.parameters())
    
    @contextmanager
    def ema_scope(self, context=None):
        """
        
        
        """
        if self.use_ema:
            self.model_ema.store(self.model.parameters())
            
            m_param = dict(self.model.named_parameters())
            shadow_params = dict(self.model_ema.named_buffers())
            
            for key in m_param:
                if key in self.model_ema.m_name2s_name:
                    if m_param[key].requires_grad:
                        s_name = self.model_ema.m_name2s_name[key]
                        if s_name in shadow_params:
                            m_param[key].data.copy_(shadow_params[s_name].data)
            
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.model.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")
    
    def on_before_optimizer_step(self, optimizer, optimizer_idx):
        """
        
        
        2. TensorBoard
        3. 
        
        - gradient_clip_val: 1.0
        - batch_size: 4
        - 8
        - accumulate_grad_batches: 2
        """
        if not hasattr(self, '_grad_monitor_counter'):
            self._grad_monitor_counter = 0
            self._grad_clip_count = 0  # 
        
        self._grad_monitor_counter += 1
        
        seen = set()
        parameters = []
        for group in optimizer.param_groups:
            for p in group.get('params', []):
                if p is None:
                    continue
                pid = id(p)
                if pid in seen:
                    continue
                seen.add(pid)
                if p.requires_grad and p.grad is not None:
                    parameters.append(p)
        
        if len(parameters) == 0:
            if hasattr(super(), 'on_before_optimizer_step'):
                super().on_before_optimizer_step(optimizer, optimizer_idx)
            return
        
        total_norm_before = torch.nn.utils.clip_grad_norm_(parameters, max_norm=float('inf'))
        
        gradient_clip_val = getattr(self.trainer, 'gradient_clip_val', None)
        if gradient_clip_val is None:
            if hasattr(self.trainer, 'gradient_clip_val'):
                gradient_clip_val = self.trainer.gradient_clip_val
            else:
                gradient_clip_val = 1.0  # 
        
        is_clipped = total_norm_before > gradient_clip_val
        if is_clipped:
            self._grad_clip_count += 1
        
        if not hasattr(self, '_grad_clip_history'):
            self._grad_clip_history = []
        self._grad_clip_history.append(1 if is_clipped else 0)
        if len(self._grad_clip_history) > 100:
            self._grad_clip_history.pop(0)
        clip_rate = sum(self._grad_clip_history) / len(self._grad_clip_history)
        
        self.log('train/grad_norm_before_clip', total_norm_before.item(), 
                on_step=True, on_epoch=False, logger=True, prog_bar=False)
        self.log('train/grad_is_clipped', float(is_clipped), 
                on_step=True, on_epoch=False, logger=True, prog_bar=False)
        self.log('train/grad_clip_rate', clip_rate, 
                on_step=True, on_epoch=False, logger=True, prog_bar=False)
        
        if total_norm_before > 10.0:
            self.log('train/grad_norm_warning', 1.0, 
                    on_step=True, on_epoch=False, logger=True, prog_bar=False)
        
        if self._grad_monitor_counter % 100 == 0:
            if hasattr(self, 'trainer') and self.trainer is not None:
                if self.trainer.global_rank == 0:
                    print(f"\n[] Step {self._grad_monitor_counter}:")
                    print(f"  : {gradient_clip_val}")
                    print(f"  : { if is_clipped else }")
                    print(f"  100: {clip_rate*100:.1f}%")
                    print(f"  : {self._grad_clip_count}/{self._grad_monitor_counter} ({self._grad_clip_count/self._grad_monitor_counter*100:.1f}%)")
                    
                    if clip_rate > 0.8:
                        print(f"     :")
                        print(f"     1. ")
                        print(f"     2. ")
                        print(f"     3. ")
                        print(f"     :  {total_norm_before.item() * 0.8:.2f}")
                    elif clip_rate > 0.5:
            else:
                print(f"\n[] Step {self._grad_monitor_counter}:")
                print(f"  : {gradient_clip_val}")
                print(f"  : { if is_clipped else }")
                print(f"  100: {clip_rate*100:.1f}%")
                print(f"  : {self._grad_clip_count}/{self._grad_monitor_counter} ({self._grad_clip_count/self._grad_monitor_counter*100:.1f}%)")
                
                if clip_rate > 0.8:
                elif clip_rate > 0.5:
        
        if hasattr(super(), 'on_before_optimizer_step'):
            super().on_before_optimizer_step(optimizer, optimizer_idx)
    
    def on_train_batch_end(self, *args, **kwargs):
        """
        
        
        """
        if self.use_ema:
            decay = self.model_ema.decay
            if self.model_ema.num_updates >= 0:
                self.model_ema.num_updates += 1
                decay = min(self.model_ema.decay, 
                           (1 + self.model_ema.num_updates) / (10 + self.model_ema.num_updates))
            
            one_minus_decay = 1.0 - decay
            
            with torch.no_grad():
                m_param = dict(self.model.named_parameters())
                shadow_params = dict(self.model_ema.named_buffers())
                
                for key in m_param:
                    if key in self.model_ema.m_name2s_name:
                        if m_param[key].requires_grad:
                            sname = self.model_ema.m_name2s_name[key]
                            if sname in shadow_params:
                                shadow_params[sname] = shadow_params[sname].type_as(m_param[key])
                                shadow_params[sname].sub_(one_minus_decay * 
                                                         (shadow_params[sname] - m_param[key]))

    def on_load_checkpoint(self, checkpoint):
        if getattr(self, "learn_logvar", False):
            optimizer_states = checkpoint.get("optimizer_states")
            if isinstance(optimizer_states, list) and optimizer_states:
                param_groups = optimizer_states[0].get("param_groups")
                if isinstance(param_groups, list) and len(param_groups) == 1:
                    checkpoint.pop("optimizer_states", None)
                    checkpoint.pop("lr_schedulers", None)
    
    def configure_optimizers(self):
        """
        [] 
        """
        lr = self.learning_rate
        model_params = []
        
        unet = self.model.diffusion_model
        for name, param in unet.named_parameters():
            if param.requires_grad:
                model_params.append(param)
        
        if self.enable_adapters:
            model_params.extend(self.adapter_manager.parameters())
        
        if hasattr(self.cond_stage_model, 'parameters') and self.cond_stage_trainable:
            model_params.extend(self.cond_stage_model.parameters())
        
        # -----------------------------------------------------------
        # -----------------------------------------------------------
        param_groups = [{"params": model_params, "weight_decay": 1e-2}]
        if getattr(self, "learn_logvar", False) and hasattr(self, "logvar") and getattr(self.logvar, "requires_grad", False):
            param_groups.append({"params": [self.logvar], "weight_decay": 0.0})
        opt = torch.optim.AdamW(param_groups, lr=lr)
        
        # -----------------------------------------------------------
        # -----------------------------------------------------------
        def lambda_lr(step):
            warmup_steps = 2000 
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            return 1.0

        scheduler = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda_lr),
            "interval": "step",
            "frequency": 1,
        }
        
        return [opt], [scheduler]
    
    def set_adapter_enable(self, enable):
        """
        
        Args:
        """
        self.enable_adapters = enable
        self.adapter_manager.set_enable(enable)
    
    def set_conditioning_mode(self, mode):
        """
        
        Args:
            mode: 'concat', 'crossattn', 'hybrid',  'none'
        """
        if mode not in ['concat', 'crossattn', 'hybrid', 'none']:
                           f": 'concat', 'crossattn', 'hybrid', 'none'")
        
        self.conditioning_mode = mode
        self.cond_stage_model.conditioning_mode = mode
        print(f": {mode}")
    
    @torch.no_grad()
    def log_images(self, batch, N=8, n_row=4, sample=True, ddim_steps=200, ddim_eta=1., return_keys=None,
                   quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True,
                   plot_diffusion_rows=True, **kwargs):
        """
        
        -  batch  target_modality 
        """
        use_ddim = ddim_steps is not None

        log = dict()

        def _to_3ch(bchw: torch.Tensor) -> torch.Tensor:
            if bchw.dim() != 4:
                raise ValueError(f"Expected 4D tensor, got {bchw.shape}")
            c = bchw.shape[1]
            if c == 3:
                return bchw
            if c == 1:
                return bchw.repeat(1, 3, 1, 1)
            if c == 2:
                gray = bchw.mean(dim=1, keepdim=True)
                return gray.repeat(1, 3, 1, 1)
            return bchw[:, :3, ...]

        def _to_log_bchw(x_in):
            if x_in is None:
                return None
            if isinstance(x_in, torch.Tensor):
                t = x_in
                if t.dim() == 3:
                    t = t.unsqueeze(0)
                if t.dim() != 4:
                    return None
                return _to_3ch(t)
            if isinstance(x_in, list):
                items = []
                for item in x_in:
                    if not isinstance(item, torch.Tensor):
                        continue
                    t = item
                    if t.dim() == 4 and t.shape[0] == 1:
                        t = t[0]
                    if t.dim() != 3:
                        continue
                    t3 = _to_3ch(t.unsqueeze(0))[0]
                    items.append(t3)
                if not items:
                    return None
                return torch.stack(items, dim=0)
            return None

        z, c, x, xrec, xc = self.get_input(batch, self.first_stage_key,
                                           return_first_stage_outputs=True,
                                           force_c_encode=True,
                                           return_original_cond=True,
                                           bs=N)
        N = min(x.shape[0], N)
        n_row = min(x.shape[0], n_row)
        log["inputs"] = x
        log["reconstruction"] = xrec

        source_modalities = batch.get("source_modality", None)
        source_modality = source_modalities[0] if isinstance(source_modalities, list) and len(source_modalities) > 0 else source_modalities
        source_modality = str(source_modality).lower() if source_modality is not None else "source"

        source_images = _to_log_bchw(batch.get("source_image", None))
        if source_images is not None:
            log[f"source_{source_modality}"] = source_images[:N]

        target_modalities = batch.get("target_modality", None)
        if target_modalities is not None:
            target_modality = target_modalities[0] if isinstance(target_modalities, list) else target_modalities
            target_modality = target_modality.lower()
        else:
            target_modality = self.modalities[0] if hasattr(self, 'modalities') and len(self.modalities) > 0 else None

        if target_modality is not None:
            log[f"target_{target_modality}_gt"] = _to_3ch(x)[:N]

        if self.model.conditioning_key is not None:
            if hasattr(self.cond_stage_model, "decode"):
                xc = self.cond_stage_model.decode(c)
                log["conditioning"] = xc
            elif self.cond_stage_key in ["caption"]:
                xc = log_txt_as_img((x.shape[2], x.shape[3]), batch["caption"])
                log["conditioning"] = xc
            elif self.cond_stage_key == 'class_label':
                xc = log_txt_as_img((x.shape[2], x.shape[3]), batch["human_label"])
                log['conditioning'] = xc
            elif isimage(xc):
                log["conditioning"] = xc
            if ismap(xc):
                log["original_conditioning"] = self.to_rgb(xc)

        if plot_diffusion_rows:
            # get diffusion row
            diffusion_row = list()
            z_start = z[:n_row]
            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                    t = t.to(self.device).long()
                    noise = torch.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
                    diffusion_row.append(self.decode_first_stage(z_noisy, modality=target_modality))

            diffusion_row = torch.stack(diffusion_row)  # n_log_step, n_row, C, H, W
            diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
            diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            log["diffusion_row"] = diffusion_grid

        if sample:
            # get denoise row
            with self.ema_scope("Plotting"):
                samples, z_denoise_row = self.sample_log(cond=c, batch_size=N, ddim=use_ddim,
                                                         ddim_steps=ddim_steps, eta=ddim_eta)
            x_samples = self.decode_first_stage(samples, modality=target_modality)
            log["samples"] = x_samples
            if source_images is not None and target_modality is not None:
                log[f"compare_{source_modality}_to_{target_modality}"] = torch.cat(
                    [source_images[:N], _to_3ch(x)[:N], _to_3ch(x_samples)[:N]], dim=3
                )
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

            if quantize_denoised:
                # also display when quantizing x0 while sampling
                target_ae = self.get_ae(target_modality) if target_modality else self.first_stage_model
                if not isinstance(target_ae, AutoencoderKL) and not isinstance(target_ae, IdentityFirstStage):
                    with self.ema_scope("Plotting Quantized Denoised"):
                        samples, z_denoise_row = self.sample_log(cond=c, batch_size=N, ddim=use_ddim,
                                                                 ddim_steps=ddim_steps, eta=ddim_eta,
                                                                 quantize_denoised=True)
                    x_samples = self.decode_first_stage(samples.to(self.device), modality=target_modality)
                    log["samples_x0_quantized"] = x_samples

            if inpaint:
                # make a simple center square
                b, h, w = z.shape[0], z.shape[2], z.shape[3]
                mask = torch.ones(N, h, w).to(self.device)
                # zeros will be filled in
                mask[:, h // 4:3 * h // 4, w // 4:3 * w // 4] = 0.
                mask = mask[:, None, ...]
                with self.ema_scope("Plotting Inpaint"):
                    samples, _ = self.sample_log(cond=c, batch_size=N, ddim=use_ddim, eta=ddim_eta,
                                                ddim_steps=ddim_steps, x0=z[:N], mask=mask)
                x_samples = self.decode_first_stage(samples.to(self.device), modality=target_modality)
                log["samples_inpainting"] = x_samples
                log["mask"] = mask

                # outpaint
                with self.ema_scope("Plotting Outpaint"):
                    samples, _ = self.sample_log(cond=c, batch_size=N, ddim=use_ddim, eta=ddim_eta,
                                                ddim_steps=ddim_steps, x0=z[:N], mask=mask)
                x_samples = self.decode_first_stage(samples.to(self.device), modality=target_modality)
                log["samples_outpainting"] = x_samples

        if plot_progressive_rows:
            with self.ema_scope("Plotting Progressives"):
                img, progressives = self.progressive_denoising(c,
                                                               shape=(self.channels, self.image_size, self.image_size),
                                                               batch_size=N)
            prog_row = self._get_denoise_row_from_list(progressives, desc="Progressive Generation")
            log["progressive_row"] = prog_row

        if return_keys:
            if isinstance(return_keys, str):
                return_keys = [return_keys]
            if np.intersect1d(list(log.keys()), return_keys).shape[0] == 0:
                return log
            else:
                return {key: log[key] for key in return_keys}
        return log
