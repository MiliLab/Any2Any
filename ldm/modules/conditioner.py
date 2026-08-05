"""
- concat: 
- crossattn: 
- hybrid: concatcrossattn
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConcatConditioner(nn.Module):
    """
    """
    
    def __init__(self, condition_channels=3, use_modality_embedding=False, num_modalities=5):
        """
        :
        :param use_modality_embedding: 
        :param num_modalities: 
        """
        super().__init__()
        
        self.condition_channels = condition_channels
        self.use_modality_embedding = use_modality_embedding
        
        if use_modality_embedding:
            self.source_mod_embedding = nn.Embedding(num_modalities, condition_channels)
            self.target_mod_embedding = nn.Embedding(num_modalities, condition_channels)
    
    def forward(self, condition_latent, source_modality_id=None, target_modality_id=None):
        """
        concat
        
        Args:
            condition_latent:  [B, C, H, W]
            
        Returns:
             {"c_concat": [condition]}
        """
        condition = condition_latent
        
        if self.use_modality_embedding and source_modality_id is not None:
            B, C, H, W = condition_latent.shape
            device = condition_latent.device
            
            source_emb = self.source_mod_embedding(source_modality_id.to(device))  # [B, C]
            target_emb = self.target_mod_embedding(target_modality_id.to(device))  # [B, C]
            
            source_emb = source_emb.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)  # [B, C, H, W]
            target_emb = target_emb.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)  # [B, C, H, W]
            
            condition = torch.cat([condition_latent, source_emb, target_emb], dim=1)  # [B, 3*C, H, W]
        
        return {"c_concat": [condition]}


class CrossAttentionConditioner(nn.Module):
    """
    
    """
    
    def __init__(self, 
                 condition_channels=3,
                 context_dim=512,
                 use_modality_embedding=False,
                 num_modalities=5,
        """
        :
        :param condition_channels: 
        :param use_modality_embedding: 
        :param num_modalities: 
        :param preserve_spatial: 
        """
        super().__init__()
        
        self.condition_channels = condition_channels
        self.context_dim = context_dim
        self.use_modality_embedding = use_modality_embedding
        self.preserve_spatial = preserve_spatial
        
        if preserve_spatial:
            # [B, C, H, W] -> [B, H*W, C] -> [B, H*W, context_dim]
            self.compressor = nn.Sequential(
                nn.Conv2d(condition_channels, context_dim // 4, kernel_size=1),
                nn.GELU(),
                nn.Conv2d(context_dim // 4, context_dim // 2, kernel_size=1),
                nn.GELU(),
                nn.Conv2d(context_dim // 2, context_dim, kernel_size=1),
            )
        else:
            self.compressor = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),  # [B, C, 1, 1]
                nn.Flatten(),  # [B, C]
                nn.Linear(condition_channels, context_dim * 2),
                nn.GELU(),
                nn.Linear(context_dim * 2, context_dim)  # [B, context_dim]
            )
        
        if use_modality_embedding:
            self.source_mod_embedding = nn.Embedding(num_modalities, context_dim // 2)
            self.target_mod_embedding = nn.Embedding(num_modalities, context_dim // 2)
    
    def forward(self, condition_latent, source_modality_id=None, target_modality_id=None):
        """
        cross-attention
        
        Args:
            condition_latent:  [B, C, H, W]
            
        Returns:
             {"c_crossattn": [context]}
        """
        B, C, H, W = condition_latent.shape
        
        if self.preserve_spatial:
            # [B, C, H, W] -> [B, context_dim, H, W]
            context_spatial = self.compressor(condition_latent)  # [B, context_dim, H, W]
            
            context = context_spatial.permute(0, 2, 3, 1).reshape(B, H * W, self.context_dim)
            
            if self.use_modality_embedding and source_modality_id is not None:
                device = condition_latent.device
                source_emb = self.source_mod_embedding(source_modality_id.to(device))  # [B, context_dim//2]
                target_emb = self.target_mod_embedding(target_modality_id.to(device))  # [B, context_dim//2]
                
                modality_emb = torch.cat([source_emb, target_emb], dim=1)  # [B, context_dim]
                modality_emb = modality_emb.unsqueeze(1).expand(-1, H * W, -1)  # [B, H*W, context_dim]
                
                context = torch.cat([context, modality_emb], dim=2)  # [B, H*W, context_dim*2]
                if not hasattr(self, 'spatial_dim_adapter'):
                    self.spatial_dim_adapter = nn.Linear(context.shape[2], self.context_dim).to(device)
                context = self.spatial_dim_adapter(context)  # [B, H*W, context_dim]
        else:
            context = self.compressor(condition_latent)  # [B, context_dim]
            
            if self.use_modality_embedding and source_modality_id is not None:
                device = condition_latent.device
                source_emb = self.source_mod_embedding(source_modality_id.to(device))  # [B, context_dim//2]
                target_emb = self.target_mod_embedding(target_modality_id.to(device))  # [B, context_dim//2]
                
                context = torch.cat([context, source_emb, target_emb], dim=1)  # [B, context_dim + context_dim]
                
                if context.shape[1] != self.context_dim:
                    if not hasattr(self, 'dim_adapter'):
                        self.dim_adapter = nn.Linear(context.shape[1], self.context_dim).to(device)
                    context = self.dim_adapter(context)  # [B, context_dim]
            
            context = context.unsqueeze(1)
        
        return {"c_crossattn": [context]}


class HybridConditioner(nn.Module):
    """
    """
    
    def __init__(self, 
                 condition_channels=3,
                 context_dim=512,
                 use_modality_embedding=False,
                 num_modalities=5,
                 preserve_spatial=True):  # 
        """
        :
        :param condition_channels: 
        :param context_dim: cross-attentioncontext
        :param use_modality_embedding: 
        :param num_modalities: 
        :param preserve_spatial: crossattn
        """
        super().__init__()
        
        self.concat_conditioner = ConcatConditioner(
            condition_channels=condition_channels,
            use_modality_embedding=use_modality_embedding,
            num_modalities=num_modalities
        )
        
        self.crossattn_conditioner = CrossAttentionConditioner(
            condition_channels=condition_channels,
            context_dim=context_dim,
            use_modality_embedding=use_modality_embedding,
            num_modalities=num_modalities,
            preserve_spatial=preserve_spatial
        )
    
    def forward(self, condition_latent, source_modality_id=None, target_modality_id=None):
        """
        
        Args:
            condition_latent:  [B, C, H, W]
            source_modality_id: ID
            target_modality_id: ID
            
        Returns:
             {"c_concat": [...], "c_crossattn": [...]}
        """
        concat_dict = self.concat_conditioner(condition_latent, source_modality_id, target_modality_id)
        crossattn_dict = self.crossattn_conditioner(condition_latent, source_modality_id, target_modality_id)
        
        return {**concat_dict, **crossattn_dict}


class UnconditionalConditioner(nn.Module):
    """
    """
    
    def __init__(self):
        """
        """
        super().__init__()
    
    def forward(self, condition_latent=None, source_modality_id=None, target_modality_id=None):
        """
        
        Args:
            
        Returns:
        """


class FlexibleConditioner(nn.Module):
    """
    """
    
    def __init__(self,
                 conditioning_mode='concat',  # 'concat', 'crossattn', 'hybrid', 'none'
                 condition_channels=3,
                 context_dim=512,
                 use_modality_embedding=False,
                 num_modalities=5,
        """
        :
        :param conditioning_mode: 
            - 'concat': concat
            - 'crossattn': cross-attention
            - 'hybrid': concatcross-attention
        :param condition_channels: 
        :param context_dim: cross-attentioncontext
        :param use_modality_embedding: 
        :param num_modalities: 
        """
        super().__init__()
        
        self.conditioning_mode = conditioning_mode
        self.condition_channels = condition_channels
        self.context_dim = context_dim
        
        if conditioning_mode == 'concat':
            self.conditioner = ConcatConditioner(
                condition_channels=condition_channels,
                use_modality_embedding=use_modality_embedding,
                num_modalities=num_modalities
            )
        elif conditioning_mode == 'crossattn':
            self.conditioner = CrossAttentionConditioner(
                condition_channels=condition_channels,
                context_dim=context_dim,
                use_modality_embedding=use_modality_embedding,
                num_modalities=num_modalities,
                preserve_spatial=preserve_spatial
            )
        elif conditioning_mode == 'hybrid':
            self.conditioner = HybridConditioner(
                condition_channels=condition_channels,
                context_dim=context_dim,
                use_modality_embedding=use_modality_embedding,
                num_modalities=num_modalities,
                preserve_spatial=preserve_spatial
            )
        elif conditioning_mode == 'none':
            self.conditioner = UnconditionalConditioner()
        else:
                           f": 'concat', 'crossattn', 'hybrid', 'none'")
    
    def forward(self, condition_latent=None, source_modality_id=None, target_modality_id=None):
        """
        
        
        Args:
            
        Returns:
        """
        return self.conditioner(condition_latent, source_modality_id, target_modality_id)
    
    def set_mode(self, mode):
        """
        
        Args:
            mode: 
        """
        self.conditioning_mode = mode
