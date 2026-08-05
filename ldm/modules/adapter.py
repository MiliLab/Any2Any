"""
"""

import torch
import torch.nn as nn


class _SE(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        c = int(channels)
        r = max(1, c // int(reduction))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(c, r, kernel_size=1)
        self.act = nn.SiLU()
        self.fc2 = nn.Conv2d(r, c, kernel_size=1)

    def forward(self, x):
        w = self.pool(x)
        w = self.fc1(w)
        w = self.act(w)
        w = self.fc2(w)
        w = torch.sigmoid(w)
        return x * w


class _RefineBlock(nn.Module):
    def __init__(self, channels, use_se=True):
        super().__init__()
        c = int(channels)
        self.norm1 = nn.GroupNorm(min(8, c), c)
        self.conv1 = nn.Conv2d(c, c, kernel_size=3, padding=1)
        self.act = nn.SiLU()
        self.norm2 = nn.GroupNorm(min(8, c), c)
        self.conv2 = nn.Conv2d(c, c, kernel_size=3, padding=1)
        self.se = _SE(c) if bool(use_se) else nn.Identity()

    def forward(self, x):
        h = self.norm1(x)
        h = self.conv1(h)
        h = self.act(h)
        h = self.norm2(h)
        h = self.conv2(h)
        h = self.se(h)
        return x + h


class LatentAdapter(nn.Module):
    """
    
    
    
    1x1
    """
    
    def __init__(
                 self,
                 in_channels=3,
                 out_channels=3,
                 hidden_channels=None,
                 use_residual=True,
                 adapter_type='simple',
                 depth=3,
                 use_se=True,
                 zero_init=True):
        """
        :
        :param in_channels: 
        :param out_channels: 
        :param use_residual: 
        :param adapter_type: 
            - 'simple': 1x1
            - 'residual': 
            - 'attention': 
        :param depth: refineblock
        :param use_se: refineSE
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_residual = use_residual and (in_channels == out_channels)
        self.adapter_type = adapter_type
        
        if hidden_channels is None:
            hidden_channels = max(in_channels, out_channels)
        
        if adapter_type == 'simple':
            self.adapter = nn.Sequential(
                nn.GroupNorm(min(8, in_channels), in_channels),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
            )
            
        elif adapter_type == 'residual':
            self.adapter = nn.Sequential(
                nn.GroupNorm(min(8, in_channels), in_channels),
                nn.Conv2d(in_channels, hidden_channels, kernel_size=3, stride=1, padding=1),
                nn.SiLU(),
                nn.GroupNorm(min(8, hidden_channels), hidden_channels),
                nn.Conv2d(hidden_channels, out_channels, kernel_size=1, stride=1, padding=0)
            )
            
        elif adapter_type == 'attention':
            self.adapter = nn.ModuleDict({
                'norm1': nn.GroupNorm(min(8, in_channels), in_channels),
                'conv1': nn.Conv2d(in_channels, hidden_channels, kernel_size=1),
                'attention': nn.MultiheadAttention(
                    embed_dim=hidden_channels,
                    num_heads=max(1, hidden_channels // 64),
                    batch_first=False
                ),
                'norm2': nn.GroupNorm(min(8, hidden_channels), hidden_channels),
                'conv2': nn.Conv2d(hidden_channels, out_channels, kernel_size=1),
            })
        elif adapter_type == 'refine':
            d = max(1, int(depth))
            self.adapter = nn.ModuleDict({
                'in_norm': nn.GroupNorm(min(8, in_channels), in_channels),
                'in_conv': nn.Conv2d(in_channels, hidden_channels, kernel_size=3, stride=1, padding=1),
                'blocks': nn.ModuleList([_RefineBlock(hidden_channels, use_se=bool(use_se)) for _ in range(d)]),
                'out_norm': nn.GroupNorm(min(8, hidden_channels), hidden_channels),
                'out_act': nn.SiLU(),
                'out_conv': nn.Conv2d(hidden_channels, out_channels, kernel_size=3, stride=1, padding=1),
            })
            if bool(zero_init) and bool(self.use_residual):
                nn.init.zeros_(self.adapter['out_conv'].weight)
                nn.init.zeros_(self.adapter['out_conv'].bias)
        else:
            raise ValueError(f"adapter: {adapter_type}")
    
    def forward(self, x):
        """
        
        
        Args:
            x:  [B, C, H, W]
            
        Returns:
             [B, C_out, H, W]
        """
        if self.adapter_type == 'simple':
            out = self.adapter(x)
            
        elif self.adapter_type == 'residual':
            out = self.adapter(x)
            
        elif self.adapter_type == 'attention':
            B, C, H, W = x.shape
            x_norm = self.adapter['norm1'](x)
            x_conv = self.adapter['conv1'](x_norm)
            x_flat = x_conv.flatten(2).permute(2, 0, 1)
            x_attn, _ = self.adapter['attention'](x_flat, x_flat, x_flat)
            x_attn = x_attn.permute(1, 2, 0).reshape(B, -1, H, W)
            x_attn = self.adapter['norm2'](x_attn)
            out = self.adapter['conv2'](x_attn)
        elif self.adapter_type == 'refine':
            h = self.adapter['in_norm'](x)
            h = self.adapter['in_conv'](h)
            for blk in self.adapter['blocks']:
                h = blk(h)
            h = self.adapter['out_norm'](h)
            h = self.adapter['out_act'](h)
            out = self.adapter['out_conv'](h)
        else:
            raise ValueError(f"adapter: {self.adapter_type}")
        
        if self.use_residual:
            out = out + x
        
        return out


class MultiModalAdapterManager(nn.Module):
    """
    
    
    """
    
    def __init__(
        self,
        modalities=['opt', 'sar', 'pan', 'nir', 'ms'],
        latent_channels=3,
        enable_adapters=True,
        adapter_type='simple',
        adapter_configs=None,
        build_before=True,
        build_after=True,
    ): 
        """
        :
            {
                'opt': {'before': {...}, 'after': {...}},
                'sar': {'before': {...}, 'after': {...}},
                ...
            }
        """
        super().__init__()
        
        self.modalities = modalities
        self.num_modalities = len(modalities)
        self.latent_channels = latent_channels
        self.enable_adapters = enable_adapters
        self.adapter_type = adapter_type
        
        self.build_before = bool(build_before)
        self.build_after = bool(build_after)

        self.adapters_before = nn.ModuleDict() if self.build_before else None
        self.adapters_after = nn.ModuleDict() if self.build_after else None

        for mod in modalities:
            mod_config = adapter_configs.get(mod, {}) if adapter_configs else {}
            before_config = mod_config.get('before', {})
            after_config = mod_config.get('after', {})

            if self.build_before:
                self.adapters_before[mod] = LatentAdapter(
                    in_channels=latent_channels,
                    out_channels=latent_channels,
                    adapter_type=before_config.get('type', adapter_type),
                    use_residual=before_config.get('use_residual', True),
                    hidden_channels=before_config.get('hidden_channels', None),
                    depth=before_config.get('depth', 3),
                    use_se=before_config.get('use_se', True),
                    zero_init=before_config.get('zero_init', True),
                )

            if self.build_after:
                self.adapters_after[mod] = LatentAdapter(
                    in_channels=latent_channels,
                    out_channels=latent_channels,
                    adapter_type=after_config.get('type', adapter_type),
                    use_residual=after_config.get('use_residual', True),
                    hidden_channels=after_config.get('hidden_channels', None),
                    depth=after_config.get('depth', 3),
                    use_se=after_config.get('use_se', True),
                    zero_init=after_config.get('zero_init', True),
                )

        print(f" {self.num_modalities} ")
        print(f"  adapter_before: { if self.build_before else }")
        print(f"  adapter_after: { if self.build_after else }")
        print(f"  : { if enable_adapters else }")
    
    def forward_before(self, x, modality):
        """
        UNetadapter
        
        Args:
            x:  [B, C, H, W]
            
        Returns:
             [B, C, H, W]
        """
        if (not self.enable_adapters) or (not bool(self.build_before)) or (self.adapters_before is None):
            return x

        if modality not in self.adapters_before:
            raise ValueError(f": {modality}")

        return self.adapters_before[modality](x)
    
    def forward_after(self, x, modality):
        """
        UNetadapter
        
        Args:
            x: UNet [B, C, H, W]
            
        Returns:
             [B, C, H, W]
        """
        if (not self.enable_adapters) or (not bool(self.build_after)) or (self.adapters_after is None):
            return x

        if modality not in self.adapters_after:
            raise ValueError(f": {modality}")

        return self.adapters_after[modality](x)
    
    def set_enable(self, enable):
        """
        /adapter
        
        Args:
        """
        self.enable_adapters = enable
        print(f"{ if enable else }")
    
    def get_adapter_params(self):
        params = []
        for mod in self.modalities:
            if self.adapters_before is not None and mod in self.adapters_before:
                params.extend(list(self.adapters_before[mod].parameters()))
            if self.adapters_after is not None and mod in self.adapters_after:
                params.extend(list(self.adapters_after[mod].parameters()))
        return params

    def get_after_adapter_params(self):
        params = []
        if self.adapters_after is None:
            return params
        for mod in self.modalities:
            if mod in self.adapters_after:
                params.extend(list(self.adapters_after[mod].parameters()))
        return params
