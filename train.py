# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT using PyTorch DDP.
"""
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data.dataloader import default_collate
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict, Counter
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os
import sys
import json
import csv
import hashlib
import heapq
import math

from models import DiT_models
from diffusion import create_diffusion
from diffusion.gaussian_diffusion import _extract_into_tensor
from diffusers.models import AutoencoderKL
from torchvision.utils import make_grid, save_image

from ldm.modules.adapter import MultiModalAdapterManager


try:
    from skimage.metrics import structural_similarity
    SKIMAGE_AVAILABLE = True
except Exception:
    SKIMAGE_AVAILABLE = False


#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def _count_params(m, only_trainable=False):
    if m is None:
        return 0
    if hasattr(m, "module"):
        m = m.module
    total = 0
    for p in m.parameters():
        if only_trainable and (not bool(getattr(p, "requires_grad", False))):
            continue
        total += int(p.numel())
    return int(total)


def _log_config_summary(
    *,
    args,
    logger,
    rank,
    device,
    world_size,
    use_manifest,
    resumed,
    train_steps,
    dataset,
    loader,
    ae_models_by_id,
    ae_info_by_modality,
    pair_weight_table,
    model,
    ema,
    eval_loaders,
):
    if int(rank) != 0:
        return

    def _yn(v):
        return  if bool(v) else 

    def _on(v):
        return  if bool(v) else 

    per_gpu_batch = int(args.global_batch_size // max(1, int(world_size)))

    mods = [str(m).lower() for m in (getattr(args, "modalities", []) or [])]


    main_target = "x0" if bool(getattr(args, "predict_xstart", False)) else "eps"
    main_snr_gamma = getattr(args, "x0_loss_snr_gamma", None)
    main_snr_gamma = None if main_snr_gamma is None else float(main_snr_gamma)

    wmin = None
    wmax = None
    if pair_weight_table is not None:
        try:
            wmin = float(pair_weight_table.min().item())
            wmax = float(pair_weight_table.max().item())
        except Exception:
            wmin = None
            wmax = None

    ae_lines = []
    if isinstance(ae_info_by_modality, dict) and ae_info_by_modality:
        for m in mods:
            meta = ae_info_by_modality.get(m, None)
            if isinstance(meta, dict):
                ck = meta.get("ckpt_resolved", None)
                used = meta.get("used_override", None)

    eval_lines = []
    if isinstance(eval_loaders, dict) and eval_loaders:
        items = []
        for _, it in eval_loaders.items():
            try:
                split = str(it.get("split", ""))
                pair_tag = str(it.get("pair_tag", ""))
                ld = it.get("loader")
                n = "?" if ld is None else str(len(getattr(ld, "dataset", [])))
                items.append(f"{split}:{pair_tag}(n={n})")
            except Exception:
                continue
        if items:
            s = ", ".join(items[:12])
            if len(items) > 12:
            eval_lines.append(f"- : {s}")

    dit_total = int(_count_params(model, only_trainable=False))
    dit_train = int(_count_params(model, only_trainable=True))
    ema_total = int(_count_params(ema, only_trainable=False))
    ema_train = int(_count_params(ema, only_trainable=True))

    ae_total = 0
    ae_train = 0
    if isinstance(ae_models_by_id, dict):
        for _, ae in ae_models_by_id.items():
            ae_total += int(_count_params(ae, only_trainable=False))
            ae_train += int(_count_params(ae, only_trainable=True))

    lines = []
    lines.append("=" * 80)

    if bool(use_manifest):
        lines.append(f"  - manifest={str(getattr(args, 'manifest_file', ''))}")
        lines.append(f"  - base_dir={str(getattr(args, 'manifest_base_dir', None))}")
        lines.append(f"- : {mod_map}")
        lsf = getattr(args, "latent_scale_factors_resolved", None)
        if isinstance(lsf, dict) and lsf:
            lines.append(
            )
        else:
    else:

    gamma_str =  if main_snr_gamma is None else str(main_snr_gamma)

    if bool(use_manifest) and bool(getattr(args, "pair_loss_reweight", True)):
        extra = ""
        if (wmin is not None) and (wmax is not None):
        lines.append(
            % (
                float(getattr(args, "pair_loss_reweight_alpha", 0.5)),
                float(getattr(args, "pair_loss_reweight_min", 0.25)),
                float(getattr(args, "pair_loss_reweight_max", 4.0)),
                extra,
            )
        )
    else:
        lines.append("- : ")

    lines.append(
        % (int(dit_total), int(dit_train), int(ema_total), int(ema_train), int(ae_total), int(ae_train))
    )

    lines.append(
        % (
            _yn(getattr(args, "run_eval", True)),
            _yn(getattr(args, "eval_preflight", True)),
            _yn(getattr(args, "eval_distributed", False)),
            int(getattr(args, "eval_batch_size", 0)),
            str(getattr(args, "eval_num_samples", None)),
            int(getattr(args, "eval_vis_samples", 0)),
            str(getattr(args, "eval_sampler", "ddim")),
            int(getattr(args, "eval_num_sampling_steps", 0)),
            float(getattr(args, "eval_cfg_scale", 1.0)),
            _yn(getattr(args, "eval_use_ema", True)),
        )
    )

    lines.append(
        % (
            _yn(getattr(args, "eval_enable_fid", False)),
            _yn(getattr(args, "eval_enable_lpips", False)),
            _yn(getattr(args, "eval_allow_download", True)),
        )
    )

    if ae_lines:
        lines.append("- AE:")
        lines.extend(ae_lines)

    if eval_lines:
        lines.extend(eval_lines)

    lines.append("=" * 80)
    logger.info("\n".join(lines))


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


def _add_unitransfer_to_path(uni_root):
    uni_root = os.path.abspath(uni_root)
    if uni_root not in sys.path:
        sys.path.append(uni_root)
    taming_root = os.path.join(uni_root, "taming-transformers")
    if os.path.isdir(taming_root) and taming_root not in sys.path:
        sys.path.append(taming_root)


def _parse_kv_list(kvs):
    if kvs is None:
        return {}
    out = {}
    for item in kvs:
        if "=" not in item:
            raise ValueError(f"Expected key=value, got: {item}")
        k, v = item.split("=", 1)
        out[str(k).strip().lower()] = str(v).strip()
    return out


def _parse_latent_scale_factors(latent_scale_factors, modalities, default_scale_factor):
    mods = [str(m).strip().lower() for m in (modalities or [])]
    base = {m: float(default_scale_factor) for m in mods}

    kv = _parse_kv_list(latent_scale_factors)
    provided = set()
    for k, v in kv.items():
        kk = "opt" if str(k).strip().lower() == "rgb" else str(k).strip().lower()
        if kk not in base:
            raise ValueError(f"Unknown modality in --latent-scale-factors: {k} (known={mods})")
        if kk in provided:
            raise ValueError(f"Duplicate modality in --latent-scale-factors: {k} (normalized={kk})")
        provided.add(kk)
        base[kk] = float(v)

    if latent_scale_factors is not None:
        if len(provided) == 0:
            raise ValueError(
                "--latent-scale-factors was provided but no valid entries were parsed. "
                "Expected format like: sar=0.42 opt=0.38 ms=0.48 nir=0.56 pan=0.44"
            )
        missing = [m for m in mods if m not in provided]
        if missing:
            raise ValueError(
                f"Strict mode: --latent-scale-factors must cover all --modalities. Missing: {missing}. "
                f"Provided(normalized): {sorted(provided)}. Expected: {mods}"
            )

    return base


def _str2bool(v):
    if isinstance(v, bool):
        return v
    if v is None:
        raise argparse.ArgumentTypeError("Expected a boolean")
    s = str(v).strip().lower()
    if s in ("true", "1", "yes", "y", "t"):
        return True
    if s in ("false", "0", "no", "n", "f"):
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean, got: {v}")


def _resolve_manifest_path(path, base_dir):
    if path is None:
        return path
    p = str(path)
    if os.path.isabs(p) or (len(p) >= 2 and p[1] == ":"):
        return p
    if base_dir is None:
        return p
    b = os.path.abspath(str(base_dir))
    p = p.lstrip("/").lstrip("\\")
    return os.path.normpath(os.path.join(b, p))


def _infer_manifest_dataset_name(path):
    if path is None:
        return None
    s = str(path).replace("\\", "/")
    parts = [p for p in s.split("/") if p]
    try:
        i = parts.index("Images")
    except ValueError:
        return None
    if i + 1 >= len(parts):
        return None
    root = str(parts[i + 1])
    if root.lower() == "spacenet" and i + 2 < len(parts):
        return str(parts[i + 2])
    return root


class BalancedDistributedSampler(Sampler):
    def __init__(
        self,
        dataset,
        group_keys,
        num_replicas=None,
        rank=None,
        shuffle=True,
        seed=0,
        drop_last=False,
        batch_size=None,
        cover_all=False,
    ):
        if num_replicas is None:
            num_replicas = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        if rank is None:
            rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0

        self.dataset = dataset
        self.group_keys = list(group_keys) if group_keys is not None else []
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.batch_size = None if batch_size is None else int(batch_size)
        self.cover_all = bool(cover_all)
        self.epoch = 0
        self._last_full_indices = None

        if len(self.group_keys) != len(self.dataset):
            raise ValueError(f"group_keys length must match dataset length: {len(self.group_keys)} vs {len(self.dataset)}")

        self.group_to_indices = {}
        for i, k in enumerate(self.group_keys):
            kk = "unknown" if k is None else str(k)
            self.group_to_indices.setdefault(kk, []).append(int(i))
        if not self.group_to_indices:
            self.group_to_indices = {"unknown": list(range(len(self.dataset)))}

        if self.cover_all and self.group_to_indices:
            max_len = max(int(len(v)) for v in self.group_to_indices.values())
            desired_total = int(max_len * max(1, len(self.group_to_indices)))
            self.num_samples = int(math.ceil(desired_total / float(self.num_replicas)))
        else:
            if self.drop_last:
                self.num_samples = int(math.floor(len(self.dataset) / self.num_replicas))
            else:
                self.num_samples = int(math.ceil(len(self.dataset) / self.num_replicas))

        if self.batch_size is not None and self.batch_size > 0:
            self.num_samples = int(math.ceil(self.num_samples / float(self.batch_size)) * self.batch_size)

        self.total_size = int(self.num_samples * self.num_replicas)

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        groups = list(self.group_to_indices.keys())
        if self.shuffle and len(groups) > 1:
            p = torch.randperm(len(groups), generator=g).tolist()
            groups = [groups[i] for i in p]

        pools = {}
        ptr = {}
        for k in groups:
            idx = list(self.group_to_indices.get(k, []))
            if not idx:
                continue
            if self.shuffle and len(idx) > 1:
                p = torch.randperm(len(idx), generator=g).tolist()
                idx = [idx[i] for i in p]
            pools[k] = idx
            ptr[k] = 0

        if not pools:
            indices = list(range(len(self.dataset)))
        else:
            indices = []
            group_list = list(pools.keys())
            n_groups = len(group_list)
            for i in range(int(self.total_size)):
                k = group_list[i % n_groups]
                p = int(ptr[k])
                if p < len(pools[k]):
                    indices.append(int(pools[k][p]))
                    ptr[k] = p + 1
                else:
                    base = self.group_to_indices.get(k, [])
                    if not base:
                        indices.append(int(torch.randint(0, len(self.dataset), (1,), generator=g).item()))
                    else:
                        j = int(torch.randint(0, len(base), (1,), generator=g).item())
                        indices.append(int(base[j]))

        if len(indices) < self.total_size:
            indices += indices[: int(self.total_size - len(indices))]
        indices = indices[: int(self.total_size)]
        self._last_full_indices = list(indices)

        start = int(self.rank * self.num_samples)
        end = int(start + self.num_samples)
        return iter(indices[start:end])

    def __len__(self):
        return int(self.num_samples)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def get_last_full_indices(self):
        if self._last_full_indices is None:
            return None
        return list(self._last_full_indices)


def _fid_from_features(fp, fg):
    fp = fp.double()
    fg = fg.double()
    mu1 = fp.mean(dim=0)
    mu2 = fg.mean(dim=0)
    xc1 = fp - mu1
    xc2 = fg - mu2
    cov1 = (xc1.t() @ xc1) / max(1, fp.shape[0] - 1)
    cov2 = (xc2.t() @ xc2) / max(1, fg.shape[0] - 1)
    eps = 1e-6
    cov1 = cov1 + torch.eye(cov1.shape[0], dtype=cov1.dtype) * eps
    cov2 = cov2 + torch.eye(cov2.shape[0], dtype=cov2.dtype) * eps
    e1, v1 = torch.linalg.eigh(cov1)
    e1 = torch.clamp(e1, min=0)
    s1 = (v1 * torch.sqrt(e1).unsqueeze(0)) @ v1.t()
    a = s1 @ cov2 @ s1
    ea = torch.linalg.eigvalsh(a)
    ea = torch.clamp(ea, min=0)
    tr_covmean = torch.sum(torch.sqrt(ea))
    return float(((mu1 - mu2).pow(2).sum() + torch.trace(cov1) + torch.trace(cov2) - 2.0 * tr_covmean).item())


def _default_ae_config_path(uni_root, latent_size, latent_channels, modality_lower):
    name = f"autoencoder_kl_{latent_size}x{latent_size}x{latent_channels}_local_{modality_lower}.yaml"
    return os.path.join(uni_root, "configs", "autoencoder", name)


def _load_unitransfer_ae_and_norm(uni_root, config_path, device, ckpt_override=None):
    from omegaconf import OmegaConf
    from ldm.util import instantiate_from_config

    config_path = os.path.abspath(str(config_path))
    cfg = OmegaConf.load(config_path)
    model_cfg = cfg["model"]
    model_dict = OmegaConf.to_container(model_cfg, resolve=True)
    params = dict(model_dict.get("params", {}))

    ckpt_from_config = params.get("ckpt_path", None)
    ckpt_from_config = None if ckpt_from_config is None else str(ckpt_from_config)

    ckpt_override = None if ckpt_override is None else str(ckpt_override)
    used_override = ckpt_override is not None

    ckpt_chosen = ckpt_override if used_override else ckpt_from_config
    ckpt_resolved = None
    if ckpt_chosen is not None:
        ckpt_resolved = str(ckpt_chosen)
        if not os.path.isabs(ckpt_resolved):
            ckpt_resolved = os.path.join(uni_root, ckpt_resolved)
        ckpt_resolved = os.path.abspath(ckpt_resolved)
        if not os.path.isfile(ckpt_resolved):
            raise FileNotFoundError(
                f"AE checkpoint not found: {ckpt_resolved} (config={config_path}, override={ckpt_override}, config_ckpt={ckpt_from_config})"
            )
        params["ckpt_path"] = ckpt_resolved

    params["lossconfig"] = {"target": "torch.nn.Identity"}

    model = instantiate_from_config({"target": model_dict["target"], "params": params})
    model = model.to(device)
    model.eval()
    requires_grad(model, False)

    data_train_params = {}
    try:
        data_train_params = OmegaConf.to_container(cfg["data"]["params"]["train"]["params"], resolve=True)
    except Exception:
        data_train_params = {}

    norm_cfg = {
        "normalization": data_train_params.get("normalization", "uint8"),
        "data_min": data_train_params.get("data_min", None),
        "data_max": data_train_params.get("data_max", None),
        "normalize_per_channel": data_train_params.get("normalize_per_channel", False),
        "stats_file": data_train_params.get("stats_file", None),
        "raw_size": data_train_params.get("size", None),
    }

    stats_file = norm_cfg.get("stats_file", None)
    if stats_file is not None and not os.path.isabs(str(stats_file)):
        norm_cfg["stats_file"] = os.path.join(uni_root, str(stats_file))

    meta = {
        "config_path": config_path,
        "ckpt_from_config": ckpt_from_config,
        "ckpt_override": ckpt_override,
        "ckpt_resolved": ckpt_resolved,
        "used_override": bool(used_override),
    }

    return model, norm_cfg, meta


@torch.no_grad()
def _encode_raw_list_to_latents(
    raw_list,
    mod_ids_cpu,
    ae_models_by_id,
    latent_size,
    latent_channels,
    latent_scale_factor,
    device,
    strict_no_resize=False,
):
    b = len(raw_list)
    out = torch.zeros(b, latent_channels, latent_size, latent_size, device=device)
    for mod_id, ae in ae_models_by_id.items():
        idx_all = (mod_ids_cpu == int(mod_id)).nonzero(as_tuple=False).flatten().tolist()
        idx = [i for i in idx_all if raw_list[i] is not None]
        if len(idx) == 0:
            continue
        xs = torch.stack([raw_list[i] for i in idx], dim=0).to(device, non_blocking=True)
        posterior = ae.encode(xs)
        z = posterior.sample()
        if z.shape[-1] != latent_size or z.shape[-2] != latent_size:
            if bool(strict_no_resize):
                raise ValueError(
                    f"VAE latent spatial size mismatch for mod_id={int(mod_id)}: got {tuple(z.shape[-2:])}, expected {(int(latent_size), int(latent_size))}. "
                    f"This run forbids any resize; fix the input raw resolution or the AE config downsampling."
                )
            z = F.interpolate(z, size=(latent_size, latent_size), mode="bilinear", align_corners=False)
        sf = latent_scale_factor
        if isinstance(latent_scale_factor, dict):
            sf = latent_scale_factor.get(int(mod_id), 1.0)
        sf = float(sf)
        if sf != 1.0:
            z = z * sf
        out[idx] = z
    return out


class ManifestLatentPairDataset(Dataset):
    def __init__(
        self,
        manifest_file,
        modalities,
        conditioning_mode,
        enable_any_to_all=True,
        allowed_pairs=None,
        augment=True,
        latent_size=None,
        latent_scale_factor=1.0,
        latent_scale_factors=None,
        base_dir=None,
    ):
        self.modalities = [m.lower() for m in modalities]
        self.mod2id = {m: i for i, m in enumerate(self.modalities)}
        self.conditioning_mode = conditioning_mode
        self.enable_any_to_all = bool(enable_any_to_all)
        self.augment = bool(augment)
        self.latent_size = None if latent_size is None else int(latent_size)
        self.latent_scale_factor = float(latent_scale_factor)
        self.latent_scale_factors = None
        if latent_scale_factors is not None:
            self.latent_scale_factors = {str(k).strip().lower(): float(v) for k, v in dict(latent_scale_factors).items()}
        self.base_dir = None if base_dir is None else str(base_dir)

        if allowed_pairs is None:
            self.allowed_pairs_set = None
        else:
            pairs_set = set()
            for p in allowed_pairs:
                if isinstance(p, str) and "->" in p:
                    s, t = p.split("->", 1)
                    pairs_set.add((s.strip().lower(), t.strip().lower()))
                elif isinstance(p, (tuple, list)) and len(p) == 2:
                    pairs_set.add((str(p[0]).strip().lower(), str(p[1]).strip().lower()))
                else:
                    raise ValueError(f"Unsupported allowed_pairs entry: {p}")
            self.allowed_pairs_set = pairs_set

        with open(manifest_file, "r") as f:
            manifest = json.load(f)
        self.samples = list(manifest.values()) if isinstance(manifest, dict) else list(manifest)

        self.pair_group_keys = []
        self.pairs = []
        if self.conditioning_mode == "none":
            for idx, sample in enumerate(self.samples):
                available = [
                    m
                    for m in self.modalities
                    if any(
                        (str(k).lower() == m) or (m == "opt" and str(k).lower() == "rgb")
                        for k in sample.keys()
                    )
                ]
                for tgt_mod in available:
                    self.pairs.append({"sample_idx": idx, "src_mod": None, "tgt_mod": tgt_mod})
                    self.pair_group_keys.append(f"{tgt_mod}->{tgt_mod}")
        else:
            for idx, sample in enumerate(self.samples):
                available = [
                    m
                    for m in self.modalities
                    if any(
                        (str(k).lower() == m) or (m == "opt" and str(k).lower() == "rgb")
                        for k in sample.keys()
                    )
                ]
                if len(available) < 2:
                    continue
                if self.enable_any_to_all:
                    for src_mod in available:
                        for tgt_mod in available:
                            if src_mod == tgt_mod:
                                continue
                            if self.allowed_pairs_set is not None and (src_mod, tgt_mod) not in self.allowed_pairs_set:
                                continue
                            self.pairs.append({"sample_idx": idx, "src_mod": src_mod, "tgt_mod": tgt_mod})
                            self.pair_group_keys.append(f"{src_mod}->{tgt_mod}")
                else:
                    for i, src_mod in enumerate(available):
                        for tgt_mod in available[i + 1 :]:
                            if self.allowed_pairs_set is None or (src_mod, tgt_mod) in self.allowed_pairs_set:
                                self.pairs.append({"sample_idx": idx, "src_mod": src_mod, "tgt_mod": tgt_mod})
                                self.pair_group_keys.append(f"{src_mod}->{tgt_mod}")
                            if self.allowed_pairs_set is None or (tgt_mod, src_mod) in self.allowed_pairs_set:
                                self.pairs.append({"sample_idx": idx, "src_mod": tgt_mod, "tgt_mod": src_mod})
                                self.pair_group_keys.append(f"{tgt_mod}->{src_mod}")

    def __len__(self):
        return len(self.pairs)

    def _get_path(self, sample, modality_lower):
        for k, v in sample.items():
            kl = str(k).lower()
            if kl == modality_lower:
                return _resolve_manifest_path(v, self.base_dir)
            if modality_lower == "opt" and kl == "rgb":
                return _resolve_manifest_path(v, self.base_dir)
        raise KeyError(modality_lower)

    def _load_latent(self, path, modality_lower):
        arr = np.load(path)
        if arr.ndim == 2:
            arr = arr[None, :, :]
        elif arr.ndim == 3:
            if arr.shape[0] <= 16:
                pass
            elif arr.shape[2] <= 16:
                arr = arr.transpose(2, 0, 1)
            else:
                pass
        else:
            raise ValueError(f"Unsupported latent shape: {arr.shape} from {path}")
        x = torch.from_numpy(arr).float()
        if self.latent_size is not None and (x.shape[1] != self.latent_size or x.shape[2] != self.latent_size):
            x = F.interpolate(x.unsqueeze(0), size=(self.latent_size, self.latent_size), mode="bilinear", align_corners=False).squeeze(0)
        sf = float(self.latent_scale_factor)
        if self.latent_scale_factors is not None:
            sf = float(self.latent_scale_factors.get(str(modality_lower).strip().lower(), sf))
        if sf != 1.0:
            x = x * sf
        return x

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        sample = self.samples[pair["sample_idx"]]
        tgt_mod = pair["tgt_mod"]
        tgt_path = self._get_path(sample, tgt_mod)
        x = self._load_latent(tgt_path, tgt_mod)

        if pair["src_mod"] is None:
            cond = torch.zeros_like(x)
            src_mod = tgt_mod
        else:
            src_mod = pair["src_mod"]
            src_path = self._get_path(sample, src_mod)
            cond = self._load_latent(src_path, src_mod)

        if self.augment:
            if torch.rand(()) < 0.5:
                x = torch.flip(x, dims=[2])
                cond = torch.flip(cond, dims=[2])
            if torch.rand(()) < 0.5:
                x = torch.flip(x, dims=[1])
                cond = torch.flip(cond, dims=[1])

        return {
            "x": x,
            "cond": cond,
            "src_mod": torch.tensor(self.mod2id[src_mod], dtype=torch.long),
            "tgt_mod": torch.tensor(self.mod2id[tgt_mod], dtype=torch.long),
        }


class ManifestRawPairDataset(Dataset):
    def __init__(
        self,
        manifest_file,
        modalities,
        conditioning_mode,
        enable_any_to_all=True,
        allowed_pairs=None,
        augment=True,
        size=None,
        normalization_configs=None,
        no_resize=False,
        base_dir=None,
        balance_key=None,
    ):
        self.modalities = [m.lower() for m in modalities]
        self.mod2id = {m: i for i, m in enumerate(self.modalities)}
        self.conditioning_mode = str(conditioning_mode)
        self.enable_any_to_all = bool(enable_any_to_all)
        self.augment = bool(augment)
        self.size = None if size is None else int(size)
        self.base_dir = None if base_dir is None else str(base_dir)
        self.balance_key = None if balance_key is None else str(balance_key)

        if allowed_pairs is None:
            self.allowed_pairs_set = None
        else:
            pairs_set = set()
            for p in allowed_pairs:
                if isinstance(p, str) and "->" in p:
                    s, t = p.split("->", 1)
                    pairs_set.add((s.strip().lower(), t.strip().lower()))
                elif isinstance(p, (tuple, list)) and len(p) == 2:
                    pairs_set.add((str(p[0]).strip().lower(), str(p[1]).strip().lower()))
                else:
                    raise ValueError(f"Unsupported allowed_pairs entry: {p}")
            self.allowed_pairs_set = pairs_set

        with open(manifest_file, "r") as f:
            manifest = json.load(f)
        self.samples = list(manifest.values()) if isinstance(manifest, dict) else list(manifest)

        self.sample_group_keys = []
        bk = "modality_set" if self.balance_key is None else str(self.balance_key).strip().lower()
        for sample in self.samples:
            key = None
            if isinstance(sample, dict) and bk == "modality_set":
                present = set()
                for k in sample.keys():
                    kl = str(k).strip().lower()
                    if kl == "rgb":
                        kl = "opt"
                    if kl in self.modalities:
                        present.add(kl)
                if present:
                    key = "+".join(sorted(present))
            elif isinstance(sample, dict) and bk not in ["infer", ""]:
                for k, v in sample.items():
                    if str(k).strip().lower() == bk:
                        key = v
                        break
            if key is None and isinstance(sample, dict):
                p = None
                for _, v in sample.items():
                    if isinstance(v, str) and v:
                        p = v
                        break
                key = _infer_manifest_dataset_name(p)
            self.sample_group_keys.append("unknown" if key is None else str(key))

        self.pair_group_keys = []

        from ldm.util.normalization import NormalizationManager

        self.no_resize = bool(no_resize)

        if normalization_configs is None:
            normalization_configs = {}
        self.norm_managers = {}
        self.modality_raw_sizes = {}
        for m in self.modalities:
            cfg = normalization_configs.get(m, {})
            self.norm_managers[m] = NormalizationManager(
                normalization=cfg.get("normalization", "uint8"),
                data_min=cfg.get("data_min", None),
                data_max=cfg.get("data_max", None),
                normalize_per_channel=cfg.get("normalize_per_channel", False),
                stats_file=cfg.get("stats_file", None),
            )
            self.modality_raw_sizes[m] = cfg.get("raw_size", None)

        self.pairs = []
        if self.conditioning_mode == "none":
            for idx, sample in enumerate(self.samples):
                available = [
                    m
                    for m in self.modalities
                    if any(
                        (str(k).lower() == m) or (m == "opt" and str(k).lower() == "rgb")
                        for k in sample.keys()
                    )
                ]
                for tgt_mod in available:
                    self.pairs.append({"sample_idx": idx, "src_mod": None, "tgt_mod": tgt_mod})
                    if bk == "pair":
                        self.pair_group_keys.append(f"{tgt_mod}->{tgt_mod}")
                    else:
                        self.pair_group_keys.append(self.sample_group_keys[idx])
        else:
            for idx, sample in enumerate(self.samples):
                available = [
                    m
                    for m in self.modalities
                    if any(
                        (str(k).lower() == m) or (m == "opt" and str(k).lower() == "rgb")
                        for k in sample.keys()
                    )
                ]
                if len(available) < 2:
                    continue
                if self.enable_any_to_all:
                    for src_mod in available:
                        for tgt_mod in available:
                            if src_mod == tgt_mod:
                                continue
                            if self.allowed_pairs_set is not None and (src_mod, tgt_mod) not in self.allowed_pairs_set:
                                continue
                            self.pairs.append({"sample_idx": idx, "src_mod": src_mod, "tgt_mod": tgt_mod})
                            if bk == "pair":
                                self.pair_group_keys.append(f"{src_mod}->{tgt_mod}")
                            else:
                                self.pair_group_keys.append(self.sample_group_keys[idx])
                else:
                    for i, src_mod in enumerate(available):
                        for tgt_mod in available[i + 1 :]:
                            if self.allowed_pairs_set is None or (src_mod, tgt_mod) in self.allowed_pairs_set:
                                self.pairs.append({"sample_idx": idx, "src_mod": src_mod, "tgt_mod": tgt_mod})
                                if bk == "pair":
                                    self.pair_group_keys.append(f"{src_mod}->{tgt_mod}")
                                else:
                                    self.pair_group_keys.append(self.sample_group_keys[idx])
                            if self.allowed_pairs_set is None or (tgt_mod, src_mod) in self.allowed_pairs_set:
                                self.pairs.append({"sample_idx": idx, "src_mod": tgt_mod, "tgt_mod": src_mod})
                                if bk == "pair":
                                    self.pair_group_keys.append(f"{tgt_mod}->{src_mod}")
                                else:
                                    self.pair_group_keys.append(self.sample_group_keys[idx])

    def __len__(self):
        return len(self.pairs)

    def _get_path(self, sample, modality_lower):
        for k, v in sample.items():
            kl = str(k).lower()
            if kl == modality_lower:
                return _resolve_manifest_path(v, self.base_dir)
            if modality_lower == "opt" and kl == "rgb":
                return _resolve_manifest_path(v, self.base_dir)
        raise KeyError(modality_lower)

    def _process_image_shape(self, arr):
        if arr.ndim == 2:
            return arr[None, :, :]
        if arr.ndim == 3:
            if arr.shape[0] <= 16:
                return arr
            if arr.shape[2] <= 16:
                return arr.transpose(2, 0, 1)
            return arr
        raise ValueError(f"Unsupported array shape: {arr.shape}")

    def _load_image(self, path, modality_lower):
        arr = np.load(path)
        arr = self._process_image_shape(arr)
        arr = self.norm_managers[modality_lower].normalize(arr)
        x = torch.from_numpy(arr).float()

        expected = self.size
        mod_expected = self.modality_raw_sizes.get(modality_lower, None)
        if mod_expected is not None:
            try:
                mod_expected = int(mod_expected)
            except Exception:
                mod_expected = None
        if expected is None and mod_expected is not None:
            expected = mod_expected

        if expected is not None and (x.shape[1] != int(expected) or x.shape[2] != int(expected)):
            if self.no_resize:
                raise ValueError(
                    f"Raw size mismatch for modality '{modality_lower}': got {(int(x.shape[1]), int(x.shape[2]))}, expected {(int(expected), int(expected))}. "
                    f"This run forbids any resize; fix the .npy shapes or the AE config 'data.params.train.params.size'."
                )
            x = F.interpolate(x.unsqueeze(0), size=(int(expected), int(expected)), mode="bilinear", align_corners=False).squeeze(0)
        return x

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        sample = self.samples[pair["sample_idx"]]
        tgt_mod = pair["tgt_mod"]
        tgt_path = self._get_path(sample, tgt_mod)
        x_raw = self._load_image(tgt_path, tgt_mod)

        if pair["src_mod"] is None:
            cond_raw = None
            src_mod = tgt_mod
        else:
            src_mod = pair["src_mod"]
            src_path = self._get_path(sample, src_mod)
            cond_raw = self._load_image(src_path, src_mod)

        if self.augment:
            if torch.rand(()) < 0.5:
                x_raw = torch.flip(x_raw, dims=[2])
                if cond_raw is not None:
                    cond_raw = torch.flip(cond_raw, dims=[2])
            if torch.rand(()) < 0.5:
                x_raw = torch.flip(x_raw, dims=[1])
                if cond_raw is not None:
                    cond_raw = torch.flip(cond_raw, dims=[1])

        return {
            "x_raw": x_raw,
            "cond_raw": cond_raw,
            "src_mod": torch.tensor(self.mod2id[src_mod], dtype=torch.long),
            "tgt_mod": torch.tensor(self.mod2id[tgt_mod], dtype=torch.long),
            "src_path": (None if pair["src_mod"] is None else src_path),
            "tgt_path": tgt_path,
        }


def raw_manifest_collate(batch):
    return {
        "x_raw": [b["x_raw"] for b in batch],
        "cond_raw": [b["cond_raw"] for b in batch],
        "src_mod": torch.stack([b["src_mod"] for b in batch], dim=0),
        "tgt_mod": torch.stack([b["tgt_mod"] for b in batch], dim=0),
        "src_path": [b.get("src_path") for b in batch],
        "tgt_path": [b.get("tgt_path") for b in batch],
    }


def _make_denoised_fn(denoised_clamp):
    if denoised_clamp is None:
        return None
    c = float(denoised_clamp)
    if c <= 0:
        return None

    def fn(x):
        return x.clamp(-c, c)

    return fn


def _make_eval_diffusion(num_sampling_steps, sampler, predict_xstart=False):
    n = int(num_sampling_steps)
    sampler = str(sampler).lower()
    if sampler == "ddim":
        timestep_respacing = f"ddim{n}"
    else:
        timestep_respacing = str(n)
    return create_diffusion(timestep_respacing, predict_xstart=bool(predict_xstart))


def _sample_loop(diffusion, sampler, model_fn, shape, noise, clip_denoised, denoised_fn, model_kwargs, device, progress, ddim_eta):
    sampler = str(sampler).lower()
    if sampler == "ddim":
        return diffusion.ddim_sample_loop(
            model_fn,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            eta=float(ddim_eta),
        )
    return diffusion.p_sample_loop(
        model_fn,
        shape,
        noise,
        clip_denoised=clip_denoised,
        denoised_fn=denoised_fn,
        model_kwargs=model_kwargs,
        device=device,
        progress=progress,
    )


@torch.no_grad()
def _decode_latents_by_modality(z, mod_ids_cpu, ae_models_by_id, latent_scale_factor, device):
    z = z.to(device)
    b = z.shape[0]
    outs = [None] * b
    for mod_id, ae in ae_models_by_id.items():
        idx = (mod_ids_cpu == int(mod_id)).nonzero(as_tuple=False).flatten().tolist()
        if len(idx) == 0:
            continue
        zs = z[idx]
        sf = latent_scale_factor
        if isinstance(latent_scale_factor, dict):
            sf = latent_scale_factor.get(int(mod_id), 1.0)
        sf = float(sf)
        if sf != 1.0:
            zs = zs / sf
        dec = ae.decode(zs)
        if hasattr(dec, "sample"):
            dec = dec.sample
        for j, k in enumerate(idx):
            outs[k] = dec[j]
    missing = [i for i, v in enumerate(outs) if v is None]
    if len(missing) > 0:
        raise RuntimeError(f"Missing decoded outputs for indices: {missing}")
    return torch.stack(outs, dim=0)


@torch.no_grad()
def _reconstruct_raw_by_modality(raw_list, mod_ids_cpu, ae_models_by_id, latent_size, latent_scale_factor, device):
    b = len(raw_list)
    outs = [None] * b
    for mod_id, ae in ae_models_by_id.items():
        idx_all = (mod_ids_cpu == int(mod_id)).nonzero(as_tuple=False).flatten().tolist()
        idx = [i for i in idx_all if raw_list[i] is not None]
        if len(idx) == 0:
            continue
        xs = torch.stack([raw_list[i] for i in idx], dim=0).to(device, non_blocking=True)
        posterior = ae.encode(xs)
        if hasattr(posterior, "mode"):
            z = posterior.mode()
        else:
            z = posterior.sample()
        if z.shape[-1] != latent_size or z.shape[-2] != latent_size:
            z = F.interpolate(z, size=(latent_size, latent_size), mode="bilinear", align_corners=False)
        rec = ae.decode(z)
        if hasattr(rec, "sample"):
            rec = rec.sample
        for j, k in enumerate(idx):
            outs[k] = rec[j]
    for i, v in enumerate(outs):
        if v is None:
            outs[i] = torch.zeros_like(raw_list[i] if raw_list[i] is not None else raw_list[0])
    return torch.stack(outs, dim=0)


def _ensure_three_channels(x):
    if x.ndim == 3:
        c, h, w = x.shape
        if c == 3:
            return x
        if c == 1:
            return x.repeat(3, 1, 1)
        if c == 2:
            z = torch.zeros(1, h, w, device=x.device, dtype=x.dtype)
            return torch.cat([x, z], dim=0)
        return x[:3]

    if x.ndim == 4:
        b, c, h, w = x.shape
        if c == 3:
            return x
        if c == 1:
            return x.repeat(1, 3, 1, 1)
        if c == 2:
            z = torch.zeros(b, 1, h, w, device=x.device, dtype=x.dtype)
            return torch.cat([x, z], dim=1)
        return x[:, :3]

    raise ValueError(f"Expected CHW or BCHW, got {list(x.shape)}")


def _ensure_three_channels_gray(x):
    if x.ndim == 3:
        if x.shape[0] == 1:
            g = x
        else:
            g = x.mean(dim=0, keepdim=True)
        return g.repeat(3, 1, 1)

    if x.ndim == 4:
        if x.shape[1] == 1:
            g = x
        else:
            g = x.mean(dim=1, keepdim=True)
        return g.repeat(1, 3, 1, 1)

    raise ValueError(f"Expected CHW or BCHW, got {list(x.shape)}")


def _ensure_three_channels_first(x):
    if x.ndim == 3:
        if x.shape[0] <= 0:
            raise ValueError(f"Expected C>=1, got {list(x.shape)}")
        g = x[:1]
        return g.repeat(3, 1, 1)

    if x.ndim == 4:
        if x.shape[1] <= 0:
            raise ValueError(f"Expected C>=1, got {list(x.shape)}")
        g = x[:, :1]
        return g.repeat(1, 3, 1, 1)

    raise ValueError(f"Expected CHW or BCHW, got {list(x.shape)}")


def _to_01(x):
    return (x.clamp(-1, 1) + 1) * 0.5


def _ssim_torch(x, y, window_size=11):
    x = x.clamp(0, 1)
    y = y.clamp(0, 1)
    pad = int(window_size) // 2
    mu_x = F.avg_pool2d(x, window_size, 1, pad)
    mu_y = F.avg_pool2d(y, window_size, 1, pad)
    sigma_x = F.avg_pool2d(x * x, window_size, 1, pad) - mu_x * mu_x
    sigma_y = F.avg_pool2d(y * y, window_size, 1, pad) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(x * y, window_size, 1, pad) - mu_x * mu_y
    c1 = (0.01 ** 2)
    c2 = (0.03 ** 2)
    num = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    den = (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)
    ssim_map = num / (den + 1e-12)
    return ssim_map.mean(dim=(1, 2, 3))


def _compute_batch_metrics(pred, gt):
    pred01 = _to_01(pred)
    gt01 = _to_01(gt)
    mse = (pred01 - gt01).pow(2).mean(dim=(1, 2, 3))
    rmse = torch.sqrt(mse + 1e-12)
    psnr = -10.0 * torch.log10(mse + 1e-12)

    if SKIMAGE_AVAILABLE:
        ssim_vals = []
        for i in range(pred01.shape[0]):
            a = gt01[i].detach().cpu().numpy().transpose(1, 2, 0)
            b = pred01[i].detach().cpu().numpy().transpose(1, 2, 0)
            try:
                ssim_vals.append(
                    float(
                        structural_similarity(
                            a,
                            b,
                            data_range=1.0,
                            channel_axis=-1 if a.ndim == 3 and a.shape[-1] > 1 else None,
                        )
                    )
                )
            except Exception:
                ssim_vals.append(float("nan"))
        ssim = torch.tensor(ssim_vals, dtype=torch.float32)
    else:
        ssim = _ssim_torch(pred01, gt01).cpu()

    return psnr.detach().cpu(), ssim.detach().cpu(), rmse.detach().cpu()


@torch.no_grad()
def _eval_pred_mse_by_t(model_to_sample, x0, cond, src_mod, tgt_mod, latent_channels, t_list, device, predict_xstart=False):
    full_diffusion = create_diffusion(timestep_respacing="", predict_xstart=bool(predict_xstart))
    out = {}
    c = int(latent_channels)
    for ti in t_list:
        t = torch.full((x0.shape[0],), int(ti), device=device, dtype=torch.long)
        noise = torch.randn_like(x0)
        x_t = full_diffusion.q_sample(x0, t, noise=noise)
        pred = model_to_sample(x_t, t, cond=cond, src_mod=src_mod, tgt_mod=tgt_mod)
        pred_main = pred[:, :c]
        if bool(predict_xstart):
            out[int(ti)] = float(((pred_main - x0) ** 2).mean().item())
        else:
            out[int(ti)] = float(((pred_main - noise) ** 2).mean().item())
    return out


@torch.no_grad()
def evaluate_manifest_raw(
    model_to_sample,
    ae_models_by_id,
    eval_loader,
    args,
    device,
    tb_writer,
    train_steps,
    experiment_dir,
    logger,
    after_adapter=None,
    eval_pair_tag=None,
    eval_out_tag=None,
    eval_split=None,
    tb_prefix="eval",
):
    eval_diffusion = _make_eval_diffusion(args.eval_num_sampling_steps, args.eval_sampler, predict_xstart=bool(getattr(args, "predict_xstart", False)))
    denoised_fn = _make_denoised_fn(args.eval_denoised_clamp)
    model_to_sample.eval()
    if after_adapter is not None:
        after_adapter.eval()

    dist_on = bool(getattr(args, "eval_distributed", False)) and dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
    rank = dist.get_rank() if dist_on else 0
    is_main = (not dist_on) or rank == 0

    psnr_sum = 0.0
    rmse_sum = 0.0
    ssim_sum = 0.0
    ssim_count = 0
    count = 0

    by_dataset = {}

    eval_target = int(getattr(args, "eval_num_samples", -1))
    eval_target = None if eval_target <= 0 else int(eval_target)

    save_tag = eval_out_tag if eval_out_tag is not None else (eval_pair_tag if eval_pair_tag is not None else "eval")
    save_tag = str(save_tag)
    save_tag = save_tag.replace("->", "_to_").replace("/", "_").replace("\\", "_")

    save_all_test = (str(eval_split).lower() == "test") and int(train_steps) > 0 and (int(train_steps) % 30000 == 0)
    if save_all_test:
        eval_target = None

    vis_target = int(getattr(args, "eval_vis_samples", 0))
    if eval_target is not None:
        vis_target = min(int(vis_target), int(eval_target))

    save_root = None
    save_idx = 0
    if save_all_test:
        save_root = os.path.join(experiment_dir, "eval", "samples", save_tag, f"step_{int(train_steps):07d}")
        os.makedirs(os.path.join(save_root, "cond"), exist_ok=True)
        os.makedirs(os.path.join(save_root, "pred"), exist_ok=True)
        os.makedirs(os.path.join(save_root, "gt"), exist_ok=True)

    eval_vis_fixed = bool(getattr(args, "eval_vis_fixed", True))
    vis_heap = []
    vis_images = []

    fid_feats_pred = []
    fid_feats_gt = []
    lpips_sum = 0.0
    lpips_count = 0

    did_request_lpips = False
    did_request_fid = False
    did_detect_opt = False

    pred_mse = None
    pred_mse_name = None

    eval_by_dataset = bool(getattr(args, "eval_by_dataset", True))
    eval_by_dataset_fid = bool(getattr(args, "eval_by_dataset_fid", False))
    eval_by_dataset_lpips = bool(getattr(args, "eval_by_dataset_lpips", True))

    for batch in eval_loader:
        remaining = None if eval_target is None else int(eval_target - count)
        if remaining is not None and remaining <= 0:
            break
        b_full = int(batch["src_mod"].shape[0])
        take = b_full if remaining is None else int(min(b_full, remaining))

        x_raw_list = list(batch["x_raw"])[:take]
        cond_raw_list = list(batch["cond_raw"])[:take]
        src_paths = list(batch.get("src_path") or [])[:take]
        tgt_paths = list(batch.get("tgt_path") or [])[:take]

        src_mod = batch["src_mod"][:take].to(device)
        tgt_mod = batch["tgt_mod"][:take].to(device)
        src_mod_cpu = src_mod.detach().cpu()
        tgt_mod_cpu = tgt_mod.detach().cpu()

        dataset_names = []
        for i in range(take):
            dn = _infer_manifest_dataset_name(tgt_paths[i] if i < len(tgt_paths) else None)
            if dn is None:
                dn = _infer_manifest_dataset_name(src_paths[i] if i < len(src_paths) else None)
            dataset_names.append("unknown" if dn is None else str(dn))

        x = _encode_raw_list_to_latents(
            x_raw_list,
            tgt_mod_cpu,
            ae_models_by_id,
            args.latent_size,
            args.latent_channels,
            args.latent_scale_factors_by_id,
            device,
            strict_no_resize=bool(getattr(args, "no_resize", False)),
        )

        if args.conditioning_mode == "none":
            cond = torch.zeros_like(x)
        else:
            cond = _encode_raw_list_to_latents(
                cond_raw_list,
                src_mod_cpu,
                ae_models_by_id,
                args.latent_size,
                args.latent_channels,
                args.latent_scale_factors_by_id,
                device,
                strict_no_resize=bool(getattr(args, "no_resize", False)),
            )

        eval_pred_mse_flag = getattr(args, "eval_pred_mse", None)
        if eval_pred_mse_flag is None:
            eval_pred_mse_flag = bool(getattr(args, "eval_eps_mse", True))

        if bool(eval_pred_mse_flag) and count == 0:
            n = int(getattr(args, "eval_eps_max_batch", 1))
            n = max(1, min(n, x.shape[0]))
            t_list = list(getattr(args, "eval_eps_timesteps", [999, 900, 800, 600, 400, 200, 100, 50, 10, 1, 0]))
            pred_mse = _eval_pred_mse_by_t(
                model_to_sample=model_to_sample,
                x0=x[:n],
                cond=cond[:n],
                src_mod=src_mod[:n],
                tgt_mod=tgt_mod[:n],
                latent_channels=args.latent_channels,
                t_list=t_list,
                device=device,
                predict_xstart=bool(getattr(args, "predict_xstart", False)),
            )
            pred_mse_name = "x0_mse" if bool(getattr(args, "predict_xstart", False)) else "eps_mse"
            msg = " ".join([f"t{ti}={pred_mse[int(ti)]:.6f}" for ti in t_list])
            tag_parts = []
            if eval_out_tag is not None:
                tag_parts.append(str(eval_out_tag))
            if eval_pair_tag is not None:
                tag_parts.append(str(eval_pair_tag))
            tag = "" if not tag_parts else f"[{'|'.join(tag_parts)}] "
            logger.info(f"(step={train_steps:07d}) {tag}Eval {pred_mse_name}: {msg}")
            if tb_writer is not None:
                for ti in t_list:
                    tb_writer.add_scalar(f"{tb_prefix}/{pred_mse_name}_t{int(ti)}", float(pred_mse[int(ti)]), int(train_steps))

        b_full = int(x.shape[0])
        remaining = None if eval_target is None else int(eval_target - count)
        if remaining is not None and remaining <= 0:
            break
        take = b_full if remaining is None else int(min(b_full, remaining))
        if take <= 0:
            break

        if take != b_full:
            x = x[:take]
            cond = cond[:take]
            src_mod = src_mod[:take]
            tgt_mod = tgt_mod[:take]
            src_mod_cpu = src_mod_cpu[:take]
            tgt_mod_cpu = tgt_mod_cpu[:take]

        x_raw_list = list(x_raw_list)[:take]
        cond_raw_list = list(cond_raw_list)[:take]
        dataset_names = list(dataset_names)[:take]
        if src_paths is not None:
            src_paths = list(src_paths)[:take]
        if tgt_paths is not None:
            tgt_paths = list(tgt_paths)[:take]

        z = torch.randn(take, args.latent_channels, args.latent_size, args.latent_size, device=device)

        cfg_scale = float(args.eval_cfg_scale)
        sampler = str(args.eval_sampler).lower()

        model_fn = model_to_sample.forward
        model_kwargs = dict(cond=cond, src_mod=src_mod, tgt_mod=tgt_mod)
        z_in = z
        shape_in = z.shape

        if cfg_scale != 1.0:
            z_in = torch.cat([z, z], 0)
            cond2 = torch.cat([cond, cond], 0)
            src2 = torch.cat([src_mod, src_mod], 0)
            tgt2 = torch.cat([tgt_mod, tgt_mod], 0)
            model_kwargs = dict(cfg_scale=cfg_scale, cond=cond2, src_mod=src2, tgt_mod=tgt2)
            model_fn = model_to_sample.forward_with_cfg
            shape_in = z_in.shape

        model_fn_to_sample = model_fn

        pred_lat = _sample_loop(
            eval_diffusion,
            sampler,
            model_fn_to_sample,
            shape_in,
            z_in,
            clip_denoised=bool(args.eval_clip_denoised),
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=False,
            ddim_eta=float(args.eval_ddim_eta),
        )

        if cfg_scale != 1.0:
            pred_lat, _ = pred_lat.chunk(2, dim=0)

        if after_adapter is not None and bool(getattr(args, "enable_after_adapter", False)):
            mods = [str(m).strip().lower() for m in (getattr(args, "modalities", []) or [])]
            out_lat = pred_lat.clone()
            for mid in torch.unique(tgt_mod).tolist():
                mi = int(mid)
                if 0 <= mi < len(mods):
                    name = mods[mi]
                    idx = (tgt_mod == mi).nonzero(as_tuple=False).flatten()
                    if int(idx.numel()) > 0:
                        out_lat[idx] = after_adapter.forward_after(out_lat[idx], name)
            pred_lat = out_lat

        pred_raw = _decode_latents_by_modality(pred_lat, tgt_mod_cpu, ae_models_by_id, args.latent_scale_factors_by_id, device)
        gt_raw = torch.stack(x_raw_list, dim=0).to(device, non_blocking=True)

        if save_root is not None:
            mods = [str(m).lower() for m in (getattr(args, "modalities", []) or [])]
            for i in range(int(take)):
                sp = src_paths[i] if i < len(src_paths) else None
                tp = tgt_paths[i] if i < len(tgt_paths) else None
                key = tp if tp is not None else (sp if sp is not None else f"{save_idx:06d}")
                base = os.path.splitext(os.path.basename(str(key)))[0]
                name = f"{save_idx:06d}_{base}.png"

                sid = int(src_mod_cpu[i].item()) if src_mod_cpu.numel() > i else -1
                tid = int(tgt_mod_cpu[i].item()) if tgt_mod_cpu.numel() > i else -1
                sname = mods[sid] if 0 <= sid < len(mods) else None
                tname = mods[tid] if 0 <= tid < len(mods) else None

                c_raw = cond_raw_list[i]
                if torch.is_tensor(c_raw):
                    c_raw = c_raw.to(device, non_blocking=True)
                if sname == "sar":
                    c_vis = _ensure_three_channels_gray(c_raw)
                elif sname == "ms":
                    c_vis = _ensure_three_channels(c_raw)
                else:
                    c_vis = _ensure_three_channels(c_raw)

                g_raw = gt_raw[i]
                if tname == "sar":
                    g_vis = _ensure_three_channels_gray(g_raw)
                elif tname == "ms":
                    g_vis = _ensure_three_channels(g_raw)
                else:
                    g_vis = _ensure_three_channels(g_raw)

                p_raw = pred_raw[i]
                if tname == "sar":
                    p_vis = _ensure_three_channels_gray(p_raw)
                elif tname == "ms":
                    p_vis = _ensure_three_channels(p_raw)
                else:
                    p_vis = _ensure_three_channels(p_raw)

                save_image(_to_01(c_vis.detach().float().cpu()), os.path.join(save_root, "cond", name))
                save_image(_to_01(p_vis.detach().float().cpu()), os.path.join(save_root, "pred", name))
                save_image(_to_01(g_vis.detach().float().cpu()), os.path.join(save_root, "gt", name))
                save_idx += 1

        psnr_b, ssim_b, rmse_b = _compute_batch_metrics(pred_raw, gt_raw)

        rmse_scale = 1.0
        try:
            base_ds = getattr(eval_loader, "dataset", None)
            if isinstance(base_ds, torch.utils.data.Subset):
                base_ds = getattr(base_ds, "dataset", base_ds)
            mods = [str(m).lower() for m in (getattr(args, "modalities", []) or [])]
            mid = int(tgt_mod_cpu[0].item()) if tgt_mod_cpu.numel() > 0 else 0
            mname = mods[mid] if 0 <= mid < len(mods) else None
            nm = None if (mname is None or base_ds is None) else getattr(base_ds, "norm_managers", {}).get(mname, None)
            if nm is not None:
                if str(getattr(nm, "normalization", "")).lower() == "uint8":
                    rmse_scale = 255.0
                elif str(getattr(nm, "normalization", "")).lower() == "zero_one":
                    rmse_scale = 1.0
                else:
                    dmin = getattr(nm, "data_min", None)
                    dmax = getattr(nm, "data_max", None)
                    if dmin is not None and dmax is not None:
                        if isinstance(dmin, (list, tuple)) or isinstance(dmax, (list, tuple)):
                            rmse_scale = float(max(dmax) - min(dmin))
                        else:
                            rmse_scale = float(dmax) - float(dmin)
        except Exception:
            rmse_scale = 1.0

        valid = torch.isfinite(ssim_b)
        psnr_sum += float(psnr_b.sum().item())
        rmse_sum += float((rmse_b * float(rmse_scale)).sum().item())
        if int(valid.sum().item()) > 0:
            ssim_sum += float(ssim_b[valid].sum().item())
            ssim_count += int(valid.sum().item())
        count += int(take)

        if eval_by_dataset:
            for dn in sorted(set(dataset_names)):
                st = by_dataset.get(dn)
                if st is None:
                    st = {"psnr_sum": 0.0, "rmse_sum": 0.0, "ssim_sum": 0.0, "ssim_count": 0, "count": 0, "lpips_sum": 0.0, "lpips_count": 0, "fid_pred": [], "fid_gt": []}
                    by_dataset[dn] = st
                idx = [ii for ii, nn in enumerate(dataset_names) if nn == dn]
                if not idx:
                    continue
                it = torch.tensor(idx, device=psnr_b.device, dtype=torch.long)
                st["psnr_sum"] += float(psnr_b[it].sum().item())
                st["rmse_sum"] += float((rmse_b[it] * float(rmse_scale)).sum().item())
                vmask = valid[it]
                if int(vmask.sum().item()) > 0:
                    st["ssim_sum"] += float(ssim_b[it][vmask].sum().item())
                    st["ssim_count"] += int(vmask.sum().item())
                st["count"] += int(len(idx))

        if int(vis_target) > 0:
            mods = [str(m).lower() for m in (getattr(args, "modalities", []) or [])]
            sar_id = None
            ms_id = None
            for j, m in enumerate(mods):
                if m == "sar":
                    sar_id = int(j)
                elif m == "ms":
                    ms_id = int(j)

            if eval_vis_fixed:
                for i in range(int(take)):
                    kt = "" if i >= len(tgt_paths) or tgt_paths[i] is None else str(tgt_paths[i])
                    ks = "" if i >= len(src_paths) or src_paths[i] is None else str(src_paths[i])
                    h = int(hashlib.sha1(f"{kt}|{ks}".encode("utf-8")).hexdigest()[:16], 16)
                    if len(vis_heap) < int(vis_target) or h < int(vis_heap[0][1]):
                        tid = int(tgt_mod_cpu[i].item())
                        if sar_id is not None and tid == int(sar_id):
                            gt_vis = _ensure_three_channels_gray(gt_raw[i])
                            pred_vis = _ensure_three_channels_gray(pred_raw[i])
                        elif ms_id is not None and tid == int(ms_id):
                            gt_vis = _ensure_three_channels(gt_raw[i])
                            pred_vis = _ensure_three_channels(pred_raw[i])
                        else:
                            gt_vis = _ensure_three_channels(gt_raw[i])
                            pred_vis = _ensure_three_channels(pred_raw[i])
                        item = (-h, h, _to_01(gt_vis.detach().cpu()), _to_01(pred_vis.detach().cpu()))
                        if len(vis_heap) < int(vis_target):
                            heapq.heappush(vis_heap, item)
                        else:
                            heapq.heapreplace(vis_heap, item)
            else:
                if len(vis_images) < 2 * int(vis_target):
                    for i in range(min(int(take), int(vis_target) - (len(vis_images) // 2))):
                        tid = int(tgt_mod_cpu[i].item())
                        if sar_id is not None and tid == int(sar_id):
                            gt_vis = _ensure_three_channels_gray(gt_raw[i])
                            pred_vis = _ensure_three_channels_gray(pred_raw[i])
                        elif ms_id is not None and tid == int(ms_id):
                            gt_vis = _ensure_three_channels(gt_raw[i])
                            pred_vis = _ensure_three_channels(pred_raw[i])
                        else:
                            gt_vis = _ensure_three_channels(gt_raw[i])
                            pred_vis = _ensure_three_channels(pred_raw[i])
                        vis_images.append(_to_01(gt_vis.detach().cpu()))
                        vis_images.append(_to_01(pred_vis.detach().cpu()))

        mid = int(tgt_mod_cpu[0].item()) if tgt_mod_cpu.numel() > 0 else -1
        mods = [str(m).lower() for m in (getattr(args, "modalities", []) or [])]
        tgt_name = mods[mid] if 0 <= mid < len(mods) else None

        tgt_is_opt = False
        if tgt_name is not None and ("opt" in str(tgt_name).lower()):
            tgt_is_opt = True
        if (not tgt_is_opt) and (eval_pair_tag is not None) and ("->opt" in str(eval_pair_tag).lower() or "to_opt" in str(eval_pair_tag).lower()):
            tgt_is_opt = True
        if (not tgt_is_opt) and (eval_out_tag is not None) and ("opt" in str(eval_out_tag).lower()):
            tgt_is_opt = True

        if tgt_is_opt:
            did_detect_opt = True
            enable_lpips = bool(getattr(args, "eval_enable_lpips", True))
            enable_fid = bool(getattr(args, "eval_enable_fid", True))
            allow_download = bool(getattr(args, "eval_allow_download", True))
            did_request_lpips = bool(did_request_lpips or enable_lpips)
            did_request_fid = bool(did_request_fid or enable_fid)

            pr = _ensure_three_channels(pred_raw)
            gt = _ensure_three_channels(gt_raw)

            if enable_lpips:
                try:
                    if pr.shape[-1] != 256 or pr.shape[-2] != 256:
                        pr_256 = F.interpolate(pr, size=(256, 256), mode="bilinear", align_corners=False)
                    else:
                        pr_256 = pr
                    if gt.shape[-1] != 256 or gt.shape[-2] != 256:
                        gt_256 = F.interpolate(gt, size=(256, 256), mode="bilinear", align_corners=False)
                    else:
                        gt_256 = gt

                    from taming.modules.losses.lpips import LPIPS
                    if not hasattr(evaluate_manifest_raw, "_lpips"):
                        evaluate_manifest_raw._lpips = LPIPS(
                            ckpt_path=getattr(args, "lpips_ckpt_path", None),
                            vgg_weights_path=getattr(args, "lpips_vgg_weights", None),
                            allow_download=allow_download,
                        ).to(device).eval()
                    lp = evaluate_manifest_raw._lpips

                    v = lp(pr_256, gt_256).view(pr_256.shape[0], -1).mean(dim=1)
                    good = torch.isfinite(v)
                    if int(good.sum().item()) > 0:
                        lpips_sum += float(v[good].sum().item())
                        lpips_count += int(good.sum().item())

                    if eval_by_dataset and eval_by_dataset_lpips:
                        for dn in sorted(set(dataset_names)):
                            st = by_dataset.get(dn)
                            if st is None:
                                continue
                            idx = [ii for ii, nn in enumerate(dataset_names) if nn == dn]
                            if not idx:
                                continue
                            it = torch.tensor(idx, device=v.device, dtype=torch.long)
                            gg = good[it]
                            if int(gg.sum().item()) > 0:
                                st["lpips_sum"] += float(v[it][gg].sum().item())
                                st["lpips_count"] += int(gg.sum().item())
                except Exception as e:
                    if not bool(allow_download):
                        raise RuntimeError(f"LPIPS init/run failed with eval_allow_download=False: {e}")
                    logger.warning(f"LPIPS init/run failed, skipped. Error: {e}")

            if enable_fid:
                try:
                    import torchvision
                    if not hasattr(evaluate_manifest_raw, "_inception"):
                        if not allow_download and getattr(args, "fid_inception_weights", None) is None:
                            raise RuntimeError("FID is enabled but eval_allow_download=False and --fid-inception-weights is not provided")

                        if getattr(args, "fid_inception_weights", None) is not None:
                            try:
                                inc = torchvision.models.inception_v3(weights=None, aux_logits=False, init_weights=False)
                            except TypeError:
                                inc = torchvision.models.inception_v3(weights=None, aux_logits=False)
                            try:
                                state = torch.load(str(getattr(args, "fid_inception_weights")), map_location="cpu", weights_only=True)
                            except TypeError:
                                state = torch.load(str(getattr(args, "fid_inception_weights")), map_location="cpu")
                            if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
                                state = state["state_dict"]
                            inc.load_state_dict(state, strict=False)
                            evaluate_manifest_raw._inc_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
                            evaluate_manifest_raw._inc_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
                        else:
                            try:
                                w = torchvision.models.Inception_V3_Weights.IMAGENET1K_V1
                                try:
                                    inc = torchvision.models.inception_v3(weights=w, aux_logits=False, init_weights=False)
                                except TypeError:
                                    inc = torchvision.models.inception_v3(weights=w, aux_logits=False)
                                mean = getattr(getattr(w, "meta", None) or {}, "get", lambda _k, _d=None: _d)("mean", None)
                                std = getattr(getattr(w, "meta", None) or {}, "get", lambda _k, _d=None: _d)("std", None)
                                evaluate_manifest_raw._inc_mean = torch.tensor(mean if mean is not None else [0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
                                evaluate_manifest_raw._inc_std = torch.tensor(std if std is not None else [0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
                            except Exception:
                                import warnings
                                with warnings.catch_warnings():
                                    warnings.filterwarnings("ignore", message="The parameter 'pretrained' is deprecated*", category=UserWarning)
                                    warnings.filterwarnings("ignore", message="Arguments other than a weight enum*", category=UserWarning)
                                    try:
                                        inc = torchvision.models.inception_v3(pretrained=True, aux_logits=False, init_weights=False)
                                    except TypeError:
                                        inc = torchvision.models.inception_v3(pretrained=True, aux_logits=False)
                                evaluate_manifest_raw._inc_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
                                evaluate_manifest_raw._inc_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

                        inc.fc = torch.nn.Identity()
                        inc.to(device).eval()
                        evaluate_manifest_raw._inception = inc

                    inc = evaluate_manifest_raw._inception
                    mean = evaluate_manifest_raw._inc_mean
                    std = evaluate_manifest_raw._inc_std

                    pr01 = _to_01(pr)
                    gt01 = _to_01(gt)
                    pr299 = F.interpolate(pr01, size=(299, 299), mode="bilinear", align_corners=False)
                    gt299 = F.interpolate(gt01, size=(299, 299), mode="bilinear", align_corners=False)
                    prn = (pr299 - mean) / std
                    gtn = (gt299 - mean) / std
                    fp = inc(prn).detach().cpu()
                    fg = inc(gtn).detach().cpu()
                    fid_feats_pred.append(fp)
                    fid_feats_gt.append(fg)

                    if eval_by_dataset and eval_by_dataset_fid:
                        for dn in sorted(set(dataset_names)):
                            st = by_dataset.get(dn)
                            if st is None:
                                continue
                            idx = [ii for ii, nn in enumerate(dataset_names) if nn == dn]
                            if not idx:
                                continue
                            st["fid_pred"].append(fp[idx])
                            st["fid_gt"].append(fg[idx])
                except Exception as e:
                    if not bool(allow_download):
                        raise RuntimeError(f"FID init/run failed with eval_allow_download=False: {e}")
                    logger.warning(f"FID init/run failed, skipped. Error: {e}")


    if dist_on:
        red = torch.tensor(
            [
                float(psnr_sum),
                float(rmse_sum),
                float(ssim_sum),
                float(ssim_count),
                float(count),
                float(lpips_sum),
                float(lpips_count),
            ],
            device=device,
            dtype=torch.float64,
        )
        dist.all_reduce(red, op=dist.ReduceOp.SUM)
        psnr_sum = float(red[0].item())
        rmse_sum = float(red[1].item())
        ssim_sum = float(red[2].item())
        ssim_count = int(red[3].item())
        count = int(red[4].item())
        lpips_sum = float(red[5].item())
        lpips_count = int(red[6].item())

        if eval_by_dataset:
            gathered = [None for _ in range(int(dist.get_world_size()))]
            dist.all_gather_object(gathered, by_dataset)
            if is_main:
                merged = {}
                for d in gathered:
                    if not isinstance(d, dict):
                        continue
                    for dn, st in d.items():
                        if not isinstance(st, dict):
                            continue
                        m = merged.get(dn)
                        if m is None:
                            m = {"psnr_sum": 0.0, "rmse_sum": 0.0, "ssim_sum": 0.0, "ssim_count": 0, "count": 0, "lpips_sum": 0.0, "lpips_count": 0, "fid_pred": [], "fid_gt": []}
                            merged[dn] = m
                        m["psnr_sum"] += float(st.get("psnr_sum", 0.0))
                        m["rmse_sum"] += float(st.get("rmse_sum", 0.0))
                        m["ssim_sum"] += float(st.get("ssim_sum", 0.0))
                        m["ssim_count"] += int(st.get("ssim_count", 0))
                        m["count"] += int(st.get("count", 0))
                        m["lpips_sum"] += float(st.get("lpips_sum", 0.0))
                        m["lpips_count"] += int(st.get("lpips_count", 0))
                        if eval_by_dataset_fid:
                            m["fid_pred"].extend(list(st.get("fid_pred", []) or []))
                            m["fid_gt"].extend(list(st.get("fid_gt", []) or []))
                by_dataset = merged
            else:
                by_dataset = {}

    if count == 0:
        if is_main:
            logger.info("Eval skipped: no samples")
        return None

    psnr = psnr_sum / count
    rmse = rmse_sum / count
    ssim = (ssim_sum / ssim_count) if ssim_count > 0 else float("nan")

    tag_parts = []
    if eval_out_tag is not None:
        tag_parts.append(str(eval_out_tag))
    if eval_pair_tag is not None:
        tag_parts.append(str(eval_pair_tag))
    tag = "" if not tag_parts else f"[{'|'.join(tag_parts)}] "
    if is_main:
        logger.info(f"(step={train_steps:07d}) {tag}Eval PSNR: {psnr:.4f}, SSIM: {ssim:.4f}, RMSE: {rmse:.6f} (n={count})")

    if tb_writer is not None:
        tb_writer.add_scalar(f"{tb_prefix}/psnr", float(psnr), int(train_steps))
        tb_writer.add_scalar(f"{tb_prefix}/ssim", float(ssim), int(train_steps))
        tb_writer.add_scalar(f"{tb_prefix}/rmse", float(rmse), int(train_steps))

    fid = None
    if len(fid_feats_pred) > 0 and len(fid_feats_gt) > 0:
        try:
            fp_local = torch.cat(fid_feats_pred, dim=0)
            fg_local = torch.cat(fid_feats_gt, dim=0)

            if dist_on:
                gathered_fp = [None for _ in range(int(dist.get_world_size()))]
                gathered_fg = [None for _ in range(int(dist.get_world_size()))]
                dist.all_gather_object(gathered_fp, fp_local)
                dist.all_gather_object(gathered_fg, fg_local)
                if is_main:
                    fp = torch.cat([t for t in gathered_fp if isinstance(t, torch.Tensor) and t.numel() > 0], dim=0)
                    fg = torch.cat([t for t in gathered_fg if isinstance(t, torch.Tensor) and t.numel() > 0], dim=0)
                    fid = _fid_from_features(fp, fg)
                else:
                    fid = None
            else:
                fid = _fid_from_features(fp_local, fg_local)
        except Exception as e:
            fid = None
            if is_main:
                logger.warning(f"(step={train_steps:07d}) {tag}FID compute failed, skipped. Error: {e}")

    lpips = None
    if lpips_count > 0:
        lpips = float(lpips_sum / float(lpips_count))

    if is_main and did_detect_opt and (did_request_lpips or did_request_fid):
        extra = []
        if lpips is not None:
            extra.append(f"LPIPS: {lpips:.6f}")
        if fid is not None:
            extra.append(f"FID: {fid:.6f}")
        if extra:
            logger.info(f"(step={train_steps:07d}) {tag}Eval " + ", ".join(extra) + f" (n={count})")
        else:
            logger.warning(
                f"(step={train_steps:07d}) {tag}LPIPS/FID requested but not computed (lpips_count={int(lpips_count)}, fid_batches={int(len(fid_feats_pred))})"
            )

    if tb_writer is not None:
        if lpips is not None:
            tb_writer.add_scalar(f"{tb_prefix}/lpips", float(lpips), int(train_steps))
        if fid is not None:
            tb_writer.add_scalar(f"{tb_prefix}/fid", float(fid), int(train_steps))

    if is_main and len(vis_heap) > 0:
        vis_heap_sorted = sorted(list(vis_heap), key=lambda x: int(x[1]))
        vis_images = []
        for _, __, gt_vis, pr_vis in vis_heap_sorted:
            vis_images.append(gt_vis)
            vis_images.append(pr_vis)

        grid = make_grid(torch.stack(vis_images, dim=0), nrow=20, normalize=False)
        out_dir = os.path.join(experiment_dir, "eval", "grids", save_tag)
        os.makedirs(out_dir, exist_ok=True)
        grid_path = os.path.join(out_dir, f"grid_gt_pred_{train_steps:07d}.png")
        save_image(grid, grid_path)
        if tb_writer is not None:
            tb_writer.add_image(f"{tb_prefix}/grid_gt_pred", grid, int(train_steps))

    out = {
        "psnr": psnr,
        "ssim": ssim,
        "rmse": rmse,
        "count": int(count),
        "fid": fid,
        "lpips": lpips,
        "eval_sampler": str(args.eval_sampler).lower(),
        "eval_num_sampling_steps": int(args.eval_num_sampling_steps),
        "eval_ddim_eta": float(args.eval_ddim_eta),
        "eval_cfg_scale": float(args.eval_cfg_scale),
        "pred_mse_name": pred_mse_name,
        "pred_mse": pred_mse,
    }

    csv_rows = []
    base_row = dict(out)
    base_row["dataset"] = "ALL"
    csv_rows.append(base_row)

    if eval_by_dataset:
        for dn, st in sorted(by_dataset.items(), key=lambda kv: kv[0]):
            c = int(st.get("count", 0))
            if c <= 0:
                continue
            row = dict(out)
            row["dataset"] = str(dn)
            row["count"] = int(c)
            row["psnr"] = float(st.get("psnr_sum", 0.0)) / float(c)
            row["rmse"] = float(st.get("rmse_sum", 0.0)) / float(c)
            sc = int(st.get("ssim_count", 0))
            row["ssim"] = (float(st.get("ssim_sum", 0.0)) / float(sc)) if sc > 0 else float("nan")

            if eval_by_dataset_lpips and int(st.get("lpips_count", 0)) > 0:
                row["lpips"] = float(st.get("lpips_sum", 0.0)) / float(int(st.get("lpips_count")))
            else:
                row["lpips"] = None

            if eval_by_dataset_fid and len(st.get("fid_pred", [])) > 0 and len(st.get("fid_gt", [])) > 0:
                try:
                    fp = torch.cat(st.get("fid_pred"), dim=0).double()
                    fg = torch.cat(st.get("fid_gt"), dim=0).double()
                    mu1 = fp.mean(dim=0)
                    mu2 = fg.mean(dim=0)
                    xc1 = fp - mu1
                    xc2 = fg - mu2
                    cov1 = (xc1.t() @ xc1) / max(1, fp.shape[0] - 1)
                    cov2 = (xc2.t() @ xc2) / max(1, fg.shape[0] - 1)
                    eps = 1e-6
                    cov1 = cov1 + torch.eye(cov1.shape[0], dtype=cov1.dtype) * eps
                    cov2 = cov2 + torch.eye(cov2.shape[0], dtype=cov2.dtype) * eps
                    e1, v1 = torch.linalg.eigh(cov1)
                    e1 = torch.clamp(e1, min=0)
                    s1 = (v1 * torch.sqrt(e1).unsqueeze(0)) @ v1.t()
                    a = s1 @ cov2 @ s1
                    ea = torch.linalg.eigvalsh(a)
                    ea = torch.clamp(ea, min=0)
                    tr_covmean = torch.sum(torch.sqrt(ea))
                    row["fid"] = float(((mu1 - mu2).pow(2).sum() + torch.trace(cov1) + torch.trace(cov2) - 2.0 * tr_covmean).item())
                except Exception:
                    row["fid"] = None
            else:
                row["fid"] = None

            csv_rows.append(row)

    out["csv_rows"] = csv_rows
    return out


def evaluate_manifest_val_loss(
    model_for_loss,
    diffusion,
    ae_models_by_id,
    eval_loader,
    args,
    device,
    train_steps,
    epoch,
    logger,
    eval_pair_tag=None,
    eval_out_tag=None,
    pair_weight_table=None,
):
    model_for_loss.eval()
    loss_sum = 0.0
    count = 0

    for batch in eval_loader:
        b = int(batch["src_mod"].shape[0])
        if b <= 0:
            continue

        src_mod = batch["src_mod"].to(device)
        tgt_mod = batch["tgt_mod"].to(device)

        with torch.no_grad():
            tgt_mod_cpu = tgt_mod.detach().cpu()
            x = _encode_raw_list_to_latents(
                batch["x_raw"],
                tgt_mod_cpu,
                ae_models_by_id,
                args.latent_size,
                args.latent_channels,
                args.latent_scale_factors_by_id,
                device,
                strict_no_resize=bool(getattr(args, "no_resize", False)),
            )

            if args.conditioning_mode == "none":
                cond = torch.zeros_like(x)
            else:
                src_mod_cpu = src_mod.detach().cpu()
                cond = _encode_raw_list_to_latents(
                    batch["cond_raw"],
                    src_mod_cpu,
                    ae_models_by_id,
                    args.latent_size,
                    args.latent_channels,
                    args.latent_scale_factors_by_id,
                    device,
                    strict_no_resize=bool(getattr(args, "no_resize", False)),
                )

        g = torch.Generator(device=device)
        g.manual_seed(int(getattr(args, "global_seed", 0)) + int(train_steps))
        t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device, generator=g)
        noise = torch.randn_like(x, generator=g)

        model_kwargs = dict(cond=cond, src_mod=src_mod, tgt_mod=tgt_mod)
        x_t = diffusion.q_sample(x, t, noise=noise)
        model_out = model_for_loss(x_t, t, **model_kwargs)

        def _cached_model(_x, _t, **_kw):
            return model_out

        loss_dict = diffusion.training_losses(_cached_model, x, t, model_kwargs=model_kwargs, noise=noise)
        per_sample = loss_dict["loss"]
        if pair_weight_table is not None:
            per_sample = per_sample * pair_weight_table[src_mod, tgt_mod]
        loss_sum += float(per_sample.sum().detach().item())
        count += int(per_sample.numel())

    if count <= 0:
        return None

    mean_loss = float(loss_sum) / float(count)

    tag_parts = []
    if eval_out_tag is not None:
        tag_parts.append(str(eval_out_tag))
    if eval_pair_tag is not None:
        tag_parts.append(str(eval_pair_tag))
    tag = "" if not tag_parts else f"[{'|'.join(tag_parts)}] "
    logger.info(f"(step={int(train_steps):07d}) {tag}ValLoss: {mean_loss:.6f} (n={int(count)})")

    return {
        "train_steps": int(train_steps),
        "epoch": int(epoch),
        "split": "val",
        "eval_pair": "" if eval_pair_tag is None else str(eval_pair_tag),
        "out_tag": "" if eval_out_tag is None else str(eval_out_tag),
        "loss": float(mean_loss),
        "count": int(count),
    }


def _append_val_loss_csv_row(experiment_dir, row):
    out_dir = os.path.join(experiment_dir, "eval")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "val_loss.csv")

    fieldnames = [
        "train_steps",
        "epoch",
        "split",
        "eval_pair",
        "out_tag",
        "loss",
        "count",
    ]

    is_new = not os.path.isfile(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if is_new:
            w.writeheader()
        w.writerow({k: row.get(k, None) for k in fieldnames})


@torch.no_grad()
def evaluate_manifest_val_loss(
    *,
    model_for_loss,
    diffusion,
    ae_models_by_id,
    eval_loader,
    args,
    device,
    train_steps,
    epoch,
    logger,
    eval_pair_tag=None,
    eval_out_tag=None,
    pair_weight_table=None,
    split="val",
):
    model_for_loss.eval()

    total = 0.0
    count = 0

    for batch in eval_loader:
        src_mod = batch["src_mod"].to(device)
        tgt_mod = batch["tgt_mod"].to(device)

        with torch.no_grad():
            tgt_mod_cpu = tgt_mod.detach().cpu()
            x = _encode_raw_list_to_latents(
                batch["x_raw"],
                tgt_mod_cpu,
                ae_models_by_id,
                args.latent_size,
                args.latent_channels,
                args.latent_scale_factors_by_id,
                device,
                strict_no_resize=bool(getattr(args, "no_resize", False)),
            )

            if args.conditioning_mode == "none":
                cond = torch.zeros_like(x)
            else:
                src_mod_cpu = src_mod.detach().cpu()
                cond = _encode_raw_list_to_latents(
                    batch["cond_raw"],
                    src_mod_cpu,
                    ae_models_by_id,
                    args.latent_size,
                    args.latent_channels,
                    args.latent_scale_factors_by_id,
                    device,
                    strict_no_resize=bool(getattr(args, "no_resize", False)),
                )

        t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
        noise = torch.randn_like(x)
        x_t = diffusion.q_sample(x, t, noise=noise)

        model_kwargs = dict(cond=cond, src_mod=src_mod, tgt_mod=tgt_mod)
        model_out = model_for_loss(x_t, t, **model_kwargs)

        def _cached_model(_x, _t, **_kw):
            return model_out

        loss_dict = diffusion.training_losses(_cached_model, x, t, model_kwargs=model_kwargs, noise=noise)
        per_sample = loss_dict["loss"]
        if pair_weight_table is not None:
            per_sample = per_sample * pair_weight_table[src_mod, tgt_mod]

        loss = per_sample.mean()
        bs = int(x.shape[0])
        total += float(loss.item()) * bs
        count += bs

    if count <= 0:
        return None

    avg = float(total) / float(count)

    tag_parts = []
    if eval_out_tag is not None:
        tag_parts.append(str(eval_out_tag))
    if eval_pair_tag is not None:
        tag_parts.append(str(eval_pair_tag))
    tag = "" if not tag_parts else f"[{'|'.join(tag_parts)}] "
    logger.info(f"(step={int(train_steps):07d}) {tag}ValLoss: {avg:.6f} (n={int(count)})")

    return {
        "train_steps": int(train_steps),
        "epoch": int(epoch),
        "split": str(split),
        "eval_pair": eval_pair_tag,
        "loss": avg,
        "count": int(count),
    }


def _append_val_loss_csv_row(experiment_dir, row):
    out_dir = os.path.join(experiment_dir, "val_loss")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "metrics.csv")

    fieldnames = [
        "train_steps",
        "epoch",
        "split",
        "eval_pair",
        "loss",
        "count",
    ]

    exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in fieldnames})


def _append_eval_csv_row(experiment_dir, row):
    out_dir = os.path.join(experiment_dir, "eval")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "metrics.csv")

    fieldnames = [
        "train_steps",
        "epoch",
        "split",
        "eval_pair",
        "dataset",
        "psnr",
        "ssim",
        "rmse",
        "fid",
        "lpips",
        "count",
        "eval_sampler",
        "eval_num_sampling_steps",
        "eval_ddim_eta",
        "eval_cfg_scale",
        "pred_mse_name",
        "pred_mse",
    ]

    if os.path.isfile(csv_path):
        try:
            with open(csv_path, "r", newline="") as f:
                header = next(csv.reader(f), None)
            if header is not None and list(header) != list(fieldnames):
                with open(csv_path, "r", newline="") as f:
                    old_rows = list(csv.DictReader(f))
                with open(csv_path, "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=fieldnames)
                    w.writeheader()
                    for rr in old_rows:
                        w.writerow({k: rr.get(k, "") for k in fieldnames})
        except Exception:
            pass

    exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in fieldnames})


def _monitor_eval(eval_history, args, logger, tb_writer, train_steps):
    if not bool(getattr(args, "monitor_enable", True)):
        return
    if not eval_history:
        return

    cur = eval_history[-1]
    psnr = float(cur.get("psnr", float("nan")))
    ssim = float(cur.get("ssim", float("nan")))

    prev = eval_history[-2] if len(eval_history) >= 2 else None
    prev_psnr = float(prev.get("psnr", float("nan"))) if prev is not None else float("nan")
    prev_ssim = float(prev.get("ssim", float("nan"))) if prev is not None else float("nan")

    alerts = []
    severity = "info"

    def _finite(x: float) -> bool:
        return bool(np.isfinite(x))

    regress_psnr = float(getattr(args, "monitor_regress_psnr", 0.2))
    regress_ssim = float(getattr(args, "monitor_regress_ssim", 0.01))
    if prev is not None and _finite(psnr) and _finite(prev_psnr) and _finite(ssim) and _finite(prev_ssim):
        if (psnr < prev_psnr - regress_psnr) and (ssim < prev_ssim - regress_ssim):
            severity = "warning"
            alerts.append(
                f"REGRESSION: PSNR {psnr:.4f} (< {prev_psnr:.4f}-{regress_psnr}), SSIM {ssim:.4f} (< {prev_ssim:.4f}-{regress_ssim})"
            )

    window = int(getattr(args, "monitor_plateau_window", 3))
    delta_psnr = float(getattr(args, "monitor_plateau_psnr_delta", 0.05))
    delta_ssim = float(getattr(args, "monitor_plateau_ssim_delta", 0.003))
    if len(eval_history) >= window:
        recent = eval_history[-window:]
        rp = [float(m.get("psnr", float("nan"))) for m in recent]
        rs = [float(m.get("ssim", float("nan"))) for m in recent]
        if all(_finite(x) for x in rp) and all(_finite(x) for x in rs):
            if (max(rp) - min(rp) < delta_psnr) and (max(rs) - min(rs) < delta_ssim):
                severity = "warning" if severity != "warning" else severity
                alerts.append(f"PLATEAU: last{window} eval PSNR range < {delta_psnr}, SSIM range < {delta_ssim}")

    if bool(getattr(args, "predict_xstart", False)):
        warn_t999 = float(getattr(args, "monitor_x0_t999_warn", 0.8))
        min_improve = float(getattr(args, "monitor_x0_t999_min_improve", 0.03))
        series = []
        for m in eval_history[-max(window, 3):]:
            pm = m.get("pred_mse", None)
            if isinstance(pm, dict) and (999 in pm):
                v = float(pm.get(999))
                if _finite(v):
                    series.append(v)
        if len(series) >= 3:
            if series[-1] >= warn_t999 and (series[0] - series[-1] < min_improve):
                severity = "warning" if severity != "warning" else severity
                alerts.append(f"HIGH-T STALL: x0_mse@t999={series[-1]:.4f} (last{len(series)} improve<{min_improve})")

    if alerts:
        header = f"(step={train_steps:07d}) [MONITOR]"
        msg = header + "\n" + "\n".join(alerts)
        if severity == "warning":
            logger.warning("=" * 80)
            logger.warning(msg)
            logger.warning("=" * 80)
        else:
            logger.info(msg)
        if tb_writer is not None:
            tb_writer.add_text("monitor/alerts", msg.replace("\n", "<br/>") , int(train_steps))


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Setup an experiment folder:
    resume_exists = args.resume is not None and os.path.isfile(args.resume)
    tb_writer = None
    experiment_dir = None
    checkpoint_dir = None
    if rank == 0:
        if resume_exists:
            checkpoint_dir = os.path.dirname(os.path.abspath(args.resume))
            experiment_dir = os.path.dirname(checkpoint_dir)
            logger = create_logger(experiment_dir)
            logger.info(f"Resuming experiment directory at {experiment_dir}")
        else:
            os.makedirs(args.results_dir, exist_ok=True)
            experiment_index = len(glob(f"{args.results_dir}/*"))
            model_string_name = args.model.replace("/", "-")
            experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"
            checkpoint_dir = f"{experiment_dir}/checkpoints"
            os.makedirs(checkpoint_dir, exist_ok=True)
            logger = create_logger(experiment_dir)
            logger.info(f"Experiment directory created at {experiment_dir}")

        tb_dir = args.tensorboard_dir
        if tb_dir is None or str(tb_dir).strip() == "":
            tb_dir = os.path.join(experiment_dir, "tensorboard")
        tb_writer = SummaryWriter(log_dir=str(tb_dir))
    else:
        logger = create_logger(None)

    if dist.is_available() and dist.is_initialized():
        exp_obj = [experiment_dir]
        ckpt_obj = [checkpoint_dir]
        dist.broadcast_object_list(exp_obj, src=0)
        dist.broadcast_object_list(ckpt_obj, src=0)
        experiment_dir = exp_obj[0]
        checkpoint_dir = ckpt_obj[0]

    if experiment_dir is None:
        raise RuntimeError("experiment_dir is None; rank0 should have broadcast it")

    use_manifest = args.manifest_file is not None

    mods = [str(m).strip().lower() for m in (getattr(args, "modalities", []) or [])]
    if use_manifest:
        resolved = _parse_latent_scale_factors(getattr(args, "latent_scale_factors", None), mods, float(args.latent_scale_factor))
        args.latent_scale_factors_resolved = resolved
        args.latent_scale_factors_by_id = {int(i): float(resolved[m]) for i, m in enumerate(mods)}
    else:
        args.latent_scale_factors_resolved = None
        args.latent_scale_factors_by_id = float(args.latent_scale_factor)

    if use_manifest:
        latent_size = args.latent_size
        model = DiT_models[args.model](
            input_size=latent_size,
            in_channels=args.latent_channels,
            num_classes=0,
            conditioning_mode=args.conditioning_mode,
            x_channels=args.latent_channels,
            cond_channels=args.latent_channels,
            cond_dropout_prob=args.cond_dropout_prob,
            use_modality_embedding=True,
            num_modalities=len(args.modalities),
            modality_dropout_prob=args.modality_dropout_prob,
        )
    else:
        assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
        latent_size = args.image_size // 8
        model = DiT_models[args.model](
            input_size=latent_size,
            num_classes=args.num_classes,
        )

    ema = deepcopy(model).to(device)
    requires_grad(ema, False)
    model = DDP(model.to(device), device_ids=[device])
    x0_loss_snr_gamma = getattr(args, "x0_loss_snr_gamma", None)
    if x0_loss_snr_gamma is not None and float(x0_loss_snr_gamma) <= 0:
        x0_loss_snr_gamma = None

    diffusion = create_diffusion(
        timestep_respacing="",
        predict_xstart=bool(getattr(args, "predict_xstart", False)),
        x0_loss_snr_gamma=x0_loss_snr_gamma,
    )
    vae = None if use_manifest else AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0)

    after_adapter = None
    after_opt = None
    pending_after_opt_state = None
    if use_manifest and bool(getattr(args, "enable_after_adapter", False)):
        mods = [str(m).strip().lower() for m in (getattr(args, "modalities", []) or [])]
        adapter_cfgs = {m: {"after": {"type": str(getattr(args, "after_adapter_type", "refine"))}} for m in mods}
        after_adapter = MultiModalAdapterManager(
            modalities=mods,
            latent_channels=int(args.latent_channels),
            enable_adapters=True,
            adapter_type=str(getattr(args, "after_adapter_type", "refine")),
            adapter_configs=adapter_cfgs,
            build_before=False,
            build_after=True,
        ).to(device)
        after_adapter = DDP(after_adapter, device_ids=[device], find_unused_parameters=True)
        after_lr = getattr(args, "after_adapter_lr", None)
        after_lr = float(args.lr) if after_lr is None else float(after_lr)
        after_opt = torch.optim.AdamW(after_adapter.parameters(), lr=after_lr, weight_decay=0)

    resumed = False
    train_steps = 0
    resume_epoch = None
    pending_opt_state = None


    if args.resume is not None:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume, map_location="cpu")
            model.module.load_state_dict(checkpoint["model"])
            ema.load_state_dict(checkpoint["ema"])

            if after_adapter is not None and isinstance(checkpoint.get("after_adapter", None), dict):
                after_adapter.module.load_state_dict(checkpoint["after_adapter"], strict=True)

            eval_preflight_before_opt = bool(getattr(args, "eval_preflight_before_opt", False))
            if not bool(getattr(args, "resume_no_opt", False)):
                pending_opt_state = checkpoint.get("opt", None)
                if pending_opt_state is not None and (not eval_preflight_before_opt):
                    opt.load_state_dict(pending_opt_state)
                    for state in opt.state.values():
                        for k, v in state.items():
                            if torch.is_tensor(v):
                                state[k] = v.to(device)
                    pending_opt_state = None

                    if bool(getattr(args, "resume_reset_lr", True)):
                        for pg in opt.param_groups:
                            pg["lr"] = float(args.lr)

                if after_opt is not None:
                    pending_after_opt_state = checkpoint.get("after_opt", None)
                    if pending_after_opt_state is not None and (not eval_preflight_before_opt):
                        after_opt.load_state_dict(pending_after_opt_state)
                        for state in after_opt.state.values():
                            for k, v in state.items():
                                if torch.is_tensor(v):
                                    state[k] = v.to(device)
                        pending_after_opt_state = None

                        if bool(getattr(args, "resume_reset_lr", True)):
                            for pg in after_opt.param_groups:
                                pg["lr"] = float(getattr(args, "after_adapter_lr", args.lr))

            try:
                train_steps = int(os.path.basename(args.resume).split(".")[0])
            except Exception:
                train_steps = 0
            if "epoch" in checkpoint:
                try:
                    resume_epoch = int(checkpoint["epoch"])
                except Exception:
                    resume_epoch = None
            resumed = True

            lr_now = float(opt.param_groups[0].get("lr", float(args.lr))) if len(opt.param_groups) > 0 else float(args.lr)
            delayed_opt = pending_opt_state is not None
            logger.info(
                f"Resumed from checkpoint: {args.resume} at step {train_steps} (resume_no_opt={bool(getattr(args, 'resume_no_opt', False))}, lr={lr_now:g}, delayed_opt={delayed_opt})"
            )
        else:
            logger.info(f"Checkpoint not found: {args.resume}")

    resume_after_adapter = getattr(args, "resume_after_adapter", None)
    if after_adapter is not None and resume_after_adapter is not None and os.path.isfile(str(resume_after_adapter)):
        ckpt = torch.load(str(resume_after_adapter), map_location="cpu")
        state = ckpt.get("after_adapter", None) if isinstance(ckpt, dict) else None
        if isinstance(state, dict):
            after_adapter.module.load_state_dict(state, strict=True)
        elif isinstance(ckpt, dict):
            after_adapter.module.load_state_dict(ckpt, strict=True)
        else:
            raise RuntimeError(f"Invalid --resume-after-adapter checkpoint: {resume_after_adapter}")
        logger.info(f"Loaded after-adapter from: {resume_after_adapter}")


    collate_fn = None
    ae_models_by_id = None
    ae_info_by_modality = None
    normalization_configs = None

    if use_manifest:
        if args.manifest_data_type == "raw":
            _add_unitransfer_to_path(args.uni_transfer_root)

            ae_config_map = _parse_kv_list(args.ae_configs)
            ae_ckpt_map = _parse_kv_list(args.ae_ckpts)
            norm_config_overrides = {}
            if args.normalization_config_json is not None:
                with open(args.normalization_config_json, "r") as f:
                    norm_config_overrides = json.load(f)
                norm_config_overrides = {str(k).lower(): v for k, v in norm_config_overrides.items()}

            mod2id = {m.lower(): i for i, m in enumerate(args.modalities)}
            ae_models_by_id = {}
            normalization_configs = {}
            ae_info_by_modality = {}

            for m in [mm.lower() for mm in args.modalities]:
                cfg_path = ae_config_map.get(m, None)
                if cfg_path is None:
                    cfg_path = _default_ae_config_path(args.uni_transfer_root, args.latent_size, args.latent_channels, m)
                if not os.path.isfile(cfg_path):
                    raise FileNotFoundError(f"AE config not found for modality '{m}': {cfg_path}")

                ckpt_override = ae_ckpt_map.get(m, None)
                ae, norm_cfg, ae_meta = _load_unitransfer_ae_and_norm(
                    args.uni_transfer_root, cfg_path, device, ckpt_override=ckpt_override
                )
                ae_models_by_id[mod2id[m]] = ae
                ae_info_by_modality[m] = ae_meta

                if args.require_ae_ckpts and not ae_meta.get("used_override", False):
                    raise ValueError(
                        f"--require-ae-ckpts is set but modality '{m}' has no checkpoint override. "
                        f"Provide --ae-ckpts {m}=/abs/path/to/ckpt.ckpt"
                    )

                if rank == 0:
                    logger.info(
                        f"AE[{m}] config={ae_meta.get('config_path')} ckpt={ae_meta.get('ckpt_resolved')} used_override={ae_meta.get('used_override')}"
                    )
                    if not ae_meta.get("used_override", False):
                        ckpt_resolved = str(ae_meta.get("ckpt_resolved") or "")
                        if "extractvae" in ckpt_resolved.lower() or "pretrained" in ckpt_resolved.lower() or "init" in ckpt_resolved.lower():
                            logger.warning(f"AE[{m}] appears to be using an init/pretrained checkpoint (no override): {ckpt_resolved}")

                if m in norm_config_overrides:
                    merged = dict(norm_cfg)
                    merged.update(norm_config_overrides[m] or {})
                    norm_cfg = merged

                normalization_configs[m] = norm_cfg

            dataset = ManifestRawPairDataset(
                manifest_file=args.manifest_file,
                modalities=args.modalities,
                conditioning_mode=args.conditioning_mode,
                enable_any_to_all=args.enable_any_to_all,
                allowed_pairs=args.allowed_pairs,
                augment=args.augment,
                size=(None if bool(getattr(args, "no_resize", False)) else args.raw_size),
                normalization_configs=normalization_configs,
                no_resize=bool(getattr(args, "no_resize", False)),
                base_dir=getattr(args, "manifest_base_dir", None),
                balance_key=getattr(args, "balance_key", None),
            )
            collate_fn = raw_manifest_collate
        else:
            dataset = ManifestLatentPairDataset(
                manifest_file=args.manifest_file,
                modalities=args.modalities,
                conditioning_mode=args.conditioning_mode,
                enable_any_to_all=args.enable_any_to_all,
                allowed_pairs=args.allowed_pairs,
                augment=args.augment,
                latent_size=args.latent_size,
                latent_scale_factor=args.latent_scale_factor,
                latent_scale_factors=getattr(args, "latent_scale_factors_resolved", None),
                base_dir=getattr(args, "manifest_base_dir", None),
            )
    else:
        transform = transforms.Compose([
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])
        dataset = ImageFolder(args.data_path, transform=transform)

    pair_weight_table = None
    if use_manifest and bool(getattr(args, "pair_loss_reweight", True)) and hasattr(dataset, "pairs"):
        try:
            num_mod = len(args.modalities)
            counts = np.zeros((num_mod, num_mod), dtype=np.int64)
            for p in getattr(dataset, "pairs"):
                s = p.get("src_mod", None)
                t_ = p.get("tgt_mod", None)
                if t_ is None:
                    continue
                if s is None:
                    s = t_
                si = int(dataset.mod2id[str(s).lower()])
                ti = int(dataset.mod2id[str(t_).lower()])
                if 0 <= si < num_mod and 0 <= ti < num_mod:
                    counts[si, ti] += 1

            pos = counts[counts > 0]
            mean_count = float(pos.mean()) if pos.size > 0 else 1.0
            alpha = float(getattr(args, "pair_loss_reweight_alpha", 0.5))
            w = np.ones_like(counts, dtype=np.float32)
            if mean_count > 0 and alpha > 0 and pos.size > 0:
                for i in range(num_mod):
                    for j in range(num_mod):
                        c = int(counts[i, j])
                        if c > 0:
                            w[i, j] = float((mean_count / float(c)) ** alpha)

            wmin = float(getattr(args, "pair_loss_reweight_min", 0.25))
            wmax = float(getattr(args, "pair_loss_reweight_max", 4.0))
            w = np.clip(w, wmin, wmax)
            pair_weight_table = torch.tensor(w, device=device, dtype=torch.float32)

            if rank == 0:
                logger.info(
                    f"Pair reweight enabled: mean_count={mean_count:.1f}, alpha={alpha:.3f}, clip=[{wmin:.3f},{wmax:.3f}], weight_range=[{float(w.min()):.3f},{float(w.max()):.3f}]"
                )
        except Exception as e:
            if rank == 0:
                logger.warning(f"Pair reweight init failed, disabled. Error: {e}")
            pair_weight_table = None

    train_sampler_name = str(getattr(args, "train_sampler", "distributed")).strip().lower()
    if train_sampler_name == "balanced" and bool(use_manifest) and hasattr(dataset, "pair_group_keys"):
        sampler = BalancedDistributedSampler(
            dataset,
            group_keys=getattr(dataset, "pair_group_keys"),
            num_replicas=dist.get_world_size(),
            rank=rank,
            shuffle=True,
            seed=int(getattr(args, "global_seed", 0)),
            drop_last=True,
            batch_size=int(args.global_batch_size // dist.get_world_size()),
            cover_all=bool(getattr(args, "balanced_cover_all", False)),
        )
    else:
        sampler = DistributedSampler(
            dataset,
            num_replicas=dist.get_world_size(),
            rank=rank,
            shuffle=True,
            seed=args.global_seed,
        )

    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )

    if rank == 0:
        if use_manifest:
            logger.info(f"Dataset contains {len(dataset):,} pairs ({args.manifest_file})")
            if hasattr(dataset, "pairs"):
                direction_counts = Counter()
                for p in getattr(dataset, "pairs", []) or []:
                    t_ = p.get("tgt_mod", None)
                    if t_ is None:
                        continue
                    s = p.get("src_mod", None)
                    if s is None:
                        s = t_
                    direction_counts[f"{str(s).lower()}->{str(t_).lower()}"] += 1
                if direction_counts:
                    logger.info("Training direction counts (filtered by --allowed-pairs):")
                    for k in sorted(direction_counts.keys()):
                        logger.info(f"  {k}: {int(direction_counts[k])}")
        else:
            logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")

    eval_loaders = {}
    eval_history_by_pair = {}
    val_loss_loaders = {}
    eval_distributed = bool(getattr(args, "eval_distributed", False))
    if (rank == 0 or eval_distributed) and use_manifest and args.manifest_data_type == "raw" and bool(args.run_eval):
        if args.allowed_pairs is not None and len(args.allowed_pairs) > 0:
            eval_pairs = [str(p).strip().lower() for p in args.allowed_pairs]
        else:
            eval_pairs = [f"{str(args.eval_src_modality).strip().lower()}->{str(args.eval_tgt_modality).strip().lower()}"]

        eval_sources = []
        enable_val = bool(getattr(args, "eval_enable_val", True))
        enable_test = bool(getattr(args, "eval_enable_test", True))
        enable_train_fixed = getattr(args, "eval_enable_train_fixed", None)
        if enable_train_fixed is None:
            enable_train_fixed = bool(getattr(args, "eval_train_fixed_enable", True))
        enable_train_fixed = bool(enable_train_fixed)

        val_manifest_path = getattr(args, "val_manifest_file", None) or args.eval_manifest_file or args.manifest_file
        if enable_val and val_manifest_path is not None:
            val_base_dir = getattr(args, "eval_manifest_base_dir", None) or getattr(args, "manifest_base_dir", None)
            eval_sources.append(("val", val_manifest_path, val_base_dir))

        if rank == 0 and val_manifest_path is not None:
            val_base_dir = getattr(args, "eval_manifest_base_dir", None) or getattr(args, "manifest_base_dir", None)
            for p in eval_pairs:
                if "->" not in p:
                    raise ValueError(f"Invalid --allowed-pairs entry for eval: {p}")
                s, t = p.split("->", 1)
                s = s.strip().lower()
                t = t.strip().lower()
                pair_tag = f"{s}->{t}"
                out_tag = f"val_loss_{s}_to_{t}"
                eval_allowed_pairs = [pair_tag]

                ds = ManifestRawPairDataset(
                    manifest_file=val_manifest_path,
                    modalities=args.modalities,
                    conditioning_mode=args.conditioning_mode,
                    enable_any_to_all=True,
                    allowed_pairs=eval_allowed_pairs,
                    augment=False,
                    size=(None if bool(getattr(args, "no_resize", False)) else args.raw_size),
                    normalization_configs=normalization_configs or {},
                    no_resize=bool(getattr(args, "no_resize", False)),
                    base_dir=val_base_dir,
                    balance_key=getattr(args, "balance_key", None),
                )
                dl = DataLoader(
                    ds,
                    batch_size=int(args.eval_batch_size),
                    shuffle=False,
                    sampler=None,
                    num_workers=args.num_workers,
                    pin_memory=True,
                    drop_last=False,
                    collate_fn=raw_manifest_collate,
                )

                key = f"val_loss:{pair_tag}"
                val_loss_loaders[key] = {"loader": dl, "out_tag": out_tag, "pair_tag": pair_tag, "split": "val"}
                logger.info(f"Val-loss dataset contains {len(ds):,} pairs ({val_manifest_path}), pairs={eval_allowed_pairs}")

        test_manifest_path = getattr(args, "test_manifest_file", None)
        if enable_test and test_manifest_path is not None:
            test_base_dir = getattr(args, "manifest_base_dir", None)
            eval_sources.append(("test", test_manifest_path, test_base_dir))

        if enable_train_fixed:
            train_fixed_n = int(getattr(args, "eval_train_fixed_samples", 100))
            if train_fixed_n > 0:
                train_base_dir = getattr(args, "manifest_base_dir", None)
                eval_sources.append(("train_fixed", args.manifest_file, train_base_dir))

        for split, manifest_path, base_dir in eval_sources:
            for p in eval_pairs:
                if "->" not in p:
                    raise ValueError(f"Invalid --allowed-pairs entry for eval: {p}")
                s, t = p.split("->", 1)
                s = s.strip().lower()
                t = t.strip().lower()
                pair_tag = f"{s}->{t}"
                out_tag = f"{split}_{s}_to_{t}"
                eval_allowed_pairs = [pair_tag]

                eval_dataset = ManifestRawPairDataset(
                    manifest_file=manifest_path,
                    modalities=args.modalities,
                    conditioning_mode=args.conditioning_mode,
                    enable_any_to_all=True,
                    allowed_pairs=eval_allowed_pairs,
                    augment=False,
                    size=(None if bool(getattr(args, "no_resize", False)) else args.raw_size),
                    normalization_configs=normalization_configs or {},
                    no_resize=bool(getattr(args, "no_resize", False)),
                    base_dir=base_dir,
                    balance_key=getattr(args, "balance_key", None),
                )
                if split == "train_fixed":
                    train_fixed_n = int(getattr(args, "eval_train_fixed_samples", 100))
                    if train_fixed_n > 0:
                        idx = list(range(min(int(train_fixed_n), int(len(eval_dataset)))))
                        eval_dataset = torch.utils.data.Subset(eval_dataset, idx)
                eval_sampler = None
                if eval_distributed and dist.is_available() and dist.is_initialized():
                    eval_sampler = DistributedSampler(
                        eval_dataset,
                        num_replicas=dist.get_world_size(),
                        rank=rank,
                        shuffle=False,
                        seed=int(getattr(args, "global_seed", 0)),
                    )
                eval_loader = DataLoader(
                    eval_dataset,
                    batch_size=int(args.eval_batch_size),
                    shuffle=False,
                    sampler=eval_sampler,
                    num_workers=args.num_workers,
                    pin_memory=True,
                    drop_last=False,
                    collate_fn=raw_manifest_collate,
                )

                history_key = f"{split}:{pair_tag}"
                eval_loaders[history_key] = {
                    "loader": eval_loader,
                    "out_tag": out_tag,
                    "pair_tag": pair_tag,
                    "split": split,
                }
                logger.info(
                    f"Eval dataset contains {len(eval_dataset):,} pairs ({manifest_path}), split={split}, pairs={eval_allowed_pairs}"
                )

    if rank == 0:
        _log_config_summary(
            args=args,
            logger=logger,
            rank=rank,
            device=device,
            world_size=dist.get_world_size(),
            use_manifest=use_manifest,
            resumed=resumed,
            train_steps=train_steps,
            dataset=dataset,
            loader=loader,
            ae_models_by_id=ae_models_by_id,
            ae_info_by_modality=ae_info_by_modality,
            pair_weight_table=pair_weight_table,
            model=model,
            ema=ema,
            eval_loaders=eval_loaders,
        )

    # Prepare models for training:
    if not resumed:
        update_ema(ema, model.module, decay=0)
    model.train()
    ema.eval()

    eval_preflight_always = bool(getattr(args, "eval_preflight_always", False))
    eval_distributed = bool(getattr(args, "eval_distributed", False))
    if (rank == 0 or eval_distributed) and use_manifest and args.manifest_data_type == "raw" and bool(args.run_eval) and bool(getattr(args, "eval_preflight", True)) and (eval_preflight_always or int(train_steps) == 0):
        if not eval_loaders:
            if rank == 0:
                logger.info("(step=0000000) Eval preflight skipped: eval_loaders is empty")
        else:
            model_to_sample = ema if bool(args.eval_use_ema) else model.module
            do_compare = (
                (after_adapter is not None)
                and bool(getattr(args, "enable_after_adapter", False))
                and bool(getattr(args, "eval_noadapt_baseline", False))
            )
            is_main_pf = (not eval_distributed) or rank == 0
            for history_key, item in eval_loaders.items():
                runs = [("adapt", after_adapter.module), ("noadapt", None)] if bool(do_compare) else [("adapt", after_adapter.module if after_adapter is not None else None)]
                for run_tag, aa in runs:
                    base_pair_tag = item.get("pair_tag")
                    base_out_tag = item.get("out_tag")
                    pair_tag = None if base_pair_tag is None else f"{base_pair_tag}|{run_tag}"
                    out_tag = None if base_out_tag is None else f"{base_out_tag}_{run_tag}"
                    hist_key_pf = f"{history_key}|{run_tag}" if bool(do_compare) else history_key

                    metrics = evaluate_manifest_raw(
                        model_to_sample=model_to_sample,
                        ae_models_by_id=ae_models_by_id,
                        eval_loader=item["loader"],
                        args=args,
                        device=device,
                        tb_writer=tb_writer,
                        train_steps=int(train_steps),
                        experiment_dir=experiment_dir,
                        logger=logger,
                        after_adapter=aa,
                        eval_pair_tag=pair_tag,
                        eval_out_tag=out_tag,
                        eval_split=item.get("split"),
                        tb_prefix=f"eval/{out_tag}" if out_tag else "eval",
                    )
                    if is_main_pf and metrics is not None and hasattr(sys.modules[__name__], "_append_eval_csv_row"):
                        hist_pf = eval_history_by_pair.setdefault(hist_key_pf, [])
                        hist_pf.append(metrics)
                        csv_rows = None
                        if isinstance(metrics, dict):
                            csv_rows = metrics.get("csv_rows", None)
                        if not csv_rows:
                            csv_rows = [metrics]

                        for rr0 in csv_rows:
                            rr = dict(rr0) if isinstance(rr0, dict) else {}
                            rr["eval_pair"] = pair_tag
                            rr["split"] = item.get("split")
                            rr["dataset"] = rr.get("dataset", "ALL")
                            rr["train_steps"] = int(train_steps)
                            rr["epoch"] = int(0)
                            rr["pred_mse"] = "" if rr.get("pred_mse") is None else json.dumps(rr.get("pred_mse"), sort_keys=True)
                            _append_eval_csv_row(experiment_dir, rr)

                        _monitor_eval(hist_pf, args, logger, tb_writer, int(train_steps))
            model.train()
            ema.eval()

    if resumed and pending_opt_state is not None:
        opt.load_state_dict(pending_opt_state)
        for state in opt.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)
        pending_opt_state = None

        if bool(getattr(args, "resume_reset_lr", True)):
            for pg in opt.param_groups:
                pg["lr"] = float(args.lr)

    if resumed and after_opt is not None and pending_after_opt_state is not None:
        after_opt.load_state_dict(pending_after_opt_state)
        for state in after_opt.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)
        pending_after_opt_state = None

        if bool(getattr(args, "resume_reset_lr", True)):
            for pg in after_opt.param_groups:
                pg["lr"] = float(getattr(args, "after_adapter_lr", args.lr))

    dist.barrier()

    if int(train_steps) == 0 and bool(getattr(args, "save_initial_checkpoint", True)):
        if rank == 0:
            checkpoint = {
                "model": model.module.state_dict(),
                "ema": ema.state_dict(),
                "opt": opt.state_dict(),
                "args": args,
                "ae_info_by_modality": ae_info_by_modality,
                "epoch": int(0),
            }
            if after_adapter is not None:
                checkpoint["after_adapter"] = after_adapter.module.state_dict()
                if after_opt is not None:
                    checkpoint["after_opt"] = after_opt.state_dict()
            init_path = f"{checkpoint_dir}/0000000.pt"
            torch.save(checkpoint, init_path)
            logger.info(f"Saved initial checkpoint to {init_path}")
        dist.barrier()

    # Variables for monitoring/logging purposes:
    log_steps = 0
    running_loss = 0.0
    start_time = time()

    steps_per_epoch = len(loader)
    start_epoch = 0
    if resumed:
        if resume_epoch is not None:
            start_epoch = resume_epoch
        elif steps_per_epoch > 0:
            start_epoch = int(train_steps // steps_per_epoch)
        if rank == 0:
            logger.info(f"Resuming epoch counter at epoch={start_epoch} (steps_per_epoch={steps_per_epoch}, train_steps={train_steps})")

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for batch in loader:
            if use_manifest:
                src_mod = batch["src_mod"].to(device)
                tgt_mod = batch["tgt_mod"].to(device)

                if args.manifest_data_type == "raw":
                    with torch.no_grad():
                        tgt_mod_cpu = tgt_mod.detach().cpu()
                        x = _encode_raw_list_to_latents(
                            batch["x_raw"],
                            tgt_mod_cpu,
                            ae_models_by_id,
                            args.latent_size,
                            args.latent_channels,
                            args.latent_scale_factors_by_id,
                            device,
                            strict_no_resize=bool(getattr(args, "no_resize", False)),
                        )

                        if args.conditioning_mode == "none":
                            cond = torch.zeros_like(x)
                        else:
                            src_mod_cpu = src_mod.detach().cpu()
                            cond = _encode_raw_list_to_latents(
                                batch["cond_raw"],
                                src_mod_cpu,
                                ae_models_by_id,
                                args.latent_size,
                                args.latent_channels,
                                args.latent_scale_factors_by_id,
                                device,
                                strict_no_resize=bool(getattr(args, "no_resize", False)),
                            )
                else:
                    x = batch["x"].to(device)
                    cond = batch["cond"].to(device)
            else:
                x, y = batch
                x = x.to(device)
                y = y.to(device)
                with torch.no_grad():
                    x = vae.encode(x).latent_dist.sample().mul_(0.18215)

            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)

            if use_manifest:
                model_kwargs = dict(cond=cond, src_mod=src_mod, tgt_mod=tgt_mod)
            else:
                model_kwargs = dict(y=y)

            noise = torch.randn(x.shape, device=device, dtype=x.dtype)
            x_t = diffusion.q_sample(x, t, noise=noise)
            model_out = model(x_t, t, **model_kwargs)

            def _cached_model(_x, _t, **_kw):
                return model_out

            loss_dict = diffusion.training_losses(_cached_model, x, t, model_kwargs=model_kwargs, noise=noise)
            per_sample = loss_dict["loss"]
            if pair_weight_table is not None and use_manifest:
                per_sample = per_sample * pair_weight_table[src_mod, tgt_mod]
            main_loss = per_sample.mean()
            loss = main_loss

            latent_loss = None
            latent_keep_ratio = None
            latent_loss_scale = float(getattr(args, "latent_loss_scale", 0.0) or 0.0)
            if after_adapter is not None and latent_loss_scale > 0:
                max_t = getattr(args, "after_adapter_max_t", None)
                if max_t is not None:
                    max_t = int(max_t)
                    if max_t < 0:
                        max_t = None

                mask = None
                do_latent = True
                if max_t is not None:
                    mask = (t <= int(max_t))
                    if int(mask.long().sum().item()) <= 0:
                        do_latent = False
                        mask = None

                if do_latent:
                    c = int(args.latent_channels)
                    pred_main = model_out[:, :c]
                    if bool(getattr(args, "predict_xstart", False)):
                        pred_x0 = pred_main
                    else:
                        pred_x0 = diffusion._predict_xstart_from_eps(x_t=x_t, t=t, eps=pred_main)

                    z = pred_x0.detach()

                    z_in = z
                    x_in = x
                    src_in = src_mod
                    tgt_in = tgt_mod
                    if mask is not None:
                        z_in = z[mask]
                        x_in = x[mask]
                        src_in = src_mod[mask]
                        tgt_in = tgt_mod[mask]

                    try:
                        latent_keep_ratio = float(z_in.shape[0]) / float(max(1, z.shape[0]))
                    except Exception:
                        latent_keep_ratio = None

                    z_after = z_in.clone()
                    mods = [str(m).strip().lower() for m in (getattr(args, "modalities", []) or [])]
                    for mid in torch.unique(tgt_in).tolist():
                        mi = int(mid)
                        if 0 <= mi < len(mods):
                            name = mods[mi]
                            idx = (tgt_in == mi).nonzero(as_tuple=False).flatten()
                            if int(idx.numel()) > 0:
                                z_after[idx] = after_adapter.module.forward_after(z_in[idx], name)

                    per_lat = (z_after - x_in).float().pow(2).flatten(1).mean(dim=1)
                    if pair_weight_table is not None and use_manifest:
                        per_lat = per_lat * pair_weight_table[src_in, tgt_in]
                    latent_loss = per_lat.mean()
                    loss = loss + latent_loss_scale * latent_loss

            opt.zero_grad()
            if after_opt is not None:
                after_opt.zero_grad()
            loss.backward()
            opt.step()
            if after_opt is not None:
                after_opt.step()
            update_ema(ema, model.module)

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1

            if (
                rank == 0
                and bool(getattr(args, "log_unused_samples", False))
                and use_manifest
                and isinstance(sampler, BalancedDistributedSampler)
                and hasattr(dataset, "pair_group_keys")
                and steps_per_epoch > 0
                and int(train_steps) % int(steps_per_epoch) == 0
            ):
                full_idx = sampler.get_last_full_indices() or []
                used = set(int(i) for i in full_idx)

                keys = list(getattr(dataset, "pair_group_keys") or [])
                total_by_key = Counter(keys)
                used_by_key = Counter(keys[i] for i in used if 0 <= int(i) < len(keys))

                unused_total = int(max(0, len(keys) - len(used)))
                logger.info(f"Epoch {int(epoch)} unused pairs (unique indices): {unused_total}")
                if total_by_key:
                    logger.info(f"Epoch {int(epoch)} unused pairs by key:")
                    for k in sorted(total_by_key.keys()):
                        unused_k = int(total_by_key[k] - used_by_key.get(k, 0))
                        if unused_k > 0:
                            logger.info(f"  {k}: {unused_k}")

            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / max(1, log_steps), device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()

                avg_main = main_loss.detach().float()
                dist.all_reduce(avg_main, op=dist.ReduceOp.SUM)
                avg_main = float((avg_main / dist.get_world_size()).item())

                lat_sum = torch.zeros((), device=device, dtype=torch.float32)
                lat_cnt = torch.zeros((), device=device, dtype=torch.float32)
                if latent_loss is not None:
                    lat_sum = latent_loss.detach().float()
                    lat_cnt = torch.ones((), device=device, dtype=torch.float32)
                dist.all_reduce(lat_sum, op=dist.ReduceOp.SUM)
                dist.all_reduce(lat_cnt, op=dist.ReduceOp.SUM)
                avg_lat = None
                if float(lat_cnt.item()) > 0:
                    avg_lat = float((lat_sum / lat_cnt).item())

                msg = f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Main: {avg_main:.4f}, Train Steps/Sec: {steps_per_sec:.2f}"
                if avg_lat is not None:
                    msg = msg + f", AdapterLat: {avg_lat:.6f} (scale={float(getattr(args, 'latent_loss_scale', 0.0) or 0.0):g})"
                    if latent_keep_ratio is not None:
                        msg = msg + f", keep={float(latent_keep_ratio):.3f}"
                logger.info(msg)
                if tb_writer is not None:
                    tb_writer.add_scalar("train/loss", float(avg_loss), int(train_steps))
                    tb_writer.add_scalar("train/main_loss", float(avg_main), int(train_steps))
                    if avg_lat is not None:
                        tb_writer.add_scalar("train/adapter_latent_loss", float(avg_lat), int(train_steps))
                        if latent_keep_ratio is not None:
                            tb_writer.add_scalar("train/adapter_keep_ratio", float(latent_keep_ratio), int(train_steps))
                    tb_writer.add_scalar("train/steps_per_sec", float(steps_per_sec), int(train_steps))
                # Reset monitoring variables:
                running_loss = 0.0
                log_steps = 0
                start_time = time()

            # Save DiT checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                checkpoint_path = None
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args,
                        "ae_info_by_modality": ae_info_by_modality,
                        "epoch": int(epoch),
                    }
                    if after_adapter is not None:
                        checkpoint["after_adapter"] = after_adapter.module.state_dict()
                        if after_opt is not None:
                            checkpoint["after_opt"] = after_opt.state_dict()
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")

                dist.barrier()

            should_eval = bool(args.run_eval) and (train_steps > 0) and (int(train_steps) % int(getattr(args, "eval_every", args.ckpt_every)) == 0)
            if should_eval:
                eval_distributed = bool(getattr(args, "eval_distributed", False)) and dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
                is_main = (not eval_distributed) or rank == 0

                if not eval_loaders:
                    if is_main:
                        logger.info(f"(step={train_steps:07d}) Eval skipped: eval_loaders is empty")
                else:
                    if is_main:
                        for history_key, item in eval_loaders.items():
                            if item.get("split") != "val":
                                continue
                            v = evaluate_manifest_val_loss(
                                model_for_loss=model.module,
                                diffusion=diffusion,
                                ae_models_by_id=ae_models_by_id,
                                eval_loader=item["loader"],
                                args=args,
                                device=device,
                                train_steps=int(train_steps),
                                epoch=int(epoch),
                                logger=logger,
                                eval_pair_tag=item.get("pair_tag"),
                                eval_out_tag=item.get("out_tag"),
                                pair_weight_table=pair_weight_table,
                            )
                            if v is not None:
                                _append_val_loss_csv_row(experiment_dir, v)

                    model_to_sample = ema if bool(args.eval_use_ema) else model.module
                    do_compare = (
                        (after_adapter is not None)
                        and bool(getattr(args, "enable_after_adapter", False))
                        and bool(getattr(args, "eval_noadapt_baseline", False))
                    )
                    for history_key, item in eval_loaders.items():
                        runs = [("adapt", after_adapter.module), ("noadapt", None)] if bool(do_compare) else [("adapt", after_adapter.module if after_adapter is not None else None)]
                        for run_tag, aa in runs:
                            base_pair_tag = item.get("pair_tag")
                            base_out_tag = item.get("out_tag")
                            pair_tag = None if base_pair_tag is None else f"{base_pair_tag}|{run_tag}"
                            out_tag = None if base_out_tag is None else f"{base_out_tag}_{run_tag}"
                            hist_key = f"{history_key}|{run_tag}" if bool(do_compare) else history_key

                            metrics = evaluate_manifest_raw(
                                model_to_sample=model_to_sample,
                                ae_models_by_id=ae_models_by_id,
                                eval_loader=item["loader"],
                                args=args,
                                device=device,
                                tb_writer=tb_writer,
                                train_steps=train_steps,
                                experiment_dir=experiment_dir,
                                logger=logger,
                                after_adapter=aa,
                                eval_pair_tag=pair_tag,
                                eval_out_tag=out_tag,
                                eval_split=item.get("split"),
                                tb_prefix=f"eval/{out_tag}" if out_tag else "eval",
                            )
                            if is_main and metrics is not None:
                                hist = eval_history_by_pair.setdefault(hist_key, [])
                                hist.append(metrics)

                                csv_rows = None
                                if isinstance(metrics, dict):
                                    csv_rows = metrics.get("csv_rows", None)
                                if not csv_rows:
                                    csv_rows = [metrics]

                                for rr0 in csv_rows:
                                    rr = dict(rr0) if isinstance(rr0, dict) else {}
                                    rr["eval_pair"] = pair_tag
                                    rr["split"] = item.get("split")
                                    rr["dataset"] = rr.get("dataset", "ALL")
                                    rr["train_steps"] = int(train_steps)
                                    rr["epoch"] = int(epoch)
                                    rr["pred_mse"] = "" if rr.get("pred_mse") is None else json.dumps(rr.get("pred_mse"), sort_keys=True)
                                    _append_eval_csv_row(experiment_dir, rr)

                                _monitor_eval(hist, args, logger, tb_writer, train_steps)
                    model.train()
                    ema.eval()
                dist.barrier()

            mx = int(getattr(args, "max_train_steps", 0) or 0)
            if mx > 0 and int(train_steps) >= mx:
                if rank == 0:
                    logger.info(f"Reached --max-train-steps={mx}, stopping training.")
                break

        mx = int(getattr(args, "max_train_steps", 0) or 0)
        if mx > 0 and int(train_steps) >= mx:
            break

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")
    if tb_writer is not None:
        tb_writer.close()
    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    data_group = parser.add_mutually_exclusive_group(required=True)




    parser.add_argument("--eval-preflight", type=_str2bool, default=True, help=" step=0 ")
    parser.add_argument("--eval-preflight-always", type=_str2bool, default=False, help=)
    parser.add_argument(
        "--eval-noadapt-baseline",
        type=_str2bool,
        default=False,
    )
    parser.add_argument("--eval-enable-val", type=_str2bool, default=False, help=" val")
    parser.add_argument("--eval-enable-test", type=_str2bool, default=False, help=" test")
    parser.add_argument("--eval-by-dataset", type=_str2bool, default=True, help=" metrics.csv")
    parser.add_argument("--eval-train-fixed-samples", type=int, default=100, help=)


    parser.add_argument("--monitor-enable", type=_str2bool, default=True)
    parser.add_argument("--monitor-plateau-window", type=int, default=5)
    parser.add_argument("--monitor-plateau-psnr-delta", type=float, default=0.05)
    parser.add_argument("--monitor-plateau-ssim-delta", type=float, default=0.003)
    parser.add_argument("--monitor-regress-psnr", type=float, default=0.2)
    parser.add_argument("--monitor-regress-ssim", type=float, default=0.01)
    parser.add_argument("--monitor-x0-t999-warn", type=float, default=0.8)
    parser.add_argument("--monitor-x0-t999-min-improve", type=float, default=0.01)

    parser.add_argument("--predict-xstart", type=_str2bool, default=False)
    parser.add_argument("--x0-loss-snr-gamma", type=float, default=None)

    parser.add_argument("--enable-after-adapter", type=_str2bool, default=False)
    parser.add_argument("--after-adapter-type", type=str, default="refine")
    parser.add_argument("--after-adapter-lr", type=float, default=None)
    parser.add_argument("--latent-loss-scale", type=float, default=1.0)
    parser.add_argument("--after-adapter-max-t", type=int, default=None)
    parser.add_argument("--resume-after-adapter", type=str, default=None)

    parser.add_argument("--eval-pred-mse", type=_str2bool, default=None)
    parser.add_argument("--eval-eps-mse", type=_str2bool, default=True)
    parser.add_argument("--eval-eps-max-batch", type=int, default=16)
    parser.add_argument("--eval-eps-timesteps", type=int, nargs="+", default=[999, 900, 800, 600, 400, 200, 100, 50, 10, 1, 0])

    default_uni_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "uniTransfer"))
    parser.add_argument("--manifest-data-type", type=str, choices=["raw", "latent"], default="raw")
    parser.add_argument("--uni-transfer-root", type=str, default=default_uni_root)
    parser.add_argument("--ae-configs", type=str, nargs="*", default=None)
    parser.add_argument("--ae-ckpts", type=str, nargs="*", default=None)
    parser.add_argument("--require-ae-ckpts", type=_str2bool, default=False)
    parser.add_argument("--normalization-config-json", type=str, default=None)
    parser.add_argument("--raw-size", type=int, default=None)
    parser.add_argument("--no-resize", type=_str2bool, default=False)

    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")

    parser.add_argument("--latent-size", type=int, default=64)
    parser.add_argument("--latent-channels", type=int, default=4)
    parser.add_argument("--latent-scale-factor", type=float, default=1.0)

    parser.add_argument("--conditioning-mode", type=str, choices=["concat", "crossattn", "hybrid", "none"], default="concat")
    parser.add_argument("--modalities", type=str, nargs="+", default=["opt", "sar", "pan", "nir", "ms"])
    parser.add_argument("--enable-any-to-all", type=_str2bool, default=True)
    parser.add_argument("--allowed-pairs", type=str, nargs="*", default=None)
    parser.add_argument("--augment", type=_str2bool, default=True)
    parser.add_argument("--cond-dropout-prob", type=float, default=0.1)
    parser.add_argument("--modality-dropout-prob", type=float, default=0.0)

    parser.add_argument("--pair-loss-reweight", type=_str2bool, default=True)
    parser.add_argument("--pair-loss-reweight-alpha", type=float, default=0.5)
    parser.add_argument("--pair-loss-reweight-min", type=float, default=0.25)
    parser.add_argument("--pair-loss-reweight-max", type=float, default=4.0)

    parser.add_argument("--train-sampler", type=str, choices=["distributed", "balanced"], default="distributed")
    parser.add_argument("--balance-key", type=str, default="modality_set")
    parser.add_argument("--balanced-cover-all", type=_str2bool, default=False)
    parser.add_argument("--log-unused-samples", action="store_true")


    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=2500)
    parser.add_argument(
        "--save-initial-checkpoint",
        type=_str2bool,
        default=True,
    )
    parser.add_argument(
        "--max-train-steps",
        type=int,
        default=0,
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--resume-no-opt", type=_str2bool, default=False)
    parser.add_argument("--resume-reset-lr", type=_str2bool, default=True)

    args = parser.parse_args()
    main(args)
