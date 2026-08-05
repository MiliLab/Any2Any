# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained DiT.
"""
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image, make_grid
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model
from models import DiT_models
import argparse
import numpy as np
import os
import json
import math
import glob
from PIL import Image
import torch.distributed as dist
import torch.nn.functional as F


def _to_01(x: torch.Tensor) -> torch.Tensor:
    x = torch.clamp(x, -1.0, 1.0)
    return (x + 1.0) / 2.0


def _minmax_01(bchw: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if bchw.dim() != 4:
        raise ValueError(f"Expected BCHW tensor, got {tuple(bchw.shape)}")
    x = bchw.float()
    x_min = x.amin(dim=(1, 2, 3), keepdim=True)
    x_max = x.amax(dim=(1, 2, 3), keepdim=True)
    return (x - x_min) / (x_max - x_min + eps)


def _gaussian_kernel2d(kernel_size: int = 11, sigma: float = 1.5, device=None, dtype=None) -> torch.Tensor:
    coords = torch.arange(kernel_size, device=device, dtype=dtype) - (kernel_size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    kernel_2d = torch.outer(g, g)
    kernel_2d = kernel_2d / kernel_2d.sum()
    return kernel_2d


def _ssim(x01: torch.Tensor, y01: torch.Tensor, kernel_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    if x01.shape != y01.shape:
        raise ValueError(f"SSIM expects same shapes, got {tuple(x01.shape)} vs {tuple(y01.shape)}")
    if x01.dim() != 4:
        raise ValueError(f"SSIM expects BCHW, got {tuple(x01.shape)}")

    b, c, h, w = x01.shape
    device = x01.device
    dtype = x01.dtype

    kernel_2d = _gaussian_kernel2d(kernel_size=kernel_size, sigma=sigma, device=device, dtype=dtype)
    weight = kernel_2d.view(1, 1, kernel_size, kernel_size).repeat(c, 1, 1, 1)
    padding = kernel_size // 2

    mu_x = F.conv2d(x01, weight, padding=padding, groups=c)
    mu_y = F.conv2d(y01, weight, padding=padding, groups=c)

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(x01 * x01, weight, padding=padding, groups=c) - mu_x2
    sigma_y2 = F.conv2d(y01 * y01, weight, padding=padding, groups=c) - mu_y2
    sigma_xy = F.conv2d(x01 * y01, weight, padding=padding, groups=c) - mu_xy

    c1 = (0.01 ** 2)
    c2 = (0.03 ** 2)
    ssim_map = ((2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)) / ((mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2))
    return ssim_map.mean(dim=(1, 2, 3))


def _to_3ch(bchw: torch.Tensor) -> torch.Tensor:
    if bchw.dim() != 4:
        raise ValueError(f"Expected BCHW tensor, got {tuple(bchw.shape)}")
    c = bchw.shape[1]
    if c == 3:
        return bchw
    if c == 1:
        return bchw.repeat(1, 3, 1, 1)
    if c == 2:
        # zeros = torch.zeros((bchw.shape[0], 1, bchw.shape[2], bchw.shape[3]), device=bchw.device, dtype=bchw.dtype)
        # return torch.cat([bchw, zeros], dim=1)
        gray = bchw.mean(dim=1, keepdim=True)
        return gray.repeat(1, 3, 1, 1)

    return bchw[:, :3, ...]


def _load_manifest(manifest_path: str):
    with open(manifest_path, "r") as f:
        m = json.load(f)
    return list(m.values()) if isinstance(m, dict) else list(m)


def _get_mod_path(sample: dict, modality: str):
    mod_lower = str(modality).lower()
    for k, v in sample.items():
        if str(k).lower() == mod_lower:
            return v
    raise KeyError(f"Modality '{modality}' not found in sample keys: {list(sample.keys())}")


def _infer_num_modalities_from_state_dict(state_dict, fallback: int):
    for k in ["src_mod_embedder.embedding_table.weight", "tgt_mod_embedder.embedding_table.weight"]:
        if k in state_dict:
            return int(state_dict[k].shape[0])
    return int(fallback)


@torch.no_grad()
def _encode_raw(ae, x, latent_size, latent_scale_factor, encode_mode: str = "mode"):
    posterior = ae.encode(x)
    if str(encode_mode).lower() == "sample":
        z = posterior.sample()
    else:
        z = posterior.mode()
    if z.shape[-1] != latent_size or z.shape[-2] != latent_size:
        z = F.interpolate(z, size=(latent_size, latent_size), mode="bilinear", align_corners=False)
    if latent_scale_factor != 1.0:
        z = z * latent_scale_factor
    return z


def _stats(x: torch.Tensor) -> dict:
    x = x.detach()
    return {
        "shape": list(x.shape),
        "min": float(x.min().item()),
        "max": float(x.max().item()),
        "mean": float(x.mean().item()),
        "std": float(x.std(unbiased=False).item()),
    }


@torch.no_grad()
def _decode_latent(ae, z, latent_scale_factor):
    if latent_scale_factor != 1.0:
        z = z / latent_scale_factor
    out = ae.decode(z)
    if hasattr(out, "sample"):
        return out.sample
    return out


def _make_denoised_fn(denoised_clamp):
    if denoised_clamp is None:
        return None
    c = float(denoised_clamp)
    if c <= 0:
        return None

    def fn(x):
        return x.clamp(-c, c)

    return fn


def _maybe_init_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return 0, 1, 0, False

    if not torch.cuda.is_available():
        raise RuntimeError("Distributed sampling requested (WORLD_SIZE>1) but CUDA is not available")

    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available")

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")

    rank = int(dist.get_rank())
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank, True


def _write_image_grids(out_dir: str, prefix: str, src_mod_lower: str, tgt_mod_lower: str, every: int, nrow: int):
    every = int(every)
    if every <= 0:
        return
    nrow = max(1, int(nrow))

    pattern = os.path.join(out_dir, f"{prefix}_??????_{src_mod_lower}_to_{tgt_mod_lower}.png")
    paths = sorted(glob.glob(pattern))
    if not paths:
        return

    for start in range(0, len(paths), every):
        chunk = paths[start : start + every]
        imgs = []
        for p in chunk:
            im = Image.open(p).convert("RGB")
            arr = np.array(im, dtype=np.uint8)
            t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
            imgs.append(t)
        batch = torch.stack(imgs, dim=0)
        grid = make_grid(batch, nrow=min(nrow, int(batch.shape[0])))
        out_path = os.path.join(out_dir, f"grid_{prefix}_{start // every:04d}_{src_mod_lower}_to_{tgt_mod_lower}.png")
        save_image(grid, out_path, normalize=False)


def _make_diffusion(args):
    n = int(args.num_sampling_steps)
    sampler = str(getattr(args, "sampler", "p")).lower()
    if sampler == "ddim":
        timestep_respacing = f"ddim{n}"
    else:
        timestep_respacing = str(n)
    return create_diffusion(timestep_respacing, predict_xstart=bool(getattr(args, "predict_xstart", False)))


def _sample_loop(diffusion, args, model_fn, shape, noise, clip_denoised, denoised_fn, model_kwargs, device, progress):
    sampler = str(getattr(args, "sampler", "p")).lower()
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
            eta=float(getattr(args, "ddim_eta", 0.0)),
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


def _load_raw_npy_to_tensor(npy_path, norm, size, device):
    arr = np.load(npy_path)
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
        raise ValueError(f"Unsupported npy shape: {arr.shape}")

    arr = norm.normalize(arr)
    x = torch.from_numpy(arr).float() if isinstance(arr, np.ndarray) else arr.float()

    if size is not None and (x.shape[1] != int(size) or x.shape[2] != int(size)):
        x = F.interpolate(x.unsqueeze(0), size=(int(size), int(size)), mode="bilinear", align_corners=False).squeeze(0)

    return x.to(device)


def main(args):
    rank, world_size, local_rank, is_distributed = _maybe_init_distributed()
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

    if args.task == "eval_manifest":
        assert args.ckpt is not None, "--ckpt is required for --task eval_manifest"
        assert args.manifest is not None, "--manifest is required for --task eval_manifest"
        assert args.out_dir is not None, "--out-dir is required for --task eval_manifest"

        os.makedirs(args.out_dir, exist_ok=True)

        from train import _add_unitransfer_to_path, _parse_kv_list, _default_ae_config_path, _load_unitransfer_ae_and_norm

        _add_unitransfer_to_path(args.uni_transfer_root)
        from ldm.util.normalization import NormalizationManager

        latent_size = int(args.latent_size)
        latent_channels = int(args.latent_channels)

        mod2id = {m.lower(): i for i, m in enumerate(args.modalities)}
        src_mod_lower = args.src_mod.lower()
        tgt_mod_lower = args.tgt_mod.lower()

        ae_config_map = _parse_kv_list(args.ae_configs) if args.ae_configs else {}
        ae_ckpt_map = _parse_kv_list(args.ae_ckpts) if args.ae_ckpts else {}

        src_cfg_path = ae_config_map.get(src_mod_lower, None)
        if src_cfg_path is None:
            src_cfg_path = _default_ae_config_path(args.uni_transfer_root, latent_size, latent_channels, src_mod_lower)
        tgt_cfg_path = ae_config_map.get(tgt_mod_lower, None)
        if tgt_cfg_path is None:
            tgt_cfg_path = _default_ae_config_path(args.uni_transfer_root, latent_size, latent_channels, tgt_mod_lower)

        if not os.path.isfile(src_cfg_path):
            raise FileNotFoundError(f"AE config not found for src modality '{src_mod_lower}': {src_cfg_path}")
        if not os.path.isfile(tgt_cfg_path):
            raise FileNotFoundError(f"AE config not found for tgt modality '{tgt_mod_lower}': {tgt_cfg_path}")

        src_ae, src_norm_cfg, _src_ae_meta = _load_unitransfer_ae_and_norm(
            args.uni_transfer_root, src_cfg_path, device, ckpt_override=ae_ckpt_map.get(src_mod_lower, None)
        )
        tgt_ae, tgt_norm_cfg, _tgt_ae_meta = _load_unitransfer_ae_and_norm(
            args.uni_transfer_root, tgt_cfg_path, device, ckpt_override=ae_ckpt_map.get(tgt_mod_lower, None)
        )

        src_norm = NormalizationManager(**src_norm_cfg)
        tgt_norm = NormalizationManager(**tgt_norm_cfg)

        ckpt = torch.load(args.ckpt, map_location="cpu")
        state_dict = ckpt["ema"] if args.use_ema and isinstance(ckpt, dict) and "ema" in ckpt else ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

        num_modalities_ckpt = _infer_num_modalities_from_state_dict(state_dict, fallback=len(args.modalities))
        src_id = int(mod2id[src_mod_lower]) if args.conditioning_mode != "none" else int(mod2id[tgt_mod_lower])
        tgt_id = int(mod2id[tgt_mod_lower])
        if src_id >= num_modalities_ckpt or tgt_id >= num_modalities_ckpt:
            raise ValueError(
                f"Checkpoint expects num_modalities={num_modalities_ckpt}, but got src_id={src_id}, tgt_id={tgt_id}. "
                f"Make sure --modalities matches the training-time modality list/order."
            )

        model = DiT_models[args.model](
            input_size=latent_size,
            in_channels=latent_channels,
            num_classes=0,
            conditioning_mode=args.conditioning_mode,
            x_channels=latent_channels,
            cond_channels=latent_channels,
            use_modality_embedding=True,
            num_modalities=num_modalities_ckpt,
        ).to(device)
        model.load_state_dict(state_dict, strict=True)
        model.eval()

        diffusion = _make_diffusion(args)
        clip_denoised = bool(args.clip_denoised)
        denoised_fn = _make_denoised_fn(args.denoised_clamp)

        samples_list = _load_manifest(args.manifest)
        max_samples = int(args.max_samples) if args.max_samples is not None else -1
        if max_samples > 0:
            samples_list = samples_list[:max_samples]

        metrics_path = os.path.join(args.out_dir, f"metrics_{src_mod_lower}_to_{tgt_mod_lower}.jsonl")
        metrics_rank_path = (
            metrics_path.replace(".jsonl", f"_rank{rank:02d}.jsonl") if is_distributed else metrics_path
        )

        mses, rmses, psnrs, ssims = [], [], [], []

        indices = list(range(len(samples_list)))
        if is_distributed:
            indices = [i for i in indices if (int(i) % int(world_size) == int(rank))]

        batch_size = int(getattr(args, "batch_size", 1))
        batch_size = max(1, batch_size)

        with open(metrics_rank_path, "w") as f:
            for start in range(0, len(indices), batch_size):
                batch_indices = indices[start : start + batch_size]

                x_src_list = []
                x_tgt_list = []
                src_paths = []
                tgt_paths = []

                for i in batch_indices:
                    sample = samples_list[int(i)]
                    src_path = _get_mod_path(sample, src_mod_lower)
                    tgt_path = _get_mod_path(sample, tgt_mod_lower)
                    src_paths.append(src_path)
                    tgt_paths.append(tgt_path)
                    x_src_list.append(_load_raw_npy_to_tensor(src_path, src_norm, size=args.raw_size, device=device))
                    x_tgt_list.append(_load_raw_npy_to_tensor(tgt_path, tgt_norm, size=args.raw_size, device=device))

                x_src = torch.stack(x_src_list, dim=0)
                x_tgt = torch.stack(x_tgt_list, dim=0)

                encode_mode = str(args.ae_encode_mode).lower()
                if args.debug and len(batch_indices) > 0 and int(batch_indices[0]) == 0:
                    print(json.dumps({"x_src": _stats(x_src), "x_tgt": _stats(x_tgt)}, indent=2))

                src_lat = _encode_raw(
                    src_ae,
                    x_src,
                    latent_size=latent_size,
                    latent_scale_factor=float(args.latent_scale_factor),
                    encode_mode=encode_mode,
                )
                tgt_lat = _encode_raw(
                    tgt_ae,
                    x_tgt,
                    latent_size=latent_size,
                    latent_scale_factor=float(args.latent_scale_factor),
                    encode_mode=encode_mode,
                )
                x_src_rec = _decode_latent(src_ae, src_lat, latent_scale_factor=float(args.latent_scale_factor))
                x_tgt_rec = _decode_latent(tgt_ae, tgt_lat, latent_scale_factor=float(args.latent_scale_factor))

                b = int(x_src.shape[0])

                if args.conditioning_mode == "none":
                    cond = torch.zeros(b, latent_channels, latent_size, latent_size, device=device)
                    src_mod = torch.tensor([mod2id[tgt_mod_lower]] * b, device=device)
                else:
                    cond = src_lat
                    if cond.shape[1] != latent_channels:
                        raise ValueError(f"Encoded cond latent has {cond.shape[1]} channels, expected {latent_channels}")
                    src_mod = torch.tensor([mod2id[src_mod_lower]] * b, device=device)

                if args.debug and len(batch_indices) > 0 and int(batch_indices[0]) == 0:
                    print(json.dumps({"src_lat": _stats(src_lat), "tgt_lat": _stats(tgt_lat)}, indent=2))

                tgt_mod = torch.tensor([mod2id[tgt_mod_lower]] * b, device=device)

                z_list = []
                for i in batch_indices:
                    torch.manual_seed(int(args.seed) + int(i))
                    z_list.append(torch.randn(1, latent_channels, latent_size, latent_size, device=device))
                z = torch.cat(z_list, dim=0)

                cfg_scale = float(args.cfg_scale)
                if cfg_scale == 1.0:
                    model_kwargs = dict(cond=cond, src_mod=src_mod, tgt_mod=tgt_mod)
                    out = _sample_loop(
                        diffusion,
                        args,
                        model.forward,
                        z.shape,
                        z,
                        clip_denoised=clip_denoised,
                        denoised_fn=denoised_fn,
                        model_kwargs=model_kwargs,
                        device=device,
                        progress=False,
                    )
                else:
                    z2 = torch.cat([z, z], 0)
                    cond2 = torch.cat([cond, cond], 0)
                    src_mod2 = torch.cat([src_mod, src_mod], 0)
                    tgt_mod2 = torch.cat([tgt_mod, tgt_mod], 0)

                    model_kwargs = dict(cfg_scale=cfg_scale, cond=cond2, src_mod=src_mod2, tgt_mod=tgt_mod2)

                    out = _sample_loop(
                        diffusion,
                        args,
                        model.forward_with_cfg,
                        z2.shape,
                        z2,
                        clip_denoised=clip_denoised,
                        denoised_fn=denoised_fn,
                        model_kwargs=model_kwargs,
                        device=device,
                        progress=False,
                    )
                    out, _ = out.chunk(2, dim=0)

                x_pred = _decode_latent(tgt_ae, out, latent_scale_factor=float(args.latent_scale_factor))

                if args.debug and len(batch_indices) > 0 and int(batch_indices[0]) == 0:
                    print(json.dumps({"out_lat": _stats(out), "x_pred": _stats(x_pred)}, indent=2))

                x01 = _to_01(x_pred)
                y01 = _to_01(x_tgt)

                mse = torch.mean((x01 - y01) ** 2, dim=(1, 2, 3))
                rmse = torch.sqrt(mse + 1e-12)
                psnr = 10.0 * torch.log10(1.0 / (mse + 1e-12))
                ssim = _ssim(x01, y01)

                xsrc01 = _to_01(x_src)
                xsrcrec01 = _to_01(x_src_rec)
                xtgt01 = y01
                xtgtrec01 = _to_01(x_tgt_rec)

                src_rec_mse = torch.mean((xsrcrec01 - xsrc01) ** 2, dim=(1, 2, 3))
                src_rec_psnr = 10.0 * torch.log10(1.0 / (src_rec_mse + 1e-12))
                src_rec_ssim = _ssim(xsrcrec01, xsrc01)

                tgt_rec_mse = torch.mean((xtgtrec01 - xtgt01) ** 2, dim=(1, 2, 3))
                tgt_rec_psnr = 10.0 * torch.log10(1.0 / (tgt_rec_mse + 1e-12))
                tgt_rec_ssim = _ssim(xtgtrec01, xtgt01)

                for j, i in enumerate(batch_indices):
                    rec = {
                        "index": int(i),
                        "src_path": str(src_paths[j]),
                        "tgt_path": str(tgt_paths[j]),
                        "mse": float(mse[j].item()),
                        "rmse": float(rmse[j].item()),
                        "psnr": float(psnr[j].item()),
                        "ssim": float(ssim[j].item()),
                        "src_recon_psnr": float(src_rec_psnr[j].item()),
                        "src_recon_ssim": float(src_rec_ssim[j].item()),
                        "tgt_recon_psnr": float(tgt_rec_psnr[j].item()),
                        "tgt_recon_ssim": float(tgt_rec_ssim[j].item()),
                    }
                    if args.debug and int(i) == 0:
                        rec["x_pred_stats"] = _stats(x_pred[j : j + 1])
                        rec["x_src_rec_stats"] = _stats(x_src_rec[j : j + 1])
                        rec["x_tgt_rec_stats"] = _stats(x_tgt_rec[j : j + 1])
                    f.write(json.dumps(rec) + "\n")

                    mses.append(float(mse[j].item()))
                    rmses.append(float(rmse[j].item()))
                    psnrs.append(float(psnr[j].item()))
                    ssims.append(float(ssim[j].item()))

                save_every = int(args.save_every)
                if save_every > 0:
                    for j, i in enumerate(batch_indices):
                        if int(i) % save_every != 0:
                            continue
                        panel = torch.cat([
                            _to_3ch(x_src[j : j + 1]),
                            _to_3ch(x_src_rec[j : j + 1]),
                            _to_3ch(x_tgt[j : j + 1]),
                            _to_3ch(x_tgt_rec[j : j + 1]),
                            _to_3ch(x_pred[j : j + 1]),
                        ], dim=3)
                        save_image(
                            panel,
                            os.path.join(args.out_dir, f"vis_{int(i):06d}_{src_mod_lower}_to_{tgt_mod_lower}.png"),
                            nrow=1,
                            normalize=True,
                            value_range=(-1, 1),
                        )

                        if args.debug:
                            panel_auto = torch.cat([
                                _minmax_01(_to_3ch(x_src[j : j + 1])),
                                _minmax_01(_to_3ch(x_src_rec[j : j + 1])),
                                _minmax_01(_to_3ch(x_tgt[j : j + 1])),
                                _minmax_01(_to_3ch(x_tgt_rec[j : j + 1])),
                                _minmax_01(_to_3ch(x_pred[j : j + 1])),
                            ], dim=3)
                            save_image(
                                panel_auto,
                                os.path.join(args.out_dir, f"vis_auto_{int(i):06d}_{src_mod_lower}_to_{tgt_mod_lower}.png"),
                                nrow=1,
                                normalize=False,
                            )

        def _mean_std(xs):
            if not xs:
                return 0.0, 0.0
            m = sum(xs) / len(xs)
            v = sum((x - m) ** 2 for x in xs) / max(1, (len(xs) - 1))
            return m, math.sqrt(v)

        if is_distributed:
            dist.barrier()

            gathered = [None for _ in range(int(world_size))]
            dist.all_gather_object(gathered, (mses, rmses, psnrs, ssims))

            if int(rank) == 0:
                mses = [x for part in gathered for x in part[0]]
                rmses = [x for part in gathered for x in part[1]]
                psnrs = [x for part in gathered for x in part[2]]
                ssims = [x for part in gathered for x in part[3]]

                with open(metrics_path, "w") as fout:
                    for r in range(int(world_size)):
                        p = metrics_path.replace(".jsonl", f"_rank{r:02d}.jsonl")
                        if os.path.isfile(p):
                            with open(p, "r") as fin:
                                for line in fin:
                                    fout.write(line)

                _write_image_grids(args.out_dir, "vis", src_mod_lower, tgt_mod_lower, args.grid_every, args.grid_nrow)
                if args.debug:
                    _write_image_grids(args.out_dir, "vis_auto", src_mod_lower, tgt_mod_lower, args.grid_every, args.grid_nrow)

        if (not is_distributed) or int(rank) == 0:
            summary = {
                "count": int(len(mses)),
                "mse_mean": _mean_std(mses)[0],
                "mse_std": _mean_std(mses)[1],
                "rmse_mean": _mean_std(rmses)[0],
                "rmse_std": _mean_std(rmses)[1],
                "psnr_mean": _mean_std(psnrs)[0],
                "psnr_std": _mean_std(psnrs)[1],
                "ssim_mean": _mean_std(ssims)[0],
                "ssim_std": _mean_std(ssims)[1],
            }
            with open(os.path.join(args.out_dir, f"metrics_summary_{src_mod_lower}_to_{tgt_mod_lower}.json"), "w") as f:
                json.dump(summary, f, indent=2)
            print(json.dumps(summary, indent=2))
        return

    if args.task == "translate_raw":
        assert args.ckpt is not None, "--ckpt is required for --task translate_raw"
        assert args.out_dir is not None, "--out-dir is required for --task translate_raw"
        if args.conditioning_mode != "none":
            assert args.src_npy is not None, "--src-npy is required for --task translate_raw when conditioning_mode != none"

        os.makedirs(args.out_dir, exist_ok=True)

        from train import _add_unitransfer_to_path, _parse_kv_list, _default_ae_config_path, _load_unitransfer_ae_and_norm

        _add_unitransfer_to_path(args.uni_transfer_root)
        from ldm.util.normalization import NormalizationManager

        latent_size = int(args.latent_size)
        latent_channels = int(args.latent_channels)

        mod2id = {m.lower(): i for i, m in enumerate(args.modalities)}
        src_mod_lower = args.src_mod.lower()
        tgt_mod_lower = args.tgt_mod.lower()
        if src_mod_lower not in mod2id:
            raise ValueError(f"Unknown src_mod: {args.src_mod}. Expected one of: {list(mod2id.keys())}")
        if tgt_mod_lower not in mod2id:
            raise ValueError(f"Unknown tgt_mod: {args.tgt_mod}. Expected one of: {list(mod2id.keys())}")

        ae_config_map = _parse_kv_list(args.ae_configs)
        ae_ckpt_map = _parse_kv_list(args.ae_ckpts) if args.ae_ckpts else {}

        src_cfg_path = ae_config_map.get(src_mod_lower, None)
        if src_cfg_path is None:
            src_cfg_path = _default_ae_config_path(args.uni_transfer_root, latent_size, latent_channels, src_mod_lower)
        tgt_cfg_path = ae_config_map.get(tgt_mod_lower, None)
        if tgt_cfg_path is None:
            tgt_cfg_path = _default_ae_config_path(args.uni_transfer_root, latent_size, latent_channels, tgt_mod_lower)

        if not os.path.isfile(src_cfg_path):
            raise FileNotFoundError(f"AE config not found for src modality '{src_mod_lower}': {src_cfg_path}")
        if not os.path.isfile(tgt_cfg_path):
            raise FileNotFoundError(f"AE config not found for tgt modality '{tgt_mod_lower}': {tgt_cfg_path}")

        src_ae, src_norm_cfg, _src_ae_meta = _load_unitransfer_ae_and_norm(
            args.uni_transfer_root, src_cfg_path, device, ckpt_override=ae_ckpt_map.get(src_mod_lower, None)
        )
        tgt_ae, tgt_norm_cfg, _tgt_ae_meta = _load_unitransfer_ae_and_norm(
            args.uni_transfer_root, tgt_cfg_path, device, ckpt_override=ae_ckpt_map.get(tgt_mod_lower, None)
        )

        src_norm = NormalizationManager(**src_norm_cfg)
        tgt_norm = NormalizationManager(**tgt_norm_cfg)

        n = int(args.num_samples)
        z = torch.randn(n, latent_channels, latent_size, latent_size, device=device)

        if args.conditioning_mode == "none":
            cond = torch.zeros_like(z)
            src_mod = torch.tensor([mod2id[tgt_mod_lower]] * n, device=device)
        else:
            x_src = _load_raw_npy_to_tensor(args.src_npy, src_norm, size=args.raw_size, device=device)
            x_src = x_src.unsqueeze(0)
            cond_one = _encode_raw(src_ae, x_src, latent_size=latent_size, latent_scale_factor=float(args.latent_scale_factor), encode_mode=str(args.ae_encode_mode).lower())
            if cond_one.shape[1] != latent_channels:
                raise ValueError(f"Encoded cond latent has {cond_one.shape[1]} channels, expected {latent_channels}")
            cond = cond_one.repeat(n, 1, 1, 1)
            src_mod = torch.tensor([mod2id[src_mod_lower]] * n, device=device)

        tgt_mod = torch.tensor([mod2id[tgt_mod_lower]] * n, device=device)


        ckpt = torch.load(args.ckpt, map_location="cpu")
        state_dict = ckpt["ema"] if args.use_ema and isinstance(ckpt, dict) and "ema" in ckpt else ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

        num_modalities_ckpt = _infer_num_modalities_from_state_dict(state_dict, fallback=len(args.modalities))
        src_id = int(mod2id[tgt_mod_lower]) if args.conditioning_mode == "none" else int(mod2id[src_mod_lower])
        tgt_id = int(mod2id[tgt_mod_lower])
        if src_id >= num_modalities_ckpt or tgt_id >= num_modalities_ckpt:
            raise ValueError(
                f"Checkpoint expects num_modalities={num_modalities_ckpt}, but got src_id={src_id}, tgt_id={tgt_id}. "
                f"Make sure --modalities matches the training-time modality list/order."
            )

        model = DiT_models[args.model](
            input_size=latent_size,
            in_channels=latent_channels,
            num_classes=0,
            conditioning_mode=args.conditioning_mode,
            x_channels=latent_channels,
            cond_channels=latent_channels,
            use_modality_embedding=True,
            num_modalities=num_modalities_ckpt,
        ).to(device)
        model.load_state_dict(state_dict, strict=True)
        model.eval()

        diffusion = _make_diffusion(args)
        clip_denoised = bool(args.clip_denoised)
        denoised_fn = _make_denoised_fn(args.denoised_clamp)

        cfg_scale = float(args.cfg_scale)
        if cfg_scale == 1.0:
            model_kwargs = dict(cond=cond, src_mod=src_mod, tgt_mod=tgt_mod)
            samples = _sample_loop(
                diffusion,
                args,
                model.forward,
                z.shape,
                z,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                model_kwargs=model_kwargs,
                device=device,
                progress=True,
            )
        else:
            z = torch.cat([z, z], 0)
            cond = torch.cat([cond, cond], 0)
            src_mod = torch.cat([src_mod, src_mod], 0)
            tgt_mod = torch.cat([tgt_mod, tgt_mod], 0)

            model_kwargs = dict(cfg_scale=cfg_scale, cond=cond, src_mod=src_mod, tgt_mod=tgt_mod)

            samples = _sample_loop(
                diffusion,
                args,
                model.forward_with_cfg,
                z.shape,
                z,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                model_kwargs=model_kwargs,
                device=device,
                progress=True,
            )
            samples, _ = samples.chunk(2, dim=0)

        np.save(os.path.join(args.out_dir, f"pred_{src_mod_lower}_to_{tgt_mod_lower}_latent.npy"), samples.detach().cpu().numpy())

        x_rec = _decode_latent(tgt_ae, samples, latent_scale_factor=float(args.latent_scale_factor))
        np.save(os.path.join(args.out_dir, f"pred_{src_mod_lower}_to_{tgt_mod_lower}_raw.npy"), x_rec.detach().cpu().numpy())

        if args.denorm_out:
            x_denorm = tgt_norm.denormalize(x_rec)
            np.save(os.path.join(args.out_dir, f"pred_{src_mod_lower}_to_{tgt_mod_lower}_denorm.npy"), x_denorm.detach().cpu().numpy())

        if x_rec.ndim == 4 and x_rec.shape[1] == 3:
            save_image(
                x_rec,
                os.path.join(args.out_dir, f"pred_{src_mod_lower}_to_{tgt_mod_lower}.png"),
                nrow=min(n, 4),
                normalize=True,
                value_range=(-1, 1),
            )
        return

    if args.task == "latent":
        assert args.ckpt is not None, "--ckpt is required for --task latent"
        ckpt = torch.load(args.ckpt, map_location="cpu")
        state_dict = ckpt["ema"] if args.use_ema and isinstance(ckpt, dict) and "ema" in ckpt else ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

        latent_size = args.latent_size
        latent_channels = args.latent_channels

        model = DiT_models[args.model](
            input_size=latent_size,
            in_channels=latent_channels,
            num_classes=0,
            conditioning_mode=args.conditioning_mode,
            x_channels=latent_channels,
            cond_channels=latent_channels,
            use_modality_embedding=True,
            num_modalities=len(args.modalities),
        ).to(device)
        model.load_state_dict(state_dict, strict=True)
        model.eval()

        diffusion = _make_diffusion(args)
        clip_denoised = bool(args.clip_denoised)
        denoised_fn = _make_denoised_fn(args.denoised_clamp)

        n = args.num_samples
        z = torch.randn(n, latent_channels, latent_size, latent_size, device=device)

        mod2id = {m.lower(): i for i, m in enumerate(args.modalities)}
        src_mod = torch.tensor([mod2id[args.src_mod.lower()]] * n, device=device)
        tgt_mod = torch.tensor([mod2id[args.tgt_mod.lower()]] * n, device=device)

        if args.conditioning_mode == "none":
            cond = torch.zeros_like(z)
        else:
            cond_arr = np.load(args.cond_npy)
            if cond_arr.ndim == 2:
                cond_arr = cond_arr[None, :, :]
            elif cond_arr.ndim == 3 and cond_arr.shape[2] <= 16 and cond_arr.shape[0] > 16:
                cond_arr = cond_arr.transpose(2, 0, 1)
            cond = torch.from_numpy(cond_arr).float().to(device)
            if cond.shape[1] != latent_size or cond.shape[2] != latent_size:
                cond = torch.nn.functional.interpolate(cond.unsqueeze(0), size=(latent_size, latent_size), mode="bilinear", align_corners=False).squeeze(0)
            cond = cond.unsqueeze(0).repeat(n, 1, 1, 1)
            if args.latent_scale_factor != 1.0:
                cond = cond * args.latent_scale_factor


        cfg_scale = float(args.cfg_scale)
        if cfg_scale == 1.0:
            model_kwargs = dict(cond=cond, src_mod=src_mod, tgt_mod=tgt_mod)
            samples = _sample_loop(
                diffusion,
                args,
                model.forward,
                z.shape,
                z,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                model_kwargs=model_kwargs,
                device=device,
                progress=True,
            )
        else:
            z = torch.cat([z, z], 0)
            cond = torch.cat([cond, cond], 0)
            src_mod = torch.cat([src_mod, src_mod], 0)
            tgt_mod = torch.cat([tgt_mod, tgt_mod], 0)

            model_kwargs = dict(cfg_scale=cfg_scale, cond=cond, src_mod=src_mod, tgt_mod=tgt_mod)

            samples = _sample_loop(
                diffusion,
                args,
                model.forward_with_cfg,
                z.shape,
                z,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                model_kwargs=model_kwargs,
                device=device,
                progress=True,
            )
            samples, _ = samples.chunk(2, dim=0)
        np.save(args.out_npy, samples.detach().cpu().numpy())
        return

    if args.ckpt is None:
        assert args.model == "DiT-XL/2", "Only DiT-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000

    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    ).to(device)
    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()
    diffusion = create_diffusion(str(args.num_sampling_steps), predict_xstart=bool(getattr(args, "predict_xstart", False)))
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    class_labels = [207, 360, 387, 974, 88, 979, 417, 279]

    n = len(class_labels)
    z = torch.randn(n, 4, latent_size, latent_size, device=device)
    y = torch.tensor(class_labels, device=device)

    z = torch.cat([z, z], 0)
    y_null = torch.tensor([1000] * n, device=device)
    y = torch.cat([y, y_null], 0)
    model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)

    samples = diffusion.p_sample_loop(
        model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
    )
    samples, _ = samples.chunk(2, dim=0)
    samples = vae.decode(samples / 0.18215).sample

    save_image(samples, "sample.png", nrow=4, normalize=True, value_range=(-1, 1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, choices=["imagenet", "latent", "translate_raw", "eval_manifest"], default="imagenet")

    parser.add_argument("--uni-transfer-root", type=str, default="../uniTransfer")
    parser.add_argument("--src-npy", type=str, default=None)
    parser.add_argument("--manifest", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--grid-every", type=int, default=0)
    parser.add_argument("--grid-nrow", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--raw-size", type=int, default=256)
    parser.add_argument("--ae-configs", type=str, nargs="*", default=None)
    parser.add_argument("--ae-ckpts", type=str, nargs="*", default=None)
    parser.add_argument("--ae-encode-mode", type=str, choices=["sample", "mode"], default="mode")
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--denorm-out", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--clip-denoised", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--denoised-clamp", type=float, default=0.0)

    parser.add_argument("--predict-xstart", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)

    parser.add_argument("--latent-size", type=int, default=64)
    parser.add_argument("--latent-channels", type=int, default=4)
    parser.add_argument("--latent-scale-factor", type=float, default=1.0)

    parser.add_argument("--conditioning-mode", type=str, choices=["concat", "crossattn", "hybrid", "none"], default="concat")
    parser.add_argument("--modalities", type=str, nargs="+", default=["opt", "sar", "pan", "nir", "ms"])
    parser.add_argument("--cond-npy", type=str, default=None)
    parser.add_argument("--src-mod", type=str, default="opt")
    parser.add_argument("--tgt-mod", type=str, default="sar")
    parser.add_argument("--out-npy", type=str, default="sample_latent.npy")
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--use-ema", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--sampler", type=str, choices=["p", "ddim"], default="ddim")
    parser.add_argument("--ddim-eta", type=float, default=0.0)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    args = parser.parse_args()
    main(args)
