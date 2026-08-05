import argparse
import csv
import math
import os

import torch
import torch.distributed as dist
from tqdm import tqdm
from torchvision.utils import save_image

import train as dit_train
from models import DiT_models
from ldm.modules.adapter import MultiModalAdapterManager


def _safe_dirname(name: str) -> str:
    name = str(name).strip()
    name = name.replace(os.sep, "_")
    return name


def _image_folder_name(src_path: str | None, tgt_path: str | None) -> str:
    p = tgt_path or src_path or ""
    p = str(p)
    parent = os.path.basename(os.path.dirname(p))
    base = os.path.splitext(os.path.basename(p))[0]

    if parent and base:
        out = f"{parent}_{base}"
    else:
        out = parent or base

    out = out if out else "unknown"
    return _safe_dirname(out)


def _vis_3ch(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 3:
        raise ValueError(f"Expected CHW, got {list(x.shape)}")
    if int(x.shape[0]) == 1:
        return dit_train._ensure_three_channels_gray(x)
    return dit_train._ensure_three_channels(x)


def _rmse_scale_for_modality(dataset, tgt_name: str) -> float:
    nm = None
    try:
        nm = getattr(dataset, "norm_managers", {}).get(str(tgt_name).lower(), None)
    except Exception:
        nm = None
    if nm is None:
        return 1.0
    norm = str(getattr(nm, "normalization", "")).lower()
    if norm == "uint8":
        return 255.0
    if norm == "zero_one":
        return 1.0
    dmin = getattr(nm, "data_min", None)
    dmax = getattr(nm, "data_max", None)
    if dmin is not None and dmax is not None:
        try:
            if isinstance(dmin, (list, tuple)) or isinstance(dmax, (list, tuple)):
                return float(max(dmax) - min(dmin))
            return float(dmax) - float(dmin)
        except Exception:
            return 1.0
    return 1.0


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-file", type=str, required=True)
    parser.add_argument("--manifest-base-dir", type=str, default=".")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--uni-transfer-root", type=str, default=".")

    parser.add_argument("--model", type=str, default="DiT-L/4")
    parser.add_argument("--latent-size", type=int, default=64)
    parser.add_argument("--latent-channels", type=int, default=4)
    parser.add_argument("--predict-xstart", type=dit_train._str2bool, default=True)

    parser.add_argument("--conditioning-mode", type=str, default="concat")
    parser.add_argument("--modalities", type=str, nargs="+", default=["sar", "opt", "ms", "nir", "pan"])

    parser.add_argument(
        "--latent-scale-factors",
        type=str,
        nargs="*",
        default=["sar=0.422003", "opt=0.387068", "ms=0.484645", "nir=0.568811", "pan=0.447582"],
    )
    parser.add_argument("--latent-scale-factor", type=float, default=1.0)

    parser.add_argument(
        "--ae-ckpts",
        type=str,
        nargs="*",
        required=True,
        help="VAE checkpoints in format: modality=/path/to/checkpoint.ckpt",
    )
    parser.add_argument("--require-ae-ckpts", type=dit_train._str2bool, default=True)
    parser.add_argument("--no-resize", type=dit_train._str2bool, default=True)

    parser.add_argument("--enable-any-to-all", type=dit_train._str2bool, default=True)
    parser.add_argument(
        "--allowed-pairs",
        type=str,
        nargs="*",
        default=[
            "opt->pan",
            "pan->opt",
        ],
    )

    parser.add_argument("--eval-sampler", type=str, default="ddim")
    parser.add_argument("--eval-num-sampling-steps", type=int, default=250)
    parser.add_argument("--eval-ddim-eta", type=float, default=0.0)
    parser.add_argument("--eval-cfg-scale", type=float, default=1.0)
    parser.add_argument("--eval-clip-denoised", type=dit_train._str2bool, default=False)
    parser.add_argument("--eval-denoised-clamp", type=float, default=None)

    parser.add_argument("--enable-after-adapter", type=dit_train._str2bool, default=True)
    parser.add_argument("--after-adapter-type", type=str, default="refine")
    parser.add_argument("--use-ema", type=dit_train._str2bool, default=True)

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--limit", type=int, default=-1)

    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    distributed = ("RANK" in os.environ) and ("WORLD_SIZE" in os.environ) and (int(os.environ.get("WORLD_SIZE", "1")) > 1)
    if distributed:
        dist.init_process_group("nccl")
        rank = int(dist.get_rank())
        world_size = int(dist.get_world_size())
        local_rank = int(os.environ.get("LOCAL_RANK", str(rank % max(1, torch.cuda.device_count()))))
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        rank = 0
        world_size = 1
        local_rank = 0
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    torch.manual_seed(0 + int(rank))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0 + int(rank))

    dit_train._add_unitransfer_to_path(args.uni_transfer_root)

    mods = [str(m).strip().lower() for m in (args.modalities or [])]
    resolved_scales = dit_train._parse_latent_scale_factors(args.latent_scale_factors, mods, float(args.latent_scale_factor))
    latent_scale_by_id = {int(i): float(resolved_scales[m]) for i, m in enumerate(mods)}

    ae_ckpts_map = dit_train._parse_kv_list(args.ae_ckpts)
    if bool(args.require_ae_ckpts):
        missing = [m for m in mods if m not in ae_ckpts_map]
        if missing:
            raise ValueError(f"require-ae-ckpts=True but missing ae ckpts for: {missing}")

    ae_models_by_id = {}
    normalization_configs = {}
    for i, m in enumerate(mods):
        cfg_path = dit_train._default_ae_config_path(args.uni_transfer_root, int(args.latent_size), int(args.latent_channels), m)
        ckpt_override = ae_ckpts_map.get(m, None)
        ae_model, norm_cfg, _meta = dit_train._load_unitransfer_ae_and_norm(
            args.uni_transfer_root, cfg_path, device, ckpt_override=ckpt_override
        )
        ae_models_by_id[int(i)] = ae_model
        normalization_configs[m] = dict(norm_cfg or {})

    model = DiT_models[args.model](
        input_size=int(args.latent_size),
        in_channels=int(args.latent_channels),
        num_classes=0,
        conditioning_mode=str(args.conditioning_mode),
        x_channels=int(args.latent_channels),
        cond_channels=int(args.latent_channels),
        cond_dropout_prob=0.0,
        use_modality_embedding=True,
        num_modalities=len(mods),
        modality_dropout_prob=0.0,
    ).to(device)
    dit_train.requires_grad(model, False)
    model.eval()

    ckpt = torch.load(str(args.ckpt), map_location="cpu")
    state = ckpt["ema"] if bool(args.use_ema) else ckpt["model"]
    model.load_state_dict(state, strict=True)

    after_adapter = None
    if bool(args.enable_after_adapter):
        adapter_cfgs = {m: {"after": {"type": str(args.after_adapter_type)}} for m in mods}
        after_adapter = MultiModalAdapterManager(
            modalities=mods,
            latent_channels=int(args.latent_channels),
            enable_adapters=True,
            adapter_type=str(args.after_adapter_type),
            adapter_configs=adapter_cfgs,
            build_before=False,
            build_after=True,
        ).to(device)
        dit_train.requires_grad(after_adapter, False)
        after_adapter.eval()
        if isinstance(ckpt, dict) and isinstance(ckpt.get("after_adapter", None), dict):
            after_adapter.load_state_dict(ckpt["after_adapter"], strict=True)

    eval_diffusion = dit_train._make_eval_diffusion(args.eval_num_sampling_steps, args.eval_sampler, predict_xstart=bool(args.predict_xstart))
    denoised_fn = dit_train._make_denoised_fn(args.eval_denoised_clamp)

    dataset = dit_train.ManifestRawPairDataset(
        manifest_file=str(args.manifest_file),
        modalities=mods,
        conditioning_mode=str(args.conditioning_mode),
        enable_any_to_all=bool(args.enable_any_to_all),
        allowed_pairs=list(args.allowed_pairs or []),
        augment=False,
        size=None,
        normalization_configs=normalization_configs,
        no_resize=bool(args.no_resize),
        base_dir=str(args.manifest_base_dir),
        balance_key="pair",
    )

    task_to_indices = {}
    pair_keys = getattr(dataset, "pair_group_keys", None)
    if not pair_keys:
        task_to_indices["ALL"] = list(range(len(dataset)))
    else:
        for i, k in enumerate(list(pair_keys)):
            kk = "ALL" if k is None else str(k)
            if kk not in task_to_indices:
                task_to_indices[kk] = []
            task_to_indices[kk].append(int(i))

    task_list = sorted(task_to_indices.keys())

    if int(rank) == 0:
        os.makedirs(str(args.out_dir), exist_ok=True)
    if distributed:
        dist.barrier()

    csv_fieldnames = [
        "task",
        "image_name",
        "src_mod",
        "tgt_mod",
        "src_path",
        "tgt_path",
        "src_png",
        "tgt_gt_png",
        "pred_png",
        "psnr",
        "ssim",
        "rmse",
    ]

    processed = 0
    for task_key in task_list:
        task_dir_for_csv = os.path.join(str(args.out_dir), _safe_dirname(task_key))
        os.makedirs(task_dir_for_csv, exist_ok=True)
        csv_path_rank = os.path.join(task_dir_for_csv, f"metrics.rank{int(rank):02d}.csv")
        fcsv = open(csv_path_rank, "w", newline="", encoding="utf-8")
        writer = csv.DictWriter(fcsv, fieldnames=csv_fieldnames)
        writer.writeheader()

        try:
            all_indices = list(task_to_indices.get(task_key, []))
            if distributed:
                my_indices = all_indices[int(rank) :: int(world_size)]
            else:
                my_indices = all_indices

            if not my_indices:
                continue

            subset = torch.utils.data.Subset(dataset, my_indices)
            loader = torch.utils.data.DataLoader(
                subset,
                batch_size=int(args.batch_size),
                shuffle=False,
                num_workers=int(args.num_workers),
                pin_memory=True,
                collate_fn=dit_train.raw_manifest_collate,
                drop_last=False,
            )

            total_batches = int(len(loader))
            desc = f"{task_key}"

            it = tqdm(
                loader,
                total=total_batches,
                desc=desc,
                position=int(rank),
                dynamic_ncols=True,
                leave=(int(rank) == 0),
            )

            for batch in it:
                x_raw_list = list(batch["x_raw"])
                cond_raw_list = list(batch["cond_raw"])
                src_paths = list(batch.get("src_path") or [])
                tgt_paths = list(batch.get("tgt_path") or [])

                src_mod = batch["src_mod"].to(device, non_blocking=True)
                tgt_mod = batch["tgt_mod"].to(device, non_blocking=True)
                src_mod_cpu = src_mod.detach().cpu()
                tgt_mod_cpu = tgt_mod.detach().cpu()

                if any(x is None for x in cond_raw_list):
                    cond_lat = torch.zeros(
                        len(cond_raw_list),
                        int(args.latent_channels),
                        int(args.latent_size),
                        int(args.latent_size),
                        device=device,
                    )
                else:
                    cond_lat = dit_train._encode_raw_list_to_latents(
                        cond_raw_list,
                        src_mod_cpu,
                        ae_models_by_id,
                        int(args.latent_size),
                        int(args.latent_channels),
                        latent_scale_by_id,
                        device,
                        strict_no_resize=bool(args.no_resize),
                    )

                b = int(src_mod.shape[0])
                z = torch.randn(b, int(args.latent_channels), int(args.latent_size), int(args.latent_size), device=device)

                model_fn = model.forward
                model_kwargs = dict(cond=cond_lat, src_mod=src_mod, tgt_mod=tgt_mod)
                z_in = z
                shape_in = z.shape

                cfg_scale = float(args.eval_cfg_scale)
                if cfg_scale != 1.0:
                    z_in = torch.cat([z, z], 0)
                    cond2 = torch.cat([cond_lat, cond_lat], 0)
                    src2 = torch.cat([src_mod, src_mod], 0)
                    tgt2 = torch.cat([tgt_mod, tgt_mod], 0)
                    model_kwargs = dict(cfg_scale=cfg_scale, cond=cond2, src_mod=src2, tgt_mod=tgt2)
                    model_fn = model.forward_with_cfg
                    shape_in = z_in.shape

                pred_lat = dit_train._sample_loop(
                    eval_diffusion,
                    str(args.eval_sampler).lower(),
                    model_fn,
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

                if after_adapter is not None and bool(args.enable_after_adapter):
                    out_lat = pred_lat.clone()
                    for mid in torch.unique(tgt_mod).tolist():
                        mi = int(mid)
                        if 0 <= mi < len(mods):
                            name = mods[mi]
                            idx = (tgt_mod == mi).nonzero(as_tuple=False).flatten()
                            if int(idx.numel()) > 0:
                                out_lat[idx] = after_adapter.forward_after(out_lat[idx], name)
                    pred_lat = out_lat

                pred_raw = dit_train._decode_latents_by_modality(pred_lat, tgt_mod_cpu, ae_models_by_id, latent_scale_by_id, device)
                gt_raw = torch.stack(x_raw_list, dim=0).to(device, non_blocking=True)

                psnr_b, ssim_b, rmse_b = dit_train._compute_batch_metrics(pred_raw, gt_raw)
                psnr_b = psnr_b.tolist()
                ssim_b = ssim_b.tolist()
                rmse_b = rmse_b.tolist()

                for i in range(b):
                    sid = int(src_mod_cpu[i].item())
                    tid = int(tgt_mod_cpu[i].item())
                    src_name = mods[sid] if 0 <= sid < len(mods) else f"id{sid}"
                    tgt_name = mods[tid] if 0 <= tid < len(mods) else f"id{tid}"
                    task = f"{src_name}->{tgt_name}"

                    sp = None if i >= len(src_paths) else src_paths[i]
                    tp = None if i >= len(tgt_paths) else tgt_paths[i]
                    image_name = _image_folder_name(sp, tp)

                    task_dir = os.path.join(str(args.out_dir), _safe_dirname(task))
                    img_dir = os.path.join(task_dir, _safe_dirname(image_name))
                    os.makedirs(img_dir, exist_ok=True)

                    src_png = os.path.join(img_dir, "src.png")
                    tgt_png = os.path.join(img_dir, "tgt_gt.png")
                    pred_png = os.path.join(img_dir, "pred.png")

                    src_raw = cond_raw_list[i]
                    if src_raw is None:
                        src_raw = torch.zeros_like(x_raw_list[i])
                    gt_i = gt_raw[i].detach().cpu().float()
                    pr_i = pred_raw[i].detach().cpu().float()

                    save_image(dit_train._to_01(_vis_3ch(src_raw.detach().cpu().float())), src_png)
                    save_image(dit_train._to_01(_vis_3ch(gt_i)), tgt_png)
                    save_image(dit_train._to_01(_vis_3ch(pr_i)), pred_png)

                    rmse_scale = _rmse_scale_for_modality(dataset, tgt_name)
                    writer.writerow(
                        {
                            "task": task,
                            "image_name": image_name,
                            "src_mod": src_name,
                            "tgt_mod": tgt_name,
                            "src_path": "" if sp is None else str(sp),
                            "tgt_path": "" if tp is None else str(tp),
                            "src_png": src_png,
                            "tgt_gt_png": tgt_png,
                            "pred_png": pred_png,
                            "psnr": float(psnr_b[i]),
                            "ssim": float(ssim_b[i]),
                            "rmse": float(rmse_b[i]) * float(rmse_scale),
                        }
                    )

                    processed += 1
                    if int(args.limit) > 0 and processed >= int(args.limit):
                        break

                if int(args.limit) > 0 and processed >= int(args.limit):
                    break
        finally:
            fcsv.close()

        if int(args.limit) > 0 and processed >= int(args.limit):
            break

    if distributed:
        dist.barrier()
        if int(rank) == 0:
            for task_key in task_list:
                task_dir_for_csv = os.path.join(str(args.out_dir), _safe_dirname(task_key))
                out_csv = os.path.join(task_dir_for_csv, "metrics.csv")
                with open(out_csv, "w", newline="", encoding="utf-8") as fout:
                    out_writer = csv.DictWriter(fout, fieldnames=csv_fieldnames)
                    out_writer.writeheader()
                    for r in range(int(world_size)):
                        p = os.path.join(task_dir_for_csv, f"metrics.rank{int(r):02d}.csv")
                        if not os.path.isfile(p):
                            continue
                        with open(p, "r", newline="", encoding="utf-8") as fin:
                            reader = csv.DictReader(fin)
                            for row in reader:
                                out_writer.writerow(row)
        dist.barrier()
        dist.destroy_process_group()
    else:
        for task_key in task_list:
            task_dir_for_csv = os.path.join(str(args.out_dir), _safe_dirname(task_key))
            p = os.path.join(task_dir_for_csv, f"metrics.rank{int(rank):02d}.csv")
            if os.path.isfile(p):
                os.replace(p, os.path.join(task_dir_for_csv, "metrics.csv"))


if __name__ == "__main__":
    main()