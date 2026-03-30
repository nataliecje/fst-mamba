#!/usr/bin/env python3
"""
Integrated Gradients (IG) and gradient×input saliency for FST-Mamba / FunctionalMambaMultiLayerST.

Treats the FC input as an edge-weighted graph: spatial axes (H, W) index region–region
interactions. Produces a symmetrized H×W attribution map and ranks nodes by total
interaction importance (sum of |attribution| over incident edges).

Run from repository root, e.g.:
  python interpretability/integrated_gradients_saliency.py \\
    --checkpoint checkpoints/trained_models/WM_LR/FunctionalMamba_WM_LR-sex_0_auc.pth \\
    --dataset_name WM_LR --target_name sex --fold_id 0 --sample_index 0
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Repo root on sys.path
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dataset import get_data
from FunctionalMamba.functional_mamba.model_hub import mambaf_multi_st_base_v1_004


def _scalar_output_for_attribution(
    outputs: torch.Tensor, num_classes: int, target_class: Optional[int]
) -> torch.Tensor:
    """Reduce model output to a single scalar per batch element for backprop."""
    if num_classes == 1:
        return outputs.view(outputs.shape[0], -1)[:, 0]
    if target_class is None:
        raise ValueError("For num_classes > 1, pass --target_class (logit index).")
    return outputs[:, target_class]


def integrated_gradients(
    model: nn.Module,
    x: torch.Tensor,
    baseline: torch.Tensor,
    num_classes: int,
    target_class: Optional[int],
    n_steps: int = 50,
) -> torch.Tensor:
    """
    IG for input x with same shape as model input (B, C, H, W, T).
    Returns tensor of same shape as x.
    """
    x = x.detach()
    baseline = baseline.detach()
    diff = x - baseline
    grads = []
    # Riemann approximation along straight-line path
    for i in range(n_steps):
        alpha = (i + 0.5) / n_steps
        x_interp = baseline + alpha * diff
        x_interp = x_interp.clone().requires_grad_(True)
        out = model(x_interp)
        scalar = _scalar_output_for_attribution(out, num_classes, target_class).sum()
        scalar.backward()
        g = x_interp.grad
        if g is None:
            raise RuntimeError("Gradients are None; check model and inputs.")
        grads.append(g.detach())
        model.zero_grad(set_to_none=True)
    avg_grad = torch.stack(grads, dim=0).mean(dim=0)
    return diff * avg_grad


def gradient_input_saliency(
    model: nn.Module,
    x: torch.Tensor,
    num_classes: int,
    target_class: Optional[int],
) -> torch.Tensor:
    """Vanilla gradient × input (one backward). Same shape as x."""
    x = x.clone().detach().requires_grad_(True)
    out = model(x)
    scalar = _scalar_output_for_attribution(out, num_classes, target_class).sum()
    scalar.backward()
    g = x.grad
    if g is None:
        raise RuntimeError("Gradients are None.")
    return x.detach() * g.detach()


def to_edge_map(attrib: torch.Tensor) -> np.ndarray:
    """
    (B, C, H, W, T) -> (H, W) by mean absolute attribution over batch, channel, time.
    """
    a = attrib.detach().abs().float().cpu().numpy()
    # mean over B, C, T -> H, W
    return np.mean(a, axis=(0, 1, 4))


def symmetrize(mat: np.ndarray) -> np.ndarray:
    return (mat + mat.T) * 0.5


def node_importance(edge_map_sym: np.ndarray) -> np.ndarray:
    """Sum of |edge weights| incident to each node (row/column of symmetric FC)."""
    return np.sum(np.abs(edge_map_sym), axis=1)


def top_k_nodes(scores: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    k = min(k, len(scores))
    idx = np.argsort(-scores)[:k]
    return idx, scores[idx]


def maybe_resize_spatial(x: torch.Tensor, img_size: int | None) -> torch.Tensor:
    """x: (B, C, H, W, T). Bilinear resize H,W if img_size is set and differs."""
    if img_size is None:
        return x
    b, c, h, w, t = x.shape
    if h == img_size and w == img_size:
        return x
    x = x.permute(0, 4, 1, 2, 3).contiguous().view(b * t, c, h, w)
    x = torch.nn.functional.interpolate(x, size=(img_size, img_size), mode="bilinear", align_corners=False)
    x = x.view(b, t, c, img_size, img_size).permute(0, 2, 3, 4, 1).contiguous()
    return x


def parse_args():
    p = argparse.ArgumentParser(description="IG + saliency maps and top-k interaction nodes")
    p.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to .pth (model_state_dict). Training saves under "
        "checkpoints/trained_models/<dataset>/<model_name>/ e.g. .../WM_LR/FunctionalMamba_WM_LR-fluid_cognition/...pth",
    )
    p.add_argument("--dataset_name", type=str, default="WM_LR")
    p.add_argument("--target_name", type=str, default="sex")
    p.add_argument("--image_format", type=str, default="2DT")
    p.add_argument("--task", type=str, default="classification")
    p.add_argument("--num_classes", type=int, default=1)
    p.add_argument("--fold_id", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--cuda_id", type=int, default=0)
    p.add_argument("--sample_index", type=int, default=0, help="Index within test split")
    p.add_argument("--n_steps", type=int, default=50, help="IG Riemann steps")
    p.add_argument("--baseline", type=str, default="zeros", choices=("zeros", "mean"))
    p.add_argument("--target_class", type=int, default=None, help="Class logit for num_classes > 1")
    p.add_argument("--top_k", type=int, default=5)
    p.add_argument(
        "--img_size",
        type=int,
        default=None,
        help="Spatial size H=W for model input (match training). If omitted, uses main.py rule: 256 for EMOTION/WM else 64.",
    )
    p.add_argument(
        "--spatial_resize",
        type=int,
        default=None,
        help="Optional: upsample attribution map to this H=W for saving/plots (e.g. 264 to match ROI grid).",
    )
    p.add_argument("--output_dir", type=str, default="interpretability_outputs")
    p.add_argument("--save_prefix", type=str, default="ig_saliency")
    p.add_argument(
        "--roi_names",
        type=str,
        default=None,
        help="Optional text file with one ROI name per line (length must equal H after model resize) to label printed nodes.",
    )
    return p.parse_args()


def build_model(args, device: torch.device) -> nn.Module:
    img_size = args.img_size
    if img_size is None:
        img_size = 256 if ("EMOTION" in args.dataset_name or "WM" in args.dataset_name) else 64
    bimamba_type = ["bi_mamba"] if "WM" in args.dataset_name else ["bi_st"]
    model = mambaf_multi_st_base_v1_004(
        in_chans=1,
        num_classes=args.num_classes,
        bimamba_type=bimamba_type,
        img_size=img_size,
        drop_rate=0.0,
        drop_path_rate=0.0,
    )
    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print("load_state_dict missing keys (first 10):", missing[:10])
    if unexpected:
        print("load_state_dict unexpected keys (first 10):", unexpected[:10])
    model.eval()
    return model.to(device)


def _resolve_checkpoint_path(path_str: str) -> Path:
    p = Path(path_str)
    if not p.is_absolute():
        p = (_ROOT / p).resolve()
    else:
        p = p.resolve()
    return p


def main():
    args = parse_args()
    ckpt_path = _resolve_checkpoint_path(args.checkpoint)
    if not ckpt_path.is_file():
        extra_lines: list[str] = []
        parent = ckpt_path.parent
        if parent.is_dir():
            contents = sorted(parent.iterdir())
            if contents:
                extra_lines.append(
                    "  Contents of expected folder:\n    "
                    + "\n    ".join(p.name for p in contents)
                )
            else:
                extra_lines.append(f"  Folder exists but is empty: {parent}")
        wm_dir = _ROOT / "checkpoints/trained_models/WM_LR"
        if wm_dir.is_dir():
            found = sorted(wm_dir.rglob("*.pth"))
            if found:
                rel = [f.relative_to(_ROOT) for f in found]
                extra_lines.append(
                    "  Any .pth under checkpoints/trained_models/WM_LR:\n    "
                    + "\n    ".join(str(p) for p in rel)
                )
        any_ckpt = sorted((_ROOT / "checkpoints").rglob("*.pth")) if (_ROOT / "checkpoints").is_dir() else []
        if any_ckpt and not extra_lines:
            extra_lines.append(
                "  Found checkpoints elsewhere under checkpoints/:\n    "
                + "\n    ".join(str(p.relative_to(_ROOT)) for p in any_ckpt[:25])
                + (f"\n    ... ({len(any_ckpt)} total)" if len(any_ckpt) > 25 else "")
            )
        extra = "\n" + "\n".join(extra_lines) if extra_lines else ""
        raise SystemExit(
            f"Checkpoint not found:\n  {ckpt_path}\n\n"
            "Fix: pass the path to an existing .pth. Training only writes these when you run "
            "main.py with --save_model; filenames look like "
            "<model_name>_<fold>_<metric>.pth (e.g. ..._0_mae.pth or ..._0_r2.pth for regression).\n"
            f"{extra}"
        )
    args.checkpoint = str(ckpt_path)

    device = torch.device(f"cuda:{args.cuda_id}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ns = argparse.Namespace(
        num_classes=args.num_classes,
        task=args.task,
        dataset_name=args.dataset_name,
        batch_size=args.batch_size,
    )
    _, loader_test = get_data(
        args.fold_id,
        ns,
        data_name=args.dataset_name,
        target=args.target_name,
        format=args.image_format,
    )

    test_ds = loader_test.dataset
    n = len(test_ds)
    if args.sample_index < 0 or args.sample_index >= n:
        raise SystemExit(f"sample_index must be in [0, {n - 1}]")

    # Single sample batch
    x0, y0 = test_ds[args.sample_index]
    x = x0.unsqueeze(0).to(device)
    label = y0

    img_size = args.img_size
    if img_size is None:
        img_size = 256 if ("EMOTION" in args.dataset_name or "WM" in args.dataset_name) else 64
    x = maybe_resize_spatial(x, img_size)

    model = build_model(args, device)

    if args.baseline == "zeros":
        baseline = torch.zeros_like(x)
    else:
        acc = []
        for i in range(min(256, n)):
            xi, _ = test_ds[i]
            acc.append(xi.unsqueeze(0))
        stack = torch.cat(acc, dim=0).to(device)
        stack = maybe_resize_spatial(stack, img_size)
        baseline = stack.mean(dim=0, keepdim=True)

    # Integrated gradients (float32; avoid AMP for stable grads)
    with torch.enable_grad():
        ig = integrated_gradients(
            model,
            x,
            baseline,
            num_classes=args.num_classes,
            target_class=args.target_class,
            n_steps=args.n_steps,
        )
        gi = gradient_input_saliency(
            model,
            x,
            num_classes=args.num_classes,
            target_class=args.target_class,
        )

    ig_map = to_edge_map(ig)
    gi_map = to_edge_map(gi)
    ig_sym = symmetrize(ig_map)
    gi_sym = symmetrize(gi_map)

    imp_ig = node_importance(ig_sym)
    top_idx, top_scores = top_k_nodes(imp_ig, args.top_k)

    os.makedirs(args.output_dir, exist_ok=True)
    prefix = os.path.join(args.output_dir, args.save_prefix)

    np.savez(
        f"{prefix}_sample{args.sample_index}.npz",
        integrated_gradients_ig_map=ig_map,
        integrated_gradients_sym=ig_sym,
        grad_input_map=gi_map,
        grad_input_sym=gi_sym,
        node_importance_ig=imp_ig,
        top_node_indices=top_idx,
        top_node_scores=top_scores,
        label=np.array(label.cpu() if torch.is_tensor(label) else label),
    )

    # Optional upsample for display / alignment with 264/265 ROI grids
    if args.spatial_resize and args.spatial_resize != ig_sym.shape[0]:
        t = torch.from_numpy(ig_sym).float().view(1, 1, ig_sym.shape[0], ig_sym.shape[1])
        t = F.interpolate(t, size=(args.spatial_resize, args.spatial_resize), mode="bilinear", align_corners=False)
        ig_sym_up = t.squeeze().numpy()
        t2 = torch.from_numpy(gi_sym).float().view(1, 1, gi_sym.shape[0], gi_sym.shape[1])
        t2 = F.interpolate(t2, size=(args.spatial_resize, args.spatial_resize), mode="bilinear", align_corners=False)
        gi_sym_up = t2.squeeze().numpy()
        np.savez(
            f"{prefix}_sample{args.sample_index}_upsampled.npz",
            integrated_gradients_sym=ig_sym_up,
            grad_input_sym=gi_sym_up,
            upsample_size=args.spatial_resize,
        )

    def _plot(mat: np.ndarray, path: str, title: str):
        plt.figure(figsize=(6, 5))
        plt.imshow(mat, cmap="magma")
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.title(title)
        plt.xlabel("region j")
        plt.ylabel("region i")
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()

    _plot(ig_sym, f"{prefix}_sample{args.sample_index}_ig_sym.png", "Integrated Gradients (symmetrized)")
    _plot(gi_sym, f"{prefix}_sample{args.sample_index}_grad_input_sym.png", "Gradient × input (symmetrized)")

    roi_names = None
    if args.roi_names and os.path.isfile(args.roi_names):
        with open(args.roi_names, encoding="utf-8") as f:
            roi_names = [ln.strip() for ln in f if ln.strip()]
        if len(roi_names) != ig_sym.shape[0]:
            print(
                f"Warning: --roi_names has {len(roi_names)} lines but map is H={ig_sym.shape[0]}; "
                "printing indices only."
            )
            roi_names = None

    print("\n--- Top-%d nodes by IG (sum_j |IG_sym[i,j]|) ---" % args.top_k)
    for rank, (idx, sc) in enumerate(zip(top_idx, top_scores), start=1):
        extra = ""
        if roi_names is not None and int(idx) < len(roi_names):
            extra = f"  ({roi_names[int(idx)]})"
        print(f"  {rank}. node index {int(idx):4d}  score={float(sc):.6e}{extra}")

    print(f"\nSaved arrays and figures under: {args.output_dir}")


if __name__ == "__main__":
    main()
