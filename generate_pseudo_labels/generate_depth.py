"""Generate metric depth pseudo-labels with Metric3D (torch.hub).

Standalone port of GaussianFlowOcc's `tools/generate_m3d_nusc.py`, driven by
the infos pkl instead of nuscenes-devkit (both generators share one
enumerator; the devkit is only needed by create_infos.py).

Pipeline per sample (all 6 cameras batched):
  1. Resize each image with scale = min(616/H, 1064/W) and pad to the ViT
     canvas (616x1064) with the ImageNet mean pixel; normalize.
  2. One Metric3D forward -> canonical depth + confidence.
  3. Un-pad, bilinearly upsample to the native resolution.
  4. Metric recovery: depth *= (fx * scale) / 1000  (canonical focal), then
     clamp to [0, 300] m.

NOTE on color order: this port feeds RGB (--color-order rgb, the Metric3D
convention). The original GaussianFlowOcc script fed cv2's BGR straight into
RGB-ordered normalization; use `--color-order bgr` to reproduce those labels
bit-for-bit-ish for A/B comparisons.

Outputs (default npz layout): <save-path>/<scene>/<token>.npz with
`depth` float16 [N_cams,H,W] in constants.CAMERA_ORDER (+ optional
`confidence`); `--cameras` overrides that list (other datasets, or one
resolution group at a time — step 1 batches all cameras of a sample).
`--layout legacy` writes <save-path>/<scene>/<token>.npy float16
[6,900,1600] in LEGACY_DEPTH_ORDER (the order the GaussianFlowOcc trainer
expects for depth, which differs from its mask order!).

Example:
    python -m generate_pseudo_labels.generate_depth \
        --infos-pkl data/bevdetv2-nuscenes_infos_train.pkl \
        --save-path data/pseudo_labels/depth
"""

import argparse

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

try:
    from .common import (add_common_args, atomic_save_npy, atomic_save_npz,
                         install_mmcv_shim, load_infos, resolve_path,
                         select_samples, write_meta)
    from . import constants as C
except ImportError:  # executed as a plain file
    from common import (add_common_args, atomic_save_npy, atomic_save_npz,
                        install_mmcv_shim, load_infos, resolve_path,
                        select_samples, write_meta)
    import constants as C


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_common_args(parser)
    parser.add_argument("--model", default=C.METRIC3D_DEFAULT_MODEL,
                        help="torch.hub entrypoint in yvanyin/metric3d, e.g. "
                             "metric3d_vit_large / metric3d_vit_small")
    parser.add_argument("--color-order", choices=["rgb", "bgr"],
                        default="rgb",
                        help="rgb = Metric3D convention (default); bgr "
                             "reproduces the original GaussianFlowOcc labels")
    parser.add_argument("--save-confidence", action="store_true",
                        help="also store Metric3D's per-pixel confidence "
                             "(discarded by the original pipeline; useful "
                             "as a depth-loss weight)")
    return parser.parse_args()


def input_size_for(model_name):
    family = "convnext" if "convnext" in model_name else "vit"
    return C.METRIC3D_INPUT_SIZES[family]


def preprocess(image_paths, data_root, input_size, color_order, mean, std):
    """Load + resize + pad a list of same-resolution images.

    Returns (batch [N,3,h,w] float32, scale, pad_info, (H, W) native).
    """
    images = [cv2.imread(str(resolve_path(p, data_root)))
              for p in image_paths]
    for p, img in zip(image_paths, images):
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {p}")
    if color_order == "rgb":
        images = [img[:, :, ::-1] for img in images]

    # One resize/pad is computed for the whole batch, so a mixed-resolution
    # sample would be silently squashed into the first camera's aspect ratio
    # (AV2: portrait ring_front_center vs. the landscape rings). Run one
    # --cameras resolution group at a time instead.
    shapes = sorted({img.shape[:2] for img in images})
    if len(shapes) > 1:
        raise ValueError(
            f"All cameras of a sample must share one resolution, got {shapes}. "
            "Pass --cameras to process one resolution group per run.")

    native_h, native_w = images[0].shape[:2]
    scale = min(input_size[0] / native_h, input_size[1] / native_w)
    resize_h, resize_w = int(native_h * scale), int(native_w * scale)
    pad_h, pad_w = input_size[0] - resize_h, input_size[1] - resize_w
    pad_info = [pad_h // 2, pad_h - pad_h // 2,
                pad_w // 2, pad_w - pad_w // 2]

    images = [cv2.resize(img, (resize_w, resize_h),
                         interpolation=cv2.INTER_LINEAR) for img in images]
    images = [cv2.copyMakeBorder(img, *pad_info[:2], *pad_info[2:],
                                 cv2.BORDER_CONSTANT, value=list(mean))
              for img in images]
    batch = torch.stack([
        torch.from_numpy(img.transpose(2, 0, 1).copy()).float()
        for img in images
    ])
    batch = (batch - torch.tensor(mean)[None, :, None, None]) \
        / torch.tensor(std)[None, :, None, None]
    return batch, scale, pad_info, (native_h, native_w)


def unpad_and_upsample(tensor, pad_info, native_hw):
    """[N, h, w] padded-canvas prediction -> [N, H, W] native resolution."""
    tensor = tensor[:,
                    pad_info[0]: tensor.shape[1] - pad_info[1],
                    pad_info[2]: tensor.shape[2] - pad_info[3]]
    return F.interpolate(tensor[:, None], native_hw,
                         mode="bilinear", align_corners=False).squeeze(1)


def main():
    args = parse_args()
    device = torch.device(args.device)
    input_size = input_size_for(args.model)

    infos, metadata = load_infos(args.infos_pkl)
    samples = select_samples(infos, args.start, args.end, args.shard,
                             args.num_shards, args.scene_prefixes)
    print(f"{len(samples)} samples in this shard "
          f"({args.shard + 1}/{args.num_shards})")

    cam_order = tuple(args.cameras) if args.cameras \
        else (C.LEGACY_DEPTH_ORDER if args.layout == "legacy"
              else C.CAMERA_ORDER)
    write_meta(args.save_path, C.DEPTH_META_FILE, {
        "stream": "depth",
        "layout": args.layout,
        "cam_order": list(cam_order),
        "model": args.model,
        "hub_repo": C.METRIC3D_HUB_REPO,
        "input_size": list(input_size),
        "color_order": args.color_order,
        "canonical_focal": C.METRIC3D_CANONICAL_FOCAL,
        "depth_clamp": list(C.METRIC3D_DEPTH_CLAMP),
        "scale": args.scale,
        "infos_metadata": metadata,
    })

    install_mmcv_shim()
    model = torch.hub.load(C.METRIC3D_HUB_REPO, args.model, pretrain=True)
    model = model.to(device).eval()

    with torch.no_grad():
        for sample in tqdm(samples, desc="depth"):
            scene, token = sample["scene_name"], sample["token"]
            suffix = ".npy" if args.layout == "legacy" else ".npz"
            out_file = args.save_path / scene / f"{token}{suffix}"
            if out_file.exists() and not args.overwrite:
                continue

            image_paths = [sample["cams"][cam]["data_path"]
                           for cam in cam_order]
            batch, scale, pad_info, native_hw = preprocess(
                image_paths, args.data_root, input_size, args.color_order,
                C.METRIC3D_MEAN, C.METRIC3D_STD)

            pred_depth, confidence, _ = model.inference(
                {"input": batch.to(device)})

            pred_depth = unpad_and_upsample(
                pred_depth.squeeze(1), pad_info, native_hw)
            # canonical -> metric: depth *= fx_resized / canonical_focal
            fx = torch.tensor(
                [np.asarray(sample["cams"][cam]["cam_intrinsic"])[0, 0]
                 for cam in cam_order],
                dtype=torch.float32, device=device)
            pred_depth = pred_depth * (
                fx[:, None, None] * scale / C.METRIC3D_CANONICAL_FOCAL)
            pred_depth = pred_depth.clamp(*C.METRIC3D_DEPTH_CLAMP)

            if args.scale != 1.0:
                pred_depth = F.interpolate(
                    pred_depth[None], scale_factor=args.scale,
                    mode="bilinear", recompute_scale_factor=False)[0]

            depth_np = pred_depth.cpu().numpy().astype(np.float16)
            if args.layout == "legacy":
                atomic_save_npy(out_file, depth_np)
            else:
                arrays = {
                    "depth": depth_np,
                    "cam_order": np.array(cam_order),
                }
                if args.save_confidence and confidence is not None:
                    conf = unpad_and_upsample(
                        confidence.squeeze(1), pad_info, native_hw)
                    if args.scale != 1.0:
                        conf = F.interpolate(
                            conf[None], scale_factor=args.scale,
                            mode="bilinear", recompute_scale_factor=False)[0]
                    arrays["confidence"] = conf.cpu().numpy().astype(
                        np.float16)
                atomic_save_npz(out_file, compress=args.compress, **arrays)


if __name__ == "__main__":
    main()
