"""Generate open-vocabulary semantic pseudo-labels with GroundingDINO + SAM.

Standalone port of GaussianFlowOcc's `groundedsam/generate_grounded_sam*.py`
("separate" method, the one used for every published label set), built
against the *upstream* pip packages:

    pip install git+https://github.com/IDEA-Research/GroundingDINO.git
    pip install git+https://github.com/facebookresearch/segment-anything.git

Pipeline per sample:
  1. GroundingDINO (SwinB, HF checkpoint) detects boxes for ~42 vocabulary
     prompts at short-side-800 / long-side<=1333. By default this runs via
     dino_cached.CachedDinoRunner: the Swin backbone runs ONCE per camera
     batch and the BERT text encoding of every prompt is computed ONCE per
     run; only the text-conditioned transformer+heads loop over prompts.
     Identical math to the stock forward (self-checked against it on the
     first sample; --no-dino-cache falls back to the public forward).
     Detections whose decoded phrase != its prompt are dropped.
  2. SAM (ViT-H) segments each surviving box at full image resolution.
  3. Fusion: each binary mask is weighted by its DINO box confidence and a
     per-pixel argmax picks the winner; unclaimed pixels get label 0.
  4. The winning prompt is mapped to a nuScenes class id (constants.py).

Throughput knobs (all default to exact original behaviour):
  --prompt-chunk K   run K prompts per transformer pass by tiling the camera
                     batch (batch = 6*K). Same outputs, more memory, better
                     GPU utilization. Try 2-4 on a 40 GB card.
  --amp bf16|fp16    autocast DINO+SAM. NOT bit-exact — argmax labels are
                     virtually identical, but boundary pixels can flip.
  --tf32             allow TF32 matmuls (Ampere+). Near-exact, ~free speed.
Image loading/preprocessing for the next sample is prefetched on a
background thread while the GPU works.

Outputs (default npz layout): <save-path>/<scene>/<token>.npz with
`labels` int8 [N_cams,H,W] in constants.CAMERA_ORDER (+ optional `scores`);
`--cameras` overrides that list (other datasets, or one resolution group at a
time — all cameras of a sample are batched, so they must share a resolution).
`--layout legacy` reproduces the exact GaussianFlowOcc format instead:
<save-path>/<scene>/<token>.npy int8 [6,900,1600] in LEGACY_MASK_ORDER.

Example (single GPU, one shard of 6):
    python -m generate_pseudo_labels.generate_semantics \
        --infos-pkl data/bevdetv2-nuscenes_infos_train.pkl \
        --save-path data/pseudo_labels/semantics \
        --shard 0 --num-shards 6
"""

import argparse
import contextlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from huggingface_hub import hf_hub_download

import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.inference import preprocess_caption
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from segment_anything import SamPredictor, sam_model_registry

try:
    from .common import (add_common_args, atomic_save_npy, atomic_save_npz,
                         load_infos, resolve_path, select_samples, write_meta)
    from .dino_cached import CachedDinoRunner
    from . import constants as C
except ImportError:  # executed as a plain file
    from common import (add_common_args, atomic_save_npy, atomic_save_npz,
                        load_infos, resolve_path, select_samples, write_meta)
    from dino_cached import CachedDinoRunner
    import constants as C


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_common_args(parser)
    parser.add_argument("--vocabulary", choices=sorted(C.VOCABULARIES),
                        default="gaussianflowocc")
    parser.add_argument("--sky-label", type=int, default=C.DEFAULT_SKY_LABEL,
                        help="label id written for sky pixels "
                             f"(default {C.DEFAULT_SKY_LABEL}; use 0 to "
                             "reproduce the original sky=ignore labels)")
    parser.add_argument("--box-threshold", type=float,
                        default=C.BOX_THRESHOLD)
    parser.add_argument("--text-threshold", type=float,
                        default=C.TEXT_THRESHOLD)
    parser.add_argument("--dino-variant", choices=sorted(C.DINO_VARIANTS),
                        default="swinb")
    parser.add_argument("--dino-size", type=int, default=C.DINO_SIZE,
                        help="short-side resize for the DINO input")
    parser.add_argument("--dino-max-size", type=int, default=C.DINO_MAX_SIZE)
    parser.add_argument("--sam-model-type",
                        choices=sorted(C.SAM_CHECKPOINT_URLS),
                        default="vit_h")
    parser.add_argument("--sam-checkpoint", type=Path,
                        default=Path("ckpts/sam_vit_h_4b8939.pth"))
    parser.add_argument("--download-sam", action="store_true",
                        help="download the SAM checkpoint if missing")
    parser.add_argument("--save-scores", action="store_true",
                        help="also store the winning per-pixel confidence")
    parser.add_argument("--save-phrase-scores", action="store_true",
                        help="store per-prompt score maps (one compressed "
                             "npz per camera; LARGE — meant for label-fusion "
                             "experiments on subsets, not full runs)")
    # throughput knobs
    parser.add_argument("--prompt-chunk", type=int, default=1,
                        help="prompts per DINO forward (batch = num_cams * "
                             "chunk). Same outputs; raise until you run out "
                             "of GPU memory")
    parser.add_argument("--amp", choices=["none", "bf16", "fp16"],
                        default="none",
                        help="autocast SAM (the ViT-H image encoder, roughly "
                             "half the per-sample cost). Faster but not "
                             "bit-exact")
    parser.add_argument("--amp-dino", choices=["none", "bf16", "fp16"],
                        default="none",
                        help="autocast GroundingDINO. OFF by default: its "
                             "CUDA deformable-attention kernel is compiled "
                             "with AT_DISPATCH_FLOATING_TYPES, i.e. fp32/"
                             "fp64 only, so autocasting raises "
                             "'ms_deform_attn_forward_cuda not implemented "
                             "for BFloat16'. Use --tf32 for DINO speed "
                             "instead")
    parser.add_argument("--tf32", action="store_true",
                        help="allow TF32 matmuls (Ampere+); near-exact")
    parser.add_argument("--no-dino-cache", action="store_true",
                        help="disable backbone/text caching and use only "
                             "GroundingDINO's public forward (slower; "
                             "fallback if the installed groundingdino "
                             "version changed internals)")
    parser.add_argument("--skip-cache-check", action="store_true",
                        help="skip the startup self-check that compares the "
                             "cached DINO path against the stock forward")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Vocabulary bookkeeping
# ---------------------------------------------------------------------------
def build_prompt_tables(vocabulary, sky_label):
    """Returns (prompts, prompt_to_label) where prompts are the exact strings
    sent to (and returned by) GroundingDINO: lower-case, hyphens->spaces."""
    prompts, prompt_to_label = [], {}
    for class_name, class_prompts in vocabulary.items():
        label = sky_label if class_name == "sky" \
            else C.CLASS_NAME_TO_LABEL[class_name]
        for prompt in class_prompts:
            spaced = prompt.replace("-", " ").lower()
            prompts.append(spaced)
            prompt_to_label[spaced] = label
    return prompts, prompt_to_label


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
def load_groundingdino(variant, device):
    repo_id, ckpt_name, cfg_name = C.DINO_VARIANTS[variant]
    cfg_file = hf_hub_download(repo_id=repo_id, filename=cfg_name)
    args = SLConfig.fromfile(cfg_file)
    args.device = str(device)
    model = build_model(args)
    ckpt_file = hf_hub_download(repo_id=repo_id, filename=ckpt_name)
    checkpoint = torch.load(ckpt_file, map_location="cpu")
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    return model.eval().to(device)


def load_sam(model_type, checkpoint, download, device):
    checkpoint = Path(checkpoint)
    if not checkpoint.exists():
        if not download:
            raise FileNotFoundError(
                f"SAM checkpoint not found: {checkpoint}. Download it from "
                f"{C.SAM_CHECKPOINT_URLS[model_type]} or pass --download-sam.")
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.hub.download_url_to_file(
            C.SAM_CHECKPOINT_URLS[model_type], str(checkpoint))
    sam = sam_model_registry[model_type](checkpoint=str(checkpoint))
    return SamPredictor(sam.to(device).eval())


# ---------------------------------------------------------------------------
# Per-sample pipeline
# ---------------------------------------------------------------------------
def load_image(path, dino_size, dino_max_size):
    """RGB numpy image at native resolution + normalized DINO input."""
    transform = T.Compose([
        T.RandomResize([dino_size], max_size=dino_max_size),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    source = Image.open(path).convert("RGB")
    image_np = np.asarray(source)
    image_t, _ = transform(source, None)
    return image_np, image_t


def detect_boxes_batched(model, images, prompts, box_threshold,
                         text_threshold, device, prompt_chunk, amp_ctx):
    """DINO detections for all cameras of one sample, batched.

    `images` is [N_cams, 3, h, w] (all cameras share one resolution). Each
    forward processes `prompt_chunk` prompts with the camera batch tiled
    along the batch dim. This is mathematically identical to the original
    one-image-one-prompt loop (the model has no batch-dependent layers),
    just with far better GPU utilization: 42 forwards of batch 6*K per
    sample instead of 252 forwards of batch 1.

    Applies the exact phrase==prompt filter. Returns a list of per-camera
    (boxes [M,4] cxcywh on cpu, scores [M] on cpu, phrases).
    """
    ncams = images.shape[0]
    boxes_per_cam = [[] for _ in range(ncams)]
    scores_per_cam = [[] for _ in range(ncams)]
    phrases_per_cam = [[] for _ in range(ncams)]
    images = images.to(device)

    for start in range(0, len(prompts), prompt_chunk):
        chunk = prompts[start:start + prompt_chunk]
        captions = [preprocess_caption(p) for p in chunk]
        batch = images.repeat(len(chunk), 1, 1, 1) if len(chunk) > 1 \
            else images
        # batch element j*ncams + c pairs caption j with camera c
        batch_captions = [cap for cap in captions for _ in range(ncams)]
        with amp_ctx():
            outputs = model(batch, captions=batch_captions)
        pred_logits = outputs["pred_logits"].float().sigmoid()  # [B,nq,256]
        pred_boxes = outputs["pred_boxes"].float()              # [B,nq,4]

        for j, (prompt, caption) in enumerate(zip(chunk, captions)):
            tokenized = model.tokenizer(caption)
            for cam in range(ncams):
                logits_i = pred_logits[j * ncams + cam]
                boxes_i = pred_boxes[j * ncams + cam]
                keep = logits_i.max(dim=1)[0] > box_threshold
                for logit, box in zip(logits_i[keep], boxes_i[keep]):
                    phrase = get_phrases_from_posmap(
                        logit > text_threshold, tokenized,
                        model.tokenizer).replace(".", "")
                    if phrase == prompt:  # exact-match filter
                        boxes_per_cam[cam].append(box.cpu())
                        scores_per_cam[cam].append(logit.max().cpu())
                        phrases_per_cam[cam].append(phrase)

    results = []
    for cam in range(ncams):
        if boxes_per_cam[cam]:
            results.append((torch.stack(boxes_per_cam[cam]),
                            torch.stack(scores_per_cam[cam]),
                            phrases_per_cam[cam]))
        else:
            results.append((torch.zeros((0, 4)), torch.zeros((0,)), []))
    return results


def segment_boxes(sam_predictor, image_np, boxes_cxcywh, device, amp_ctx):
    """SAM at full resolution -> bool masks [N, H, W]."""
    h, w, _ = image_np.shape
    with amp_ctx():
        sam_predictor.set_image(image_np)
        boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes_cxcywh) \
            * torch.tensor([w, h, w, h], dtype=torch.float32)
        transformed = sam_predictor.transform.apply_boxes_torch(
            boxes_xyxy, image_np.shape[:2]).to(device)
        logits_masks, _, _ = sam_predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed,
            return_logits=True,
            multimask_output=False,
        )
    return (logits_masks.float()
            > sam_predictor.model.mask_threshold).squeeze(1)


def fuse_masks(masks, phrases, logits, prompt_to_label, device):
    """Confidence-weighted argmax fusion (GaussianFlowOcc behaviour).

    Returns (labels int64 [H,W], scores float32 [H,W]).
    """
    h, w = masks.shape[-2:]
    weighted = masks.float() * logits[:, None, None].to(device)
    weighted = torch.cat(
        [torch.zeros((1, h, w), device=device), weighted], dim=0)
    scores, idx = weighted.max(dim=0)
    label_lut = torch.tensor(
        [0] + [prompt_to_label[p] for p in phrases],
        dtype=torch.int64, device=device)
    return label_lut[idx], scores


def phrase_score_maps(masks, phrases, logits, prompts, device):
    """Per-prompt max-confidence maps [P, H, W] for offline re-fusion."""
    h, w = masks.shape[-2:]
    maps = torch.zeros((len(prompts), h, w), device=device)
    for mask, phrase, logit in zip(masks, phrases, logits):
        i = prompts.index(phrase)
        maps[i] = torch.maximum(maps[i], mask.float() * logit.to(device))
    return maps


def downscale(tensor, scale, mode):
    """Downscale a [H, W] or [C, H, W] tensor. Labels use nearest, scores
    bilinear."""
    if scale == 1.0:
        return tensor
    squeeze = tensor.ndim == 2
    if squeeze:
        tensor = tensor[None]
    out = F.interpolate(
        tensor[None].float(), scale_factor=scale, mode=mode,
        recompute_scale_factor=False)[0]
    return out[0] if squeeze else out


# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    device = torch.device(args.device)
    vocabulary = C.VOCABULARIES[args.vocabulary]
    prompts, prompt_to_label = build_prompt_tables(vocabulary, args.sky_label)

    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    def make_amp_ctx(name):
        dtype = {"none": None, "bf16": torch.bfloat16,
                 "fp16": torch.float16}[name]
        if dtype is None:
            return contextlib.nullcontext
        return lambda: torch.autocast(device.type, dtype=dtype)

    # SAM and DINO are autocast independently: DINO's CUDA kernel only
    # supports fp32 (see --amp-dino).
    sam_amp_ctx = make_amp_ctx(args.amp)
    dino_amp_ctx = make_amp_ctx(args.amp_dino)

    infos, metadata = load_infos(args.infos_pkl)
    samples = select_samples(infos, args.start, args.end, args.shard,
                             args.num_shards, args.scene_prefixes)
    print(f"{len(samples)} samples in this shard "
          f"({args.shard + 1}/{args.num_shards})")

    cam_order = tuple(args.cameras) if args.cameras \
        else (C.LEGACY_MASK_ORDER if args.layout == "legacy"
              else C.CAMERA_ORDER)
    suffix = ".npy" if args.layout == "legacy" else ".npz"
    write_meta(args.save_path, C.SEMANTICS_META_FILE, {
        "stream": "semantics",
        "layout": args.layout,
        "cam_order": list(cam_order),
        "vocabulary_name": args.vocabulary,
        "vocabulary": vocabulary,
        "prompts": prompts,
        "sky_label": args.sky_label,
        "class_names": C.CLASS_NAMES,
        "box_threshold": args.box_threshold,
        "text_threshold": args.text_threshold,
        "dino_variant": args.dino_variant,
        "dino_size": args.dino_size,
        "dino_max_size": args.dino_max_size,
        "sam_model_type": args.sam_model_type,
        "scale": args.scale,
        "amp": args.amp,
        "amp_dino": args.amp_dino,
        "infos_metadata": metadata,
    })

    dino = load_groundingdino(args.dino_variant, device)
    sam_predictor = load_sam(args.sam_model_type, args.sam_checkpoint,
                             args.download_sam, device)
    dino_runner = None if args.no_dino_cache \
        else CachedDinoRunner(dino, prompts, device)
    cache_checked = args.no_dino_cache or args.skip_cache_check

    # Filter to remaining work up front so the prefetcher never touches
    # samples that will be skipped.
    def out_file(sample):
        return args.save_path / sample["scene_name"] \
            / f"{sample['token']}{suffix}"

    todo = [s for s in samples if args.overwrite or not out_file(s).exists()]
    print(f"{len(todo)} samples to generate ({len(samples) - len(todo)} "
          "already done)")

    def load_sample_images(sample):
        loaded = []
        for cam in cam_order:
            path = resolve_path(sample["cams"][cam]["data_path"],
                                args.data_root)
            loaded.append(load_image(path, args.dino_size,
                                     args.dino_max_size))
        return loaded

    executor = ThreadPoolExecutor(max_workers=1)
    next_future = None

    with torch.inference_mode():
        for i, sample in enumerate(tqdm(todo, desc="semantics")):
            scene, token = sample["scene_name"], sample["token"]
            loaded = next_future.result() if next_future is not None \
                else load_sample_images(sample)
            # prefetch the next sample's images while the GPU works
            next_future = executor.submit(load_sample_images, todo[i + 1]) \
                if i + 1 < len(todo) else None

            images_np = [img_np for img_np, _ in loaded]
            images_t = torch.stack([img_t for _, img_t in loaded])
            if dino_runner is not None:
                if not cache_checked:
                    deviation = dino_runner.self_check(
                        images_t, box_threshold=args.box_threshold)
                    print(f"DINO cache self-check passed "
                          f"(max abs diff {deviation:.2e})")
                    cache_checked = True
                detections = dino_runner.detect(
                    images_t, args.box_threshold, args.text_threshold,
                    args.prompt_chunk, dino_amp_ctx)
            else:
                detections = detect_boxes_batched(
                    dino, images_t, prompts, args.box_threshold,
                    args.text_threshold, device, args.prompt_chunk,
                    dino_amp_ctx)

            labels_per_cam, scores_per_cam = [], []
            for cam, image_np, (boxes, box_scores, phrases) in zip(
                    cam_order, images_np, detections):
                h, w, _ = image_np.shape
                if boxes.shape[0] == 0:
                    labels = torch.zeros((h, w), dtype=torch.int64,
                                         device=device)
                    scores = torch.zeros((h, w), device=device)
                else:
                    masks = segment_boxes(sam_predictor, image_np, boxes,
                                          device, sam_amp_ctx)
                    labels, scores = fuse_masks(
                        masks, phrases, box_scores, prompt_to_label, device)
                    if args.save_phrase_scores:
                        maps = phrase_score_maps(masks, phrases, box_scores,
                                                 prompts, device)
                        maps = downscale(maps, args.scale, "bilinear")
                        atomic_save_npz(
                            args.save_path / scene /
                            f"{token}_{cam}_phrase_scores.npz",
                            compress=True,
                            phrase_names=np.array(prompts),
                            phrase_labels=np.array(
                                [prompt_to_label[p] for p in prompts],
                                dtype=np.int8),
                            phrase_scores=maps.cpu().numpy().astype(
                                np.float16))

                labels = downscale(labels, args.scale, "nearest").long()
                scores = downscale(scores, args.scale, "bilinear")
                labels_per_cam.append(
                    labels.cpu().numpy().astype(np.int8))
                scores_per_cam.append(
                    scores.cpu().numpy().astype(np.float16))

            labels_stack = np.stack(labels_per_cam)
            if args.layout == "legacy":
                atomic_save_npy(out_file(sample), labels_stack)
            else:
                arrays = {
                    "labels": labels_stack,
                    "cam_order": np.array(cam_order),
                }
                if args.save_scores:
                    arrays["scores"] = np.stack(scores_per_cam)
                atomic_save_npz(out_file(sample), compress=args.compress,
                                **arrays)

    executor.shutdown(wait=False)


if __name__ == "__main__":
    main()
