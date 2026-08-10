"""Check a job array before submitting it, instead of hours into it.

Every failure mode that kills a `slurm_generate_*.sh` task at *startup* is
per-shard: each task loads its own first sample and runs its own DINO cache
self-check, so a shard can die on sample 0 while its 15 siblings run for
hours. This reproduces exactly those startup steps for every shard in one
short GPU job:

  * the infos pkl loads and every shard's slice is non-empty
  * the SAM checkpoint and (optionally) the Metric3D weights are reachable
  * the save path is writable and its existing `_meta_*.json` (if any) does
    not conflict with the settings you are about to run
  * the first sample of every shard: images resolve and decode, and the
    cached GroundingDINO path agrees with the stock forward

Exits non-zero if anything fails, so it can gate a submission:

    python -m generate_pseudo_labels.preflight \
        --infos-pkl data/bevdetv2-nuscenes_infos_train.pkl \
        --data-root data/nuscenes --num-shards 16 \
        --sam-checkpoint ckpts/sam_vit_h_4b8939.pth \
    && sbatch --array=0-15%8 scripts/slurm_generate_semantics.sh
"""

import argparse
import os
import time
from pathlib import Path

# torch is imported lazily: --skip-dino is meant to be runnable in a plain
# environment (a login node, a venv with just nuscenes-devkit) where the
# generation container's dependencies are not installed.

try:
    from .common import load_infos, resolve_path, select_samples
    from . import constants as C
except ImportError:  # run as a plain file, not a package
    from common import load_infos, resolve_path, select_samples
    import constants as C


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--infos-pkl", required=True, type=Path)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--num-shards", type=int, default=1,
                        help="the array size you are about to submit")
    parser.add_argument("--save-path", type=Path, default=None,
                        help="check writability and meta compatibility")
    parser.add_argument("--sam-checkpoint", type=Path, default=None,
                        help="checked for existence (not loaded)")
    parser.add_argument("--check-depth", action="store_true",
                        help="also load Metric3D, i.e. check that the "
                             "torch.hub cache works on this node")
    parser.add_argument("--skip-dino", action="store_true",
                        help="only run the cheap checks (no GPU needed)")
    # must match what the real run will use, or the check proves nothing
    parser.add_argument("--vocabulary", choices=sorted(C.VOCABULARIES),
                        default="gaussianflowocc")
    parser.add_argument("--sky-label", type=int, default=C.DEFAULT_SKY_LABEL)
    parser.add_argument("--box-threshold", type=float, default=C.BOX_THRESHOLD)
    parser.add_argument("--dino-variant", choices=sorted(C.DINO_VARIANTS),
                        default="swinb")
    parser.add_argument("--tf32", action="store_true")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def check_save_path(save_path):
    """Writability, and the same meta comparison the generators make."""
    problems = []
    try:
        save_path.mkdir(parents=True, exist_ok=True)
        probe = save_path / ".preflight_write_probe"
        probe.write_text("ok")
        probe.unlink()
    except OSError as exc:
        return [f"save path not writable: {exc}"]
    for meta_file in (C.SEMANTICS_META_FILE, C.DEPTH_META_FILE):
        path = save_path / meta_file
        if path.exists():
            print(f"  note: {path} already exists — the run will refuse to "
                  "start if any generation setting differs from it")
    free_gb = os.statvfs(save_path).f_bavail * os.statvfs(save_path).f_frsize \
        / 2 ** 30
    print(f"  {free_gb:.0f} GB free at {save_path}")
    return problems


def main():
    args = parse_args()
    failures = []

    infos, metadata = load_infos(args.infos_pkl)
    print(f"{args.infos_pkl}: {len(infos)} samples, metadata={metadata}")

    first_samples = []
    for shard in range(args.num_shards):
        samples = select_samples(infos, None, None, shard, args.num_shards)
        if not samples:
            failures.append(f"shard {shard}/{args.num_shards} is empty")
            continue
        first_samples.append((shard, samples[0], len(samples)))
    print(f"{args.num_shards} shards, "
          f"{min(n for _, _, n in first_samples)}-"
          f"{max(n for _, _, n in first_samples)} samples each")

    if args.sam_checkpoint and not args.sam_checkpoint.exists():
        failures.append(f"SAM checkpoint missing: {args.sam_checkpoint}")
    if args.save_path:
        failures.extend(check_save_path(args.save_path))

    if args.check_depth:
        import torch
        try:
            from .common import install_mmcv_shim
        except ImportError:
            from common import install_mmcv_shim
        install_mmcv_shim()
        try:
            torch.hub.load(C.METRIC3D_HUB_REPO, C.METRIC3D_DEFAULT_MODEL,
                           pretrain=True)
            print("metric3d: loaded from the torch.hub cache")
        except Exception as exc:
            failures.append(f"metric3d failed to load: {exc}")

    if not args.skip_dino:
        # Imported late: pulls in torch and groundingdino (which needs its
        # CUDA extension), so the cheap checks stay usable without them.
        import torch
        try:
            from .dino_cached import CachedDinoRunner
            from .generate_semantics import (build_prompt_tables,
                                             load_groundingdino, load_image)
        except ImportError:
            from dino_cached import CachedDinoRunner
            from generate_semantics import (build_prompt_tables,
                                            load_groundingdino, load_image)

        if args.tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        device = torch.device(args.device)
        prompts, _ = build_prompt_tables(C.VOCABULARIES[args.vocabulary],
                                         args.sky_label)
        model = load_groundingdino(args.dino_variant, device)
        runner = CachedDinoRunner(model, prompts, device)

        worst = 0.0
        with torch.inference_mode():
            for shard, sample, _ in first_samples:
                label = f"shard {shard:>3}/{args.num_shards}"
                try:
                    images = torch.stack([
                        load_image(
                            resolve_path(sample["cams"][cam]["data_path"],
                                         args.data_root),
                            C.DINO_SIZE, C.DINO_MAX_SIZE)[1]
                        for cam in C.CAMERA_ORDER
                    ])
                except (FileNotFoundError, OSError) as exc:
                    failures.append(f"{label}: image load failed: {exc}")
                    print(f"{label}: FAIL (image load) {exc}", flush=True)
                    continue
                start = time.perf_counter()
                try:
                    deviation = runner.self_check(
                        images, box_threshold=args.box_threshold)
                except RuntimeError as exc:
                    failures.append(f"{label}: {exc}")
                    print(f"{label}: FAIL {exc}", flush=True)
                    continue
                worst = max(worst, deviation)
                print(f"{label}: OK (dino cache max abs diff "
                      f"{deviation:.2e}, {time.perf_counter() - start:.1f} s)",
                      flush=True)
        print(f"worst DINO cache deviation across shards: {worst:.2e}")

    if failures:
        print(f"\nPREFLIGHT FAILED ({len(failures)} problems):")
        for problem in failures:
            print(f"  - {problem}")
        raise SystemExit(1)
    print("\nPREFLIGHT OK — safe to submit the array")


if __name__ == "__main__":
    main()
