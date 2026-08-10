"""Pre-download every model weight the two generators need.

Run this ONCE on a node with internet access (e.g. the cluster login node)
with the cache directories pointed at persistent storage, then bind those
directories into the container on (possibly offline) compute nodes:

    export TORCH_HOME=/path/to/model_cache/torch
    export HF_HOME=/path/to/model_cache/huggingface
    python -m generate_pseudo_labels.prefetch_weights --sam-dir ckpts

Downloads:
  * GroundingDINO SwinB checkpoint + config    (HF hub -> $HF_HOME)
  * SAM ViT-H checkpoint                       (-> --sam-dir)
  * Metric3D code + ViT-L weights              (torch.hub -> $TORCH_HOME)
"""

import argparse
from pathlib import Path

try:
    from .common import install_mmcv_shim
    from . import constants as C
except ImportError:
    from common import install_mmcv_shim
    import constants as C


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sam-dir", type=Path, default=Path("ckpts"))
    parser.add_argument("--sam-model-type",
                        choices=sorted(C.SAM_CHECKPOINT_URLS),
                        default="vit_h")
    parser.add_argument("--dino-variant", choices=sorted(C.DINO_VARIANTS),
                        default="swinb")
    parser.add_argument("--metric3d-model",
                        default=C.METRIC3D_DEFAULT_MODEL)
    parser.add_argument("--skip", nargs="*", default=[],
                        choices=["dino", "sam", "metric3d"])
    args = parser.parse_args()

    if "dino" not in args.skip:
        from huggingface_hub import hf_hub_download
        repo_id, ckpt, cfg = C.DINO_VARIANTS[args.dino_variant]
        print(f"[dino] {repo_id}/{cfg}")
        hf_hub_download(repo_id=repo_id, filename=cfg)
        print(f"[dino] {repo_id}/{ckpt}")
        hf_hub_download(repo_id=repo_id, filename=ckpt)

    if "sam" not in args.skip:
        import torch
        url = C.SAM_CHECKPOINT_URLS[args.sam_model_type]
        target = args.sam_dir / url.rsplit("/", 1)[-1]
        if target.exists():
            print(f"[sam] {target} already present")
        else:
            print(f"[sam] {url} -> {target}")
            target.parent.mkdir(parents=True, exist_ok=True)
            torch.hub.download_url_to_file(url, str(target))

    if "metric3d" not in args.skip:
        import torch
        print(f"[metric3d] torch.hub {C.METRIC3D_HUB_REPO}:"
              f"{args.metric3d_model}")
        install_mmcv_shim()
        torch.hub.load(C.METRIC3D_HUB_REPO, args.metric3d_model,
                       pretrain=True)
        print("[metric3d] cached (model instantiated on CPU, weights in "
              "$TORCH_HOME)")

    print("done.")


if __name__ == "__main__":
    main()
