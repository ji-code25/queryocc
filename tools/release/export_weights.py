"""Turn a training checkpoint into the file we publish.

    python -m tools.release.export_weights \
        --ckpt /outputs/experiments/<run>/<ts>/checkpoints/<epoch>-<step>.ckpt \
        --preset queryocc-nusc \
        --out release/queryocc-nusc.pth \
        --metrics-json release/queryocc-nusc.metrics.json

Keeps the network weights and a small provenance header; drops the optimizer state, the pickled
config tree and the callback state (which carries the run's absolute output paths). The result
loads under ``weights_only=True`` — verified here, not assumed — so nobody has to unpickle our
file to evaluate the model. Format: queryocc/utils/release.py.

By default the export is verified by building the preset's network on CPU and loading the
exported weights into it strictly, which is the same path ``train.py`` takes.
"""

import argparse
import copy
import hashlib
import json
import logging
import subprocess
from pathlib import Path

import torch

from queryocc.utils.release import RELEASE_FORMAT_VERSION, RELEASE_KEY, clean_state_dict

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("export_weights")


def git_sha() -> str:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent, stderr=subprocess.DEVNULL
        )
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=Path(__file__).resolve().parent, stderr=subprocess.DEVNULL
        )
        return sha.decode().strip() + ("-dirty" if dirty.strip() else "")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_reference_net(preset: str):
    """Build the preset's network exactly as train.py does when loading a checkpoint."""
    from queryocc.config.method_configs import method_configs

    config = copy.deepcopy(method_configs[preset])
    config.model.net.compile = False
    # Without this, building the model would need the gated DINOv3 weights; the checkpoint
    # overwrites every backbone weight anyway. Same setting train.py applies.
    config.model.net.cerveza_encoder.image_encoder_pretrained = False
    config.model.configure_dataset(config.datamodule.dataset)
    return config.model.setup().net


def prune_recomputable_buffers(state_dict: dict, net) -> list:
    """Drop entries the loader rebuilds anyway, when the fresh network reproduces them exactly.

    In practice this is the lifter's constant BEV seed grid (`initial_features`): a persistent
    buffer holding a config-derived constant, 262 MB of zeros in queryocc-nusc — a third of the
    download.

    Two conditions, both required. The key must be one the runtime loader tolerates as missing
    (`TOLERATED_MISSING_KEYS` — otherwise dropping it would make the file unloadable), and a
    freshly built network must already hold a bit-identical tensor (otherwise dropping it would
    lose something the training produced). Many other buffers pass the second test alone — the
    voxel grid, the positional-encoding frequencies — and are deliberately kept: they cost
    kilobytes, and the value of the file staying loadable by the plain loader is worth more.
    """
    from queryocc.model.pre_train import TOLERATED_MISSING_KEYS

    buffers = dict(net.named_buffers())
    dropped = []
    for key in list(state_dict):
        if not any(tolerated in key for tolerated in TOLERATED_MISSING_KEYS):
            continue
        reference = buffers.get(key[len("net.") :])
        if reference is not None and torch.equal(reference, state_dict[key]):
            numel = state_dict.pop(key).numel()
            dropped.append(key)
            logger.info("Dropping %s (%.1f M constant values, rebuilt at load time)", key, numel / 1e6)
    return dropped


def build_payload(ckpt_path: Path, preset: str, metrics: dict, notes: str, net=None) -> dict:
    # weights_only=False: this is our own training checkpoint, which pickles the config
    # dataclasses. Nothing pickled survives into the exported file.
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    raw = checkpoint.get("state_dict", checkpoint)
    state_dict = clean_state_dict(raw)
    if not state_dict:
        raise SystemExit(f"No network weights ('net.*' keys) found in {ckpt_path}.")

    non_network = len(raw) - len(state_dict)
    if non_network:
        logger.info("Dropping %d non-network state-dict entries (metric buffers and the like).", non_network)

    if net is not None:
        prune_recomputable_buffers(state_dict, net)

    num_params = sum(v.numel() for v in state_dict.values() if torch.is_tensor(v))
    logger.info("Exporting %d tensors, %.1f M values", len(state_dict), num_params / 1e6)

    return {
        RELEASE_KEY: RELEASE_FORMAT_VERSION,
        "preset": preset,
        "git_sha": git_sha(),
        "source_checkpoint": ckpt_path.name,
        "num_parameters": num_params,
        "metrics": metrics,
        "notes": notes,
        "state_dict": state_dict,
    }


def verify(out_path: Path, preset: str, net) -> None:
    """Load the exported file the way a user will: weights_only, then into a fresh network.

    Tolerates exactly what the runtime loader tolerates (`PreTrainModel.okay_to_miss`), so a
    pruned buffer that the loader would NOT rebuild fails here rather than in someone's eval.
    """
    payload = torch.load(out_path, map_location="cpu", weights_only=True)
    logger.info("Reloads with weights_only=True (no pickled objects in the file).")

    from queryocc.model.pre_train import TOLERATED_MISSING_KEYS as okay_to_miss

    sdrep = {k[len("net.") :]: v for k, v in payload["state_dict"].items()}
    info = net.load_state_dict(sdrep, strict=False)
    missing = [k for k in info.missing_keys if not any(ok in k for ok in okay_to_miss)]
    if missing or info.unexpected_keys:
        raise SystemExit(f"Verification failed. Missing: {missing[:10]}, unexpected: {list(info.unexpected_keys)[:10]}")
    logger.info("Loads into the '%s' network with no unexplained key mismatches.", preset)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", type=Path, required=True, help="Training .ckpt to export.")
    parser.add_argument("--preset", required=True, help="Preset the checkpoint was trained with.")
    parser.add_argument("--out", type=Path, help="Output file (default: release/<preset>.pth).")
    parser.add_argument("--metrics-json", type=Path, help="JSON file of eval metrics to record in the header.")
    parser.add_argument("--notes", default="", help="Free-text note stored in the header.")
    parser.add_argument("--no-verify", action="store_true", help="Skip the reload check (not recommended).")
    args = parser.parse_args()

    from queryocc.config.method_configs import method_configs

    if args.preset not in method_configs:
        raise SystemExit(f"Unknown preset {args.preset!r}. Known: {', '.join(sorted(method_configs))}")

    out_path = args.out or Path("release") / f"{args.preset}.pth"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    metrics = json.loads(args.metrics_json.read_text()) if args.metrics_json else {}
    # One reference network serves both jobs: deciding what is safely recomputable, and
    # verifying that what is left loads.
    net = None if args.no_verify else build_reference_net(args.preset)
    payload = build_payload(args.ckpt, args.preset, metrics, args.notes, net=net)
    torch.save(payload, out_path)

    if net is not None:
        verify(out_path, args.preset, net)

    size_mb = out_path.stat().st_size / 1e6
    logger.info("Wrote %s (%.1f MB)", out_path, size_mb)
    logger.info("sha256  %s", sha256(out_path))
    logger.info(
        "Upload with:\n  hf upload <owner>/<repo> %s %s --repo-type=model",
        out_path,
        out_path.name,
    )


if __name__ == "__main__":
    main()
