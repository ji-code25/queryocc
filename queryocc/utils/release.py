"""The released-checkpoint format, and how ``--load_ckpt_from`` resolves to a local file.

A training checkpoint is not a good thing to publish: Lightning stores the optimizer state
(~3x the weights), the pickled config tree (``JointDataModule.save_hyperparameters``) and the
callback state with whatever absolute paths the run used, and loading it back needs
``weights_only=False`` — i.e. everyone who downloads it has to unpickle our file with full
trust.

A *release checkpoint* is the published artifact instead: one file per preset, holding the
network weights plus a small provenance header, and nothing else. It contains only tensors and
primitives, so it loads under ``torch.load(..., weights_only=True)``. The ``state_dict`` keeps
the ``net.``-prefixed keys of a training checkpoint, so the ordinary loader
(``PreTrainModel.load_from_checkpoint``) reads it unchanged.

Written by ``tools/release/export_weights.py``, read by ``train.py``.
"""

import logging
import re
from pathlib import Path
from typing import Optional, Union

import torch

logger = logging.getLogger(__name__)

RELEASE_KEY = "queryocc_release"
"""Presence of this key marks a file as a release checkpoint; its value is the format version."""

RELEASE_FORMAT_VERSION = 1

# Path() collapses "hf://org/repo" to "hf:/org/repo", and tyro has already parsed the CLI
# string into a Path by the time we see it — so accept one or two slashes.
_HF_SPEC = re.compile(r"^hf:/{1,2}(?P<rest>.+)$")


def resolve_checkpoint_spec(spec: Union[str, Path], preset: Optional[str] = None) -> Path:
    """Resolve a ``--load_ckpt_from`` value to a local path, downloading from the Hub if needed.

    Local paths pass through untouched. A ``hf://<owner>/<repo>[/<file>]`` spec is fetched with
    ``huggingface_hub.hf_hub_download`` (cached under ``HF_HOME``, so repeat runs are offline).
    With no ``<file>``, the file is ``<preset>.pth``.
    """
    match = _HF_SPEC.match(str(spec))
    if match is None:
        return Path(spec)

    parts = [p for p in match.group("rest").split("/") if p]
    if len(parts) < 2:
        raise ValueError(f"Malformed Hub spec {spec!r}; expected hf://<owner>/<repo>[/<filename>].")

    repo_id = "/".join(parts[:2])
    filename = "/".join(parts[2:])
    if not filename:
        if preset is None:
            raise ValueError(f"Hub spec {spec!r} has no filename and no preset to derive one from.")
        filename = f"{preset}.pth"

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "Resolving an hf:// checkpoint needs the huggingface_hub package (pip install huggingface_hub), "
            "or download the file yourself and pass its path."
        ) from exc

    logger.info("Fetching %s from the Hugging Face Hub repo %s", filename, repo_id)
    try:
        return Path(hf_hub_download(repo_id=repo_id, filename=filename))
    except Exception as exc:
        # The filename is derived from the preset, so the common failure is asking a repo for a
        # preset it does not publish — a bare 404 leaves that unexplained. Matched by class name
        # because huggingface_hub renamed this error (EntryNotFoundError ->
        # RemoteEntryNotFoundError) without giving the two a shared base.
        if "EntryNotFoundError" not in type(exc).__name__:
            raise
        try:
            from huggingface_hub import list_repo_files

            published = ", ".join(f for f in list_repo_files(repo_id) if f.endswith(".pth")) or "no .pth files"
        except Exception:  # pragma: no cover - listing is best-effort context for the error
            published = "unknown (could not list the repo)"
        raise FileNotFoundError(
            f"The repo {repo_id} has no file '{filename}'. It publishes: {published}. "
            f"The filename comes from the selected preset; pass hf://{repo_id}/<filename> to "
            "choose one explicitly."
        ) from exc


def read_release_header(path: Path) -> Optional[dict]:
    """Return the provenance header of a release checkpoint, or None for any other file.

    Reads without materializing the weights (mmap) and without unpickling anything
    (``weights_only=True``) — a training checkpoint simply fails that load and returns None.
    """
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except Exception:
        return None

    if not isinstance(payload, dict) or RELEASE_KEY not in payload:
        return None

    header = {k: v for k, v in payload.items() if k != "state_dict"}
    version = header.get(RELEASE_KEY)
    if version != RELEASE_FORMAT_VERSION:
        logger.warning(
            "Release checkpoint %s declares format version %s; this code speaks version %s.",
            path,
            version,
            RELEASE_FORMAT_VERSION,
        )
    return header


def clean_state_dict(state_dict: dict) -> dict:
    """Reduce a training state dict to the publishable network weights.

    Strips the ``module.`` prefix that DDP-wrapped saves carry, renames the pre-rename ``dos.``
    prefix to ``net.``, and drops everything that is not part of the network (metric buffers and
    the like), which the loader ignores anyway.
    """
    cleaned = {}
    for key, value in state_dict.items():
        name = key[len("module.") :] if key.startswith("module.") else key
        if name.startswith("dos."):
            name = "net." + name[len("dos.") :]
        if not name.startswith("net."):
            continue
        cleaned[name] = value.detach().clone() if torch.is_tensor(value) else value
    return cleaned
