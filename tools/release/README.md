# How the released weights are produced

The checkpoints in the [Model zoo](../../README.md#model-zoo) are not training checkpoints.
They are built from one by `export_weights.py`, which is in this repository so that the claims
the model cards make about those files can be checked rather than taken on trust: run it on any
QueryOcc checkpoint and it produces a file in exactly the published format.

## What a release checkpoint is

The network weights plus a small provenance header (preset, git sha, metrics), and nothing
else. Dropped from the training checkpoint on the way:

| Dropped | Why |
|---|---|
| optimizer + LR scheduler state | ~2/3 of the file, useless to anyone evaluating |
| `datamodule_hyper_parameters` | the pickled config tree — forces `weights_only=False` on the reader |
| callback state | carries the run's absolute output paths and run names |
| `initial_features` (262 MB) | a constant seed grid rebuilt identically at load time |

Two properties follow, and the export **verifies** both rather than asserting them:

- The file reloads under `torch.load(..., weights_only=True)` — nothing in it is unpickled, so
  downloading it does not mean trusting whoever published it.
- Its weights load into a freshly built network for the stated preset with no unexplained key
  mismatches, tolerating only what the runtime loader rebuilds
  (`TOLERATED_MISSING_KEYS` in `queryocc/model/pre_train.py`).

The `initial_features` pruning is guarded by both conditions at once: a key is dropped only if
the loader tolerates it missing *and* a fresh network reproduces the tensor bit-exactly, so
nothing a training run produced can be lost. The format lives in `queryocc/utils/release.py`.

## Reproducing an export

```bash
python -m tools.release.export_weights \
    --ckpt /outputs/experiments/<run>/<timestamp>/checkpoints/<epoch>-<step>.ckpt \
    --preset queryocc-lidar-nusc \
    --out release/queryocc-lidar-nusc.pth \
    --metrics-json release/queryocc-lidar-nusc.metrics.json   # optional, recorded in the header
```

Prints the tensor count, the size and the sha256 — which is published on the model card, so an
export from the same checkpoint can be compared against it byte for byte. Metrics come from the
W&B run's test keys (the table in the top-level README maps paper metric -> logged key).

## Using a released checkpoint

```bash
python queryocc/train.py queryocc-lidar-nusc --no-train --test \
    --load_ckpt_from=hf://QueryOcc/queryocc
```

The `hf://<owner>/<repo>[/<file>]` spec downloads and caches `<preset>.pth` from that repo. The
preset stamped in the file is checked against the selected preset, the weights load strictly — a
mismatch is an error, never a quietly half-initialized model — and the backbone's DINOv3
initialization is skipped, since the checkpoint overwrites it anyway. Evaluation still needs
nuScenes, the Occ3D labels and a CUDA toolkit for the JIT-compiled `dvr` raycaster; see the
top-level README.

## Publishing (maintainers)

```bash
pip install huggingface_hub
hf auth login
hf repos create <owner>/queryocc --type model --private     # once
hf upload <owner>/queryocc release/queryocc-lidar-nusc.pth queryocc-lidar-nusc.pth --repo-type=model
hf upload <owner>/queryocc model_card.md README.md --repo-type=model
```

`model_card_template.md` is the card to fill in; on the Hub the card must be named `README.md`.
Fill in the metrics, filenames and sha256 sums from the export output, then point the Model zoo
table at the Hub URL. One repo serves every preset — the `hf://` spec resolves `<preset>.pth`
inside it — so publishing another arm is one more `hf upload` and one more table row.
