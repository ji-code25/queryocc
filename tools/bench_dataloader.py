"""Time raw dataset sample construction, isolating the dataloader from the GPU.

Answers one question: is a preset's training step slow because of compute, or because
building a sample takes too long to keep the GPU fed?

    python -m tools.bench_dataloader --preset queryocc-nusc --num_samples 20

Reports seconds per sample and, with --trace_opens, how many file opens each sample costs
(a shared parallel filesystem is latency-bound, so open count matters as much as bytes).
"""

import argparse
import builtins
import io
import time
from collections import Counter


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", required=True)
    parser.add_argument("--version", default="full", help="Dataset version, as scripts/train.sh passes it.")
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--trace_opens", type=str, choices=["true", "false"], default="true")
    return parser.parse_args()


def main():
    args = parse_args()
    from queryocc.config.method_configs import method_configs

    config = method_configs[args.preset]
    # Must match the training run: the version is part of the infos-cache hash, so getting it
    # wrong misses the cache and rebuilds the index from the devkit (which needs map masks the
    # training binds do not provide). Set it on the dataset too -- JointDataModuleConfig only
    # propagates version to dataset in __post_init__, i.e. at construction, and we are past it.
    config.datamodule.version = args.version
    config.datamodule.dataset.version = args.version
    config.model.configure_dataset(config.datamodule.dataset)

    datamodule = config.datamodule.setup()
    datamodule.setup("fit")
    dataset = datamodule.train_dataloader().dataset
    print(f"{args.preset}: {len(dataset)} training samples")

    opens = Counter()
    real_open = builtins.open
    tracing = {"on": False}

    def counting_open(file, *rest, **kwargs):
        if tracing["on"]:
            name = str(file)
            key = "npz" if name.endswith(".npz") else "lidar" if ".pcd.bin" in name else "other"
            opens[key] += 1
        return real_open(file, *rest, **kwargs)

    if args.trace_opens == "true":
        builtins.open = counting_open
        io.open = counting_open

    for i in range(args.warmup):
        dataset[i]

    tracing["on"] = True
    start = time.perf_counter()
    for i in range(args.warmup, args.warmup + args.num_samples):
        dataset[i]
    elapsed = time.perf_counter() - start
    tracing["on"] = False
    builtins.open = real_open

    per_sample = elapsed / args.num_samples
    print(f"{args.preset}: {per_sample:.3f} s/sample ({args.num_samples} samples, {elapsed:.1f} s total)")
    if args.trace_opens == "true":
        total = sum(opens.values())
        detail = ", ".join(f"{k}={v / args.num_samples:.1f}" for k, v in sorted(opens.items()))
        print(f"{args.preset}: {total / args.num_samples:.1f} file opens/sample ({detail})")


if __name__ == "__main__":
    main()
