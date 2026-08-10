"""Numerical-equivalence test for the occ3d RayIoU raycast-dedup refactor.

Runs the OLD per-instance `.update()` path and the NEW deduplicated driver path
(`PreTrainMetrics._update_binary_rayiou_dedup` / `_update_semantic_rayiou_dedup`) on the
same random occupancy grids and asserts the accumulated metric states (intersection
/ union) are bit-identical. Requires CUDA + the built `dvr` extension, so run on a
cluster GPU inside the container.
"""

import types

import torch

from queryocc.model.common.metrics import BinaryRayIoU, SemanticRayIoU
from queryocc.model.pre_train_metrics import PreTrainMetrics

DEV = "cuda"
PROBS = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
DEPTHS = [1, 2, 4]
CLASSES = tuple(f"c{i}" for i in range(14))  # 14 -> +noise/+free = 16 grid channels
BS, NVOX = 2, 200 * 200 * 16


def pstr(p):
    return str(p).replace(".", "p")


def make_stub(split, semantic):
    """A minimal object carrying just the attrs the dedup methods touch."""
    stub = types.SimpleNamespace()
    stub.occ3d_ray_metrics_prob_thresholds = [pstr(p) for p in PROBS]
    stub.occ3d_ray_metrics_depth_thresholds = [pstr(d) for d in DEPTHS]
    binary, semantic_m = {}, {}
    for d in DEPTHS:
        for p in PROBS:
            m = BinaryRayIoU(depth_threshold=d, prob_threshold=p).to(DEV)
            setattr(stub, f"occ3d_metrics_binary_rayiou_pthr{pstr(p)}_dthr{pstr(d)}_{split}", m)
            binary[(p, d)] = m
        if semantic:
            m = SemanticRayIoU(depth_threshold=d, class_names=CLASSES).to(DEV)
            setattr(stub, f"occ3d_metrics_semantic_rayiou_dthr{pstr(d)}_{split}", m)
            semantic_m[d] = m
    stub._update_binary_rayiou_dedup = types.MethodType(PreTrainMetrics._update_binary_rayiou_dedup, stub)
    stub._update_semantic_rayiou_dedup = types.MethodType(PreTrainMetrics._update_semantic_rayiou_dedup, stub)
    return stub, binary, semantic_m


def main():
    torch.manual_seed(0)
    binary_probs = torch.rand(BS, NVOX, device=DEV)
    binary_target = (torch.rand(BS, NVOX, device=DEV) > 0.7).bool()
    n_cls = len(CLASSES) + 2
    semantic_preds = torch.randint(0, n_cls, (BS, NVOX), device=DEV)
    labels = torch.randint(0, n_cls, (BS, NVOX), device=DEV)
    origins = torch.zeros(BS, 1, 3, device=DEV)
    tokens = ["tok0", "n/a"]  # exercise the n/a skip path too

    # OLD path: every instance raycasts for itself.
    old, ob, os_ = make_stub("val", semantic=True)
    for m in ob.values():
        m.update(binary_probs, binary_target, origins, tokens)
    for m in os_.values():
        m.update(semantic_preds, labels, origins, tokens)

    # NEW path: dedup driver.
    new, nb, ns = make_stub("val", semantic=True)
    new._update_binary_rayiou_dedup("binary_rayiou", binary_probs, binary_target, origins, tokens, "val")
    new._update_semantic_rayiou_dedup("semantic_rayiou", semantic_preds, labels, origins, tokens, "val")

    ok = True
    for key in ob:
        for state in ("intersection", "union"):
            a = getattr(ob[key], state)
            b = getattr(nb[key], state)
            if not torch.equal(a, b):
                ok = False
                print(f"MISMATCH binary {key} {state}: old={a.tolist()} new={b.tolist()}")
    for d in os_:
        for state in ("intersection", "union"):
            a = getattr(os_[d], state)
            b = getattr(ns[d], state)
            if not torch.equal(a, b):
                ok = False
                print(f"MISMATCH semantic d={d} {state}: max|delta|={(a - b).abs().max().item()}")

    # Sanity: at least one metric actually accumulated something nonzero.
    nonzero = any(getattr(m, "union").sum() > 0 for m in ob.values())
    print(f"accumulated_nonzero={nonzero}")
    print("RESULT:", "IDENTICAL ✅" if ok else "DIVERGED ❌")

    # --- timing: old per-instance path vs new dedup path (the raycast cost in isolation) ---
    import time

    def bench(fn, iters=20, warmup=3):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / iters

    s_old, ob_t, os_t = make_stub("t", semantic=True)
    s_new, nb_t, ns_t = make_stub("t2", semantic=True)

    def old_path():
        for m in list(ob_t.values()) + list(os_t.values()):
            m.reset()
        for m in ob_t.values():
            m.update(binary_probs, binary_target, origins, tokens)
        for m in os_t.values():
            m.update(semantic_preds, labels, origins, tokens)

    def new_path():
        for m in list(nb_t.values()) + list(ns_t.values()):
            m.reset()
        s_new._update_binary_rayiou_dedup("binary_rayiou", binary_probs, binary_target, origins, tokens, "t2")
        s_new._update_semantic_rayiou_dedup("semantic_rayiou", semantic_preds, labels, origins, tokens, "t2")

    t_old, t_new = bench(old_path), bench(new_path)
    print(
        f"TIMING per val-batch (bs={BS}): old={t_old * 1e3:.1f}ms  new={t_new * 1e3:.1f}ms  speedup={t_old / t_new:.2f}x"
    )
    raise SystemExit(0 if (ok and nonzero) else 1)


if __name__ == "__main__":
    main()
