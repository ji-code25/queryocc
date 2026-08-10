"""Compute-cluster environment helpers."""

import resource


def set_ulimit() -> None:
    """Raise the soft open-file limit: many dataloader workers each hold dataset files,
    sockets and shared-memory segments open, and the default soft limit can be low."""
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    target = 16384 if hard == resource.RLIM_INFINITY else min(16384, hard)
    if soft < target:
        resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
