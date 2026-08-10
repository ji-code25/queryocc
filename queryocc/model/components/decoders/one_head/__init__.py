"""OneHeadDecoder — the primary decode head (occupancy + semantic segmentation, real and
pseudo-labelled) behind NetConfig.one_head.

Package layout:
- config.py  — OneHeadDecoderConfig + the head-spec table (get_active_heads/sources)
- decoder.py — the OneHeadDecoder class: construction, query assembly, decode, forward
- losses.py  — QueryLossesMixin: the per-query loss methods + the empty-pseudo grad anchor

Mixins hold methods only — all parameters/buffers are created in decoder.py.__init__, so
checkpoint keys are independent of this file layout.
"""

from queryocc.model.components.decoders.one_head.config import OneHeadDecoderConfig
from queryocc.model.components.decoders.one_head.decoder import OneHeadDecoder

__all__ = ["OneHeadDecoder", "OneHeadDecoderConfig"]
