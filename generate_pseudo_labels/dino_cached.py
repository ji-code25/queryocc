"""Cached GroundingDINO inference: backbone once per image, text once per run.

GroundingDINO's `forward()` recomputes two prompt- or image-independent
things on every call:

  * the Swin image backbone + input projections — identical for all ~42
    prompts of an image;
  * the BERT text encoding of the caption — identical for all ~200k images
    of a run (only its *fused* representation inside the transformer depends
    on the image).

`CachedDinoRunner` splits the forward accordingly: per sample it runs the
backbone once for the camera batch, then loops the (text-conditioned)
transformer + heads over prompt chunks using text features encoded once at
startup. The math is identical to the stock forward:

  * all prompts are tokenized together with `padding="longest"` — BERT's
    block-diagonal attention mask and the downstream `text_token_mask`
    exclude padding, so per-prompt results equal unpadded single-caption
    encoding (this is GroundingDINO's own batched-captions path);
  * the transformer mutates `text_dict["encoded_text"]` in place with the
    image-fused text memory, and `class_embed` must see that update — we
    build a fresh text_dict per call from the cache, exactly like the stock
    forward builds one per call;
  * `pred_boxes`/`pred_logits` replicate the last-decoder-layer head
    computation (`bbox_embed[-1](hs[-1]) + inverse_sigmoid(reference[-2])`,
    `class_embed[-1](hs[-1], text_dict)`).

Because this reaches into `model.*` internals (stable across GroundingDINO
history, but not API), `self_check()` compares the cached path against the
stock `model(images, captions=...)` forward and raises on divergence — run
it once at startup (generate_semantics.py does this automatically on the
first sample). It compares the *detections* rather than the raw per-query
tensors; see `self_check` for why the latter are not elementwise comparable.
"""

import contextlib

import torch
import torch.nn.functional as F

from groundingdino.models.GroundingDINO.bertwarper import (
    generate_masks_with_special_tokens_and_transfer_map,
)
from groundingdino.util.inference import preprocess_caption
from groundingdino.util.misc import (NestedTensor, inverse_sigmoid,
                                     nested_tensor_from_tensor_list)
from groundingdino.util.utils import get_phrases_from_posmap

_REQUIRED_ATTRS = (
    "tokenizer", "specical_tokens", "sub_sentence_present", "bert",
    "feat_map", "max_text_len", "backbone", "input_proj",
    "num_feature_levels", "transformer", "bbox_embed", "class_embed",
)


class CachedDinoRunner:

    def __init__(self, model, prompts, device):
        missing = [a for a in _REQUIRED_ATTRS if not hasattr(model, a)]
        if missing:
            raise AttributeError(
                f"GroundingDINO model lacks attributes {missing} — the "
                "installed groundingdino version diverged from the layout "
                "this cache was written against. Rerun with "
                "--no-dino-cache (slower, uses only the public forward).")
        self.model = model
        self.device = device
        self.prompts = list(prompts)
        self.captions = [preprocess_caption(p) for p in self.prompts]
        # plain per-prompt tokenization for phrase decoding, exactly as
        # upstream predict() does it
        self.tokenized_per_prompt = [model.tokenizer(c) for c in self.captions]
        self.text_cache = self._encode_text(self.captions)

    # ------------------------------------------------------------------
    # text: once per run
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def _encode_text(self, captions):
        """Replicates the text branch of GroundingDINO.forward for all
        prompts in one batch. Returns tensors [P, L, ...] indexed by
        prompt."""
        model = self.model
        tokenized = model.tokenizer(
            captions, padding="longest", return_tensors="pt").to(self.device)
        (text_self_attention_masks, position_ids, _
         ) = generate_masks_with_special_tokens_and_transfer_map(
            tokenized, model.specical_tokens, model.tokenizer)

        if text_self_attention_masks.shape[1] > model.max_text_len:
            text_self_attention_masks = text_self_attention_masks[
                :, : model.max_text_len, : model.max_text_len]
            position_ids = position_ids[:, : model.max_text_len]
            tokenized["input_ids"] = \
                tokenized["input_ids"][:, : model.max_text_len]
            tokenized["attention_mask"] = \
                tokenized["attention_mask"][:, : model.max_text_len]
            tokenized["token_type_ids"] = \
                tokenized["token_type_ids"][:, : model.max_text_len]

        if model.sub_sentence_present:
            tokenized_for_encoder = {
                k: v for k, v in tokenized.items() if k != "attention_mask"}
            tokenized_for_encoder["attention_mask"] = text_self_attention_masks
            tokenized_for_encoder["position_ids"] = position_ids
        else:
            tokenized_for_encoder = tokenized

        bert_output = model.bert(**tokenized_for_encoder)
        encoded_text = model.feat_map(bert_output["last_hidden_state"])
        text_token_mask = tokenized.attention_mask.bool()

        if encoded_text.shape[1] > model.max_text_len:
            encoded_text = encoded_text[:, : model.max_text_len, :]
            text_token_mask = text_token_mask[:, : model.max_text_len]
            position_ids = position_ids[:, : model.max_text_len]
            text_self_attention_masks = text_self_attention_masks[
                :, : model.max_text_len, : model.max_text_len]

        return {
            "encoded_text": encoded_text,
            "text_token_mask": text_token_mask,
            "position_ids": position_ids,
            "text_self_attention_masks": text_self_attention_masks,
        }

    # ------------------------------------------------------------------
    # image: once per sample
    # ------------------------------------------------------------------
    def image_features(self, images):
        """Backbone + input projections for a camera batch [N, 3, H, W].
        Replicates the image branch of GroundingDINO.forward."""
        model = self.model
        samples = nested_tensor_from_tensor_list(images)
        features, poss = model.backbone(samples)

        srcs, masks = [], []
        for level, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(model.input_proj[level](src))
            masks.append(mask)
        if model.num_feature_levels > len(srcs):
            n_base = len(srcs)
            for level in range(n_base, model.num_feature_levels):
                if level == n_base:
                    src = model.input_proj[level](features[-1].tensors)
                else:
                    src = model.input_proj[level](srcs[-1])
                mask = F.interpolate(
                    samples.mask[None].float(),
                    size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = model.backbone[1](
                    NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                poss.append(pos_l)
        return srcs, masks, poss

    # ------------------------------------------------------------------
    # transformer + heads: per prompt chunk
    # ------------------------------------------------------------------
    def decode_prompts(self, srcs, masks, poss, prompt_indices):
        """Text-conditioned transformer + last-layer heads for a chunk of
        prompts over the camera batch. Batch element j*ncams + c pairs
        prompt_indices[j] with camera c. Returns (pred_logits, pred_boxes),
        shapes [K*ncams, nq, max_text_len] / [K*ncams, nq, 4]."""
        model = self.model
        ncams = srcs[0].shape[0]
        k = len(prompt_indices)
        if k > 1:
            srcs = [s.repeat(k, 1, 1, 1) for s in srcs]
            masks = [m.repeat(k, 1, 1) for m in masks]
            poss = [p.repeat(k, 1, 1, 1) for p in poss]

        idx = torch.as_tensor(prompt_indices, device=self.device)
        # fresh dict per call: the transformer overwrites "encoded_text"
        # with the image-fused text memory (repeat_interleave copies, so
        # the cache itself is never touched)
        text_dict = {
            key: value[idx].repeat_interleave(ncams, dim=0)
            for key, value in self.text_cache.items()
        }

        hs, reference, _, _, _ = model.transformer(
            srcs, masks, None, poss, None, None, text_dict)

        # last-decoder-layer outputs, exactly as in forward():
        # zip(reference[:-1], bbox_embed, hs)[-1] pairs reference[-2] with
        # bbox_embed[-1] and hs[-1]
        pred_boxes = (model.bbox_embed[-1](hs[-1])
                      + inverse_sigmoid(reference[-2])).sigmoid()
        pred_logits = model.class_embed[-1](hs[-1], text_dict)
        return pred_logits, pred_boxes

    # ------------------------------------------------------------------
    def detect(self, images, box_threshold, text_threshold, prompt_chunk=1,
               amp_ctx=contextlib.nullcontext):
        """Full detection for one sample: backbone once, prompts chunked.

        Applies the exact phrase==prompt filter. Returns a list of
        per-camera (boxes [M,4] cxcywh cpu, scores [M] cpu, phrases).
        """
        ncams = images.shape[0]
        boxes_per_cam = [[] for _ in range(ncams)]
        scores_per_cam = [[] for _ in range(ncams)]
        phrases_per_cam = [[] for _ in range(ncams)]

        with amp_ctx():
            feats = self.image_features(images.to(self.device))

        for start in range(0, len(self.prompts), prompt_chunk):
            indices = list(range(
                start, min(start + prompt_chunk, len(self.prompts))))
            with amp_ctx():
                pred_logits, pred_boxes = self.decode_prompts(*feats, indices)
            pred_logits = pred_logits.float().sigmoid()
            pred_boxes = pred_boxes.float()

            for j, prompt_idx in enumerate(indices):
                prompt = self.prompts[prompt_idx]
                tokenized = self.tokenized_per_prompt[prompt_idx]
                for cam in range(ncams):
                    logits_i = pred_logits[j * ncams + cam]
                    boxes_i = pred_boxes[j * ncams + cam]
                    keep = logits_i.max(dim=1)[0] > box_threshold
                    for logit, box in zip(logits_i[keep], boxes_i[keep]):
                        phrase = get_phrases_from_posmap(
                            logit > text_threshold, tokenized,
                            self.model.tokenizer).replace(".", "")
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

    # ------------------------------------------------------------------
    @staticmethod
    def _detections(scores, boxes, threshold):
        """(scores, boxes) of one image's queries above `threshold`, sorted
        by score — the permutation-invariant view of a DINO output that the
        pipeline actually consumes."""
        per_query = scores.max(dim=-1).values
        keep = per_query > threshold
        order = per_query[keep].argsort(descending=True)
        return per_query[keep][order], boxes[keep][order]

    def self_check(self, images, box_threshold=0.2, score_tolerance=5e-2,
                   box_tolerance=2e-2, margin=0.05):
        """Compare the cached path against the stock forward on real images
        (fp32, first and last prompt) — the guard against upstream
        groundingdino changing its internals.

        Compares *detections*, not the raw per-query tensors. GroundingDINO
        selects its decoder queries by ranking encoder proposals, so
        float-level differences (~1e-5, from batching the text encoding)
        reorder near-tied queries in the low-score tail: the tensors are not
        elementwise comparable even when every detection above threshold is
        identical. (Raw pred_logits also carry -inf in masked text columns,
        which makes plain differences NaN.) So we check what the pipeline
        consumes: the number of detections, their scores and their boxes —
        and the full sorted score distribution, which catches any structural
        change even on images without detections.

        Detections are only compared above `box_threshold + margin`: a query
        sitting exactly on the threshold flips sides under TF32/bf16 or
        plain float nondeterminism, which says nothing about correctness.

        Runs in full fp32 (TF32 matmuls disabled for the duration) so that
        the tolerances stay meaningful regardless of the run's speed settings
        — this executes once per run, so the cost is irrelevant.

        Tolerance scale, measured on nuScenes with SwinB: encoding all
        prompts in one padded batch instead of one caption at a time shifts
        the text features by ~1e-5, which the image-text fusion amplifies to
        ~3e-4..5e-3 in box coordinates and ~7e-3..3e-2 in sigmoid scores,
        depending on the sample (each shard checks its own first sample, so
        the tolerances have to cover the spread, not one measurement).
        Actual structural breakage is orders of magnitude larger — comparing
        mismatched query sets gives ~1.0 — so these still catch it.

        Measured over the first sample of all 20 shards of a nuScenes
        trainval run: 1e-3..4e-3 typical, 1.3e-2 worst on detection scores,
        5e-3 worst on boxes. Each shard only ever checks one sample, so the
        tolerances are set well above that spread — a false alarm costs a
        multi-hour shard, while the containment test below catches real
        breakage without depending on any tolerance at all.
        """
        return self._self_check_fp32(images, box_threshold, score_tolerance,
                                     box_tolerance, margin)

    def _self_check_fp32(self, images, box_threshold, score_tolerance,
                         box_tolerance, margin):
        tf32_matmul = torch.backends.cuda.matmul.allow_tf32
        tf32_cudnn = torch.backends.cudnn.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        try:
            return self._compare_paths(images, box_threshold,
                                       score_tolerance, box_tolerance,
                                       margin)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = tf32_matmul
            torch.backends.cudnn.allow_tf32 = tf32_cudnn

    # A strong detection with no counterpart within this box distance (L1
    # over cxcywh, normalised coordinates) is a different object, not a
    # numerically shifted one.
    MISSING_BOX_DISTANCE = 0.1

    def _match_strong(self, strong, candidates, where, side):
        """Every confident detection of one path must exist in the other.

        Matching is by nearest box, never by score rank (near-equal scores
        swap rank under float noise) and never by counting: a detection
        sitting on the threshold legitimately flips sides between the two
        paths, so a count comparison is flaky by construction. What must
        hold is containment — each confident detection is present on both
        sides with the same box and score.
        """
        s_scores, s_boxes = strong
        c_scores, c_boxes = candidates
        worst_score = worst_box = 0.0
        for i in range(len(s_scores)):
            distance = ((c_boxes - s_boxes[i]).abs().sum(dim=-1)
                        if len(c_scores) else None)
            if (distance is None
                    or float(distance.min()) > self.MISSING_BOX_DISTANCE):
                raise RuntimeError(
                    f"Cached DINO path diverges from the stock forward "
                    f"({where}: a detection found by the {side} path "
                    f"(score {float(s_scores[i]):.3f}) has no counterpart "
                    "in the other). The installed groundingdino version "
                    "likely changed internals. Rerun with --no-dino-cache "
                    "and report this.")
            j = int(distance.argmin())
            worst_score = max(worst_score,
                              abs(float(s_scores[i]) - float(c_scores[j])))
            worst_box = max(worst_box,
                            (s_boxes[i] - c_boxes[j]).abs().max().item())
        return worst_score, worst_box

    def _compare_paths(self, images, box_threshold, score_tolerance,
                       box_tolerance, margin):
        # Worst deviation per comparison, each with its own tolerance: the
        # low-score tail is made of *different queries* (query selection
        # ranks near-tied encoder proposals), so its score distribution is
        # only a coarse structural signal, not a numerical one.
        worst = {"detection score": 0.0, "detection box": 0.0,
                 "score distribution": 0.0}
        tolerances = {"detection score": score_tolerance,
                      "detection box": box_tolerance,
                      # coarse fallback only — see below
                      "score distribution": 0.1}
        for prompt_idx in (0, len(self.prompts) - 1):
            caption = self.captions[prompt_idx]
            ncams = images.shape[0]
            images = images.to(self.device)
            ref = self.model(images, captions=[caption] * ncams)
            feats = self.image_features(images)
            logits, boxes = self.decode_prompts(*feats, [prompt_idx])
            ref_scores = ref["pred_logits"].float().sigmoid()
            ref_boxes = ref["pred_boxes"].float()
            cache_scores = logits.float().sigmoid()
            cache_boxes = boxes.float()

            for cam in range(ncams):
                where = f"prompt {self.prompts[prompt_idx]!r}, camera {cam}"
                # Confident detections of each path, matched against
                # everything the other path saw above the bare threshold.
                strong_ref = self._detections(
                    ref_scores[cam], ref_boxes[cam], box_threshold + margin)
                strong_cache = self._detections(
                    cache_scores[cam], cache_boxes[cam],
                    box_threshold + margin)
                all_ref = self._detections(
                    ref_scores[cam], ref_boxes[cam], box_threshold - margin)
                all_cache = self._detections(
                    cache_scores[cam], cache_boxes[cam],
                    box_threshold - margin)
                for strong, candidates, side in (
                        (strong_ref, all_cache, "stock"),
                        (strong_cache, all_ref, "cached")):
                    d_score, d_box = self._match_strong(
                        strong, candidates, where, side)
                    worst["detection score"] = max(worst["detection score"],
                                                   d_score)
                    worst["detection box"] = max(worst["detection box"], d_box)

                if len(strong_ref[0]) or len(strong_cache[0]):
                    continue
                # Fallback for a camera where this prompt found nothing: the
                # containment test above is vacuous, so compare the sorted
                # top query scores instead. Only a coarse structural signal
                # — these weak queries are *different* encoder proposals in
                # the two paths (near-tied proposals get ranked differently),
                # so their scores routinely differ by a few 1e-2 while every
                # real detection matches to 1e-4.
                top = min(10, ref_scores.shape[1])
                r_top = ref_scores[cam].max(dim=-1).values.sort(
                    descending=True).values[:top]
                c_top = cache_scores[cam].max(dim=-1).values.sort(
                    descending=True).values[:top]
                worst["score distribution"] = max(
                    worst["score distribution"],
                    (r_top - c_top).abs().max().item())

        over = {k: v for k, v in worst.items() if v > tolerances[k]}
        if over:
            detail = ", ".join(f"{k} {v:.2e} > {tolerances[k]:.0e}"
                               for k, v in sorted(over.items()))
            raise RuntimeError(
                f"Cached DINO path diverges from the stock forward "
                f"({detail}; all: "
                + ", ".join(f"{k}={v:.2e}" for k, v in sorted(worst.items()))
                + "). The installed groundingdino version likely changed "
                "internals. Rerun with --no-dino-cache and report this.")
        return max(worst.values())
