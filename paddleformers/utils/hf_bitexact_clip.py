# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Gradient clipping that reproduces torch's ``clip_grad_norm_`` bit-for-bit.

Only used when ``config.use_accuracy_compatible="hf"``; the default path is
untouched.

Why a separate clip is needed
-----------------------------
``paddle.nn.ClipGradByGlobalNorm`` and ``torch.nn.utils.clip_grad_norm_`` are the
same *formula* but a different *recipe*, and with a BF16 model the recipe decides
the bits:

===========================  ==================================  ==================================
step                         torch (transformers / accelerate)   paddle (ClipGradByGlobalNorm)
===========================  ==================================  ==================================
operand                      the BF16 ``p.grad``                 the FP32 ``p.main_grad``
per-tensor norm              ``_foreach_norm`` -> **BF16**       no per-tensor norm at all
global accumulation          squares of the BF16 per-tensor      squares of every element, summed
                             norms                               straight into one FP32 accumulator
global norm dtype            **BF16**                            FP32 (or FP64)
coefficient                  ``max_norm / (norm + 1e-6)``,       ``max_norm / max(norm, max_norm)``
                             then ``clamp(max=1.0)``, in BF16    in FP32
scaling                      BF16 grad * BF16 coef -> BF16       FP32 grad * FP32 coef -> FP32
===========================  ==================================  ==================================

The two differ in the value fed to the optimizer, so the trajectories separate on
the first clipped step. This class follows the torch column exactly:

1. per parameter: FP32 sum of squares of the gradient, ``sqrt``, **round to BF16**
   (matches ``torch._foreach_norm`` on a BF16 tensor -- FP32 opmath, BF16 result);
2. stack those BF16 per-tensor norms, FP32 sum of squares, ``sqrt``,
   **round to BF16** (matches ``torch.linalg.vector_norm`` over the stack);
3. ``den = bf16(fp32(norm) + fp32(1e-6))``; ``coef = bf16(fp32(max_norm) / fp32(den))``;
   ``coef = min(coef, 1.0)`` -- torch multiplies by the clamped coefficient
   unconditionally, so there is no "skip when below the threshold" branch;
4. every gradient is scaled with FP32 opmath and **rounded back to BF16**, then
   written into the FP32 ``main_grad`` buffer, so the buffer holds exactly the
   BF16 value torch's optimizer would read.

The pre-clip global norm and the clamped coefficient of the most recent step are
kept on the instance as ``last_global_norm`` / ``last_clip_coef``. Nothing is
printed or written to disk: this class only has to make the training step
match, and per-step dumps are of no use to CI/CE monitoring.
"""

from __future__ import annotations

from typing import Optional

import paddle
import paddle.nn as nn

from .accuracy_target import ACCURACY_TARGET_HF

__all__ = [
    "HFBitexactClipGradByGlobalNorm",
    "hf_bitexact_clip_enabled",
]


def hf_bitexact_clip_enabled(accuracy_target) -> bool:
    """Whether ``accuracy_target`` selects the torch ``clip_grad_norm_`` recipe.

    Takes the value rather than reading a global so the model config stays the
    single source of truth; see ``Trainer._build_grad_clip``. The value reaching
    here has already been canonicalized by ``LlmMetaConfig.set_llm_config``, so a
    plain equality test is enough.
    """
    return accuracy_target == ACCURACY_TARGET_HF


def _bf16(value: paddle.Tensor) -> paddle.Tensor:
    """Round an FP32 scalar/tensor through BF16 and return it as FP32.

    Every ``_foreach_*`` / reduction kernel torch runs on a BF16 tensor allocates
    a BF16 output, so each of them rounds once. Keeping the value in FP32 between
    the roundings (rather than in a BF16 buffer) avoids a second rounding on read
    while still discarding exactly the bits torch discards.
    """
    return value.astype("bfloat16").astype("float32")


# -- shared primitives ---------------------------------------------------------
#
# These four are the *recipe* of ``torch.nn.utils.clip_grad_norm_`` and are used
# both by the single-card clip below and by ``MoEHybridParallelClipGrad`` so the
# distributed path keeps the exact same roundings. In the distributed setting
# each rank holds a shard of the model, so the per-tensor BF16 norms are
# computed locally, their squares are what travel through the all-reduce, and
# the global norm / coefficient / scaling steps are shared verbatim.


def _hf_norm_sq(g: paddle.Tensor) -> paddle.Tensor:
    """FP32 square of the BF16-rounded L2 norm of ``g`` (a ``[1]`` tensor).

    Torch's ``_foreach_norm`` on a BF16 tensor uses FP32 opmath and rounds the
    result to BF16. The square is exact in FP32 (a BF16 value fits), so sending
    the squared BF16 norm through the distributed reduction loses nothing.
    """
    norm = _bf16(paddle.sqrt(paddle.sum(g.astype("float32") * g.astype("float32"), dtype="float32")))
    return norm * norm


def _hf_global_norm(total_sq: paddle.Tensor) -> paddle.Tensor:
    """``sqrt`` of the summed squared BF16 per-tensor norms, rounded to BF16."""
    return _bf16(paddle.sqrt(paddle.cast(total_sq, "float32")))


def _hf_clip_coef(global_norm: paddle.Tensor, clip_norm: float) -> paddle.Tensor:
    """Torch's coefficient: ``min(bf16(max / bf16(norm + 1e-6)), 1.0)``.

    One rounding per torch kernel, then the unconditional clamp -- torch
    multiplies by the clamped coefficient even when it is 1.0.
    """
    denom = _bf16(global_norm + paddle.full_like(global_norm, float(1e-6)))
    coef = _bf16(paddle.full_like(denom, float(clip_norm)) / denom)
    return paddle.minimum(coef, paddle.ones_like(coef))


def _hf_scale_grads(params_grads, coef: paddle.Tensor):
    """Scale every gradient with FP32 opmath, rounding back to its own dtype.

    Mirrors the tail of ``HFBitexactClipGradByGlobalNorm._dygraph_clip``: torch
    writes the coefficient into the BF16 ``p.grad`` in place, while this
    codebase promotes the FP32 ``main_grad`` back to the activation dtype. The
    FP32 (master) grads are rewritten in place; everything else is returned as a
    new tensor of the original dtype.
    """
    params_and_grads = []
    for p, g in params_grads:
        if g is None or not getattr(p, "need_clip", True):
            params_and_grads.append((p, g))
            continue
        scaled = _bf16(g.astype("float32") * coef)
        if g.dtype == paddle.float32:
            g[:] = scaled
            params_and_grads.append((p, g))
        else:
            params_and_grads.append((p, scaled.astype(g.dtype)))
    return params_and_grads


class HFBitexactClipGradByGlobalNorm(nn.ClipGradByGlobalNorm):
    """``ClipGradByGlobalNorm`` with torch's operation order and roundings."""

    def __init__(self, clip_norm: float, trainer=None) -> None:
        super().__init__(clip_norm)
        self._trainer = trainer
        #: pre-clip global norm of the most recent step (python float, BF16-exact)
        self.last_global_norm: Optional[float] = None
        #: clamped coefficient actually multiplied into the gradients
        self.last_clip_coef: Optional[float] = None

    # -- the clip itself -----------------------------------------------------
    @staticmethod
    def _norm_groups(param):
        """Column groups this parameter must be split into before norming.

        PaddleFleet fuses projections the reference keeps separate (GDN's four
        ``in_proj_*``, attention's ``q/k/v_proj``, the shared expert's
        ``gate_proj``/``up_proj``). Gradient clipping is **partition sensitive**:
        torch takes one BF16 per-tensor norm per ``nn.Linear`` and sums their
        squares, and ``bf16(sqrt(a^2+b^2))^2 != bf16(sqrt(a^2))^2 +
        bf16(sqrt(b^2))^2`` in general -- rounding each sub-norm to BF16 first
        discards different bits than rounding the combined one. The modules that
        fuse therefore publish ``hf_norm_groups``, and the clip norms each group
        separately so the global norm is taken over the reference's 141-tensor
        partition rather than the fused 111-tensor one.
        """
        return getattr(param, "hf_norm_groups", None)

    @paddle.no_grad()
    def _dygraph_clip(self, params_grads):
        selected = [(p, g) for p, g in params_grads if g is not None and getattr(p, "need_clip", True)]
        if not selected:
            return params_grads

        # (1) per-tensor norm: FP32 sum of squares -> sqrt -> BF16, over the
        #     REFERENCE's tensor partition (fused projections are split first).
        per_tensor = []
        for p, g in selected:
            gf = g.astype("float32")
            groups = self._norm_groups(p)
            if groups:
                flat = gf.reshape([-1, gf.shape[-1]])
                for columns in groups:
                    sub = flat.index_select(axis=-1, index=columns)
                    per_tensor.append(_bf16(paddle.sqrt(paddle.sum(sub * sub, dtype="float32"))))
            else:
                per_tensor.append(_bf16(paddle.sqrt(paddle.sum(gf * gf, dtype="float32"))))

        # (2) global norm over the BF16 per-tensor norms: FP32 accumulate -> sqrt -> BF16.
        stacked = paddle.stack(per_tensor).astype("float32")
        global_norm = _hf_global_norm(paddle.sum(stacked * stacked, dtype="float32"))

        # (3) coefficient, one rounding per torch kernel, then the unconditional clamp.
        clip_coef = _hf_clip_coef(global_norm, self.clip_norm)

        norm_value = float(global_norm)
        coef_value = float(clip_coef)
        self.last_global_norm = norm_value
        self.last_clip_coef = coef_value

        # (4) scale with FP32 opmath and round back to BF16, in place.
        return _hf_scale_grads(params_grads, clip_coef)
