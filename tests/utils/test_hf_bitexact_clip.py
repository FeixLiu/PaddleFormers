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

"""Tests for the HF bit-exact gradient clip and its distributed wrapper."""

import itertools
from types import SimpleNamespace

import paddle

from paddleformers.utils.hf_bitexact_clip import (
    HFBitexactClipGradByGlobalNorm,
    _bf16,
    _hf_global_norm,
)
from paddleformers.utils.moe_hybrid_parallel_optimizer import MoEHybridParallelClipGrad


def _param(name, *, is_distributed=False, no_sync=False, need_clip=True, groups=None):
    p = SimpleNamespace(
        name=name,
        is_distributed=is_distributed,
        no_sync=no_sync,
        need_clip=need_clip,
    )
    if groups is not None:
        p.hf_norm_groups = [paddle.to_tensor(c, dtype="int64") for c in groups]
    return p


def _clip(norm=1.0):
    return HFBitexactClipGradByGlobalNorm(norm)


class TestHFClipRecipe:
    def test_global_norm_rounds_through_bf16(self):
        # sqrt(2) in BF16 must come from the rounded value, not the FP32 literal.
        gn = _hf_global_norm(paddle.to_tensor([2.0], dtype="float32"))
        assert float(gn) == 1.4140625  # bf16(sqrt(2)), not 1.4142135...

    def test_no_clip_when_need_clip_false(self):
        clip = _clip(0.1)
        g = paddle.to_tensor([1.0, 2.0], dtype="float32")
        p = _param("p", need_clip=False)
        out = clip._dygraph_clip([(p, g)])
        # untouched and in place
        assert out[0][1] is g
        assert float(g.sum()) == 3.0

    def test_scale_clamps_coefficient_at_one(self):
        # ``norm < 1`` makes ``max/denom > 1``; the BF16-rounded coefficient is
        # clamped to 1.0 and the multiplication still happens (bit-identical),
        # so the FP32 grad ends up holding exactly its own BF16 rounding.
        clip = _clip(1.0)
        g = paddle.to_tensor([0.1, 0.2], dtype="float32")
        p = _param("p")
        clip._dygraph_clip([(p, g)])
        expected = _bf16(paddle.to_tensor([0.1], dtype="float32")) + _bf16(paddle.to_tensor([0.2], dtype="float32"))
        assert float(g.sum()) == float(expected)


class TestHFClipNormGroups:
    """``hf_norm_groups`` must change the global norm whenever the reference's
    per-Linear partition does.

    The registration itself happens in PaddleFleet, at model construction
    (``_maybe_tag_qkv_dgrad_groups`` / ``_maybe_tag_in_proj_dgrad_groups`` /
    ``_maybe_tag_up_gate_norm_groups``); this test pins the *contract* the clip
    relies on: a fused parameter that publishes two column groups normed
    separately must produce a different global norm than the unsplit one
    whenever BF16 rounding makes them differ.
    """

    @staticmethod
    def _find_differing_grouping():
        """Return ``(values, split_w, split_g)`` of four positive halves where
        ``bf16(sqrt(sum(v^2)))`` differs from ``bf16(sqrt(sum(2*bf16(half)^2)))``."""
        import math

        candidates = [1.0, 2.0, 3.0, 4.0, 1.5, 2.5, 3.5, 5.0]

        def bf16(x):
            return float(paddle.to_tensor([x], dtype="float32").astype("bfloat16").astype("float32")[0])

        for a, b, c, d in itertools.product(candidates, repeat=4):
            whole = bf16(math.sqrt(a * a + b * b + c * c + d * d))
            h1 = bf16(math.sqrt(a * a + b * b))
            h2 = bf16(math.sqrt(c * c + d * d))
            grouped = bf16(math.sqrt(h1 * h1 + h2 * h2))
            if whole != grouped and h1 > 0 and h2 > 0 and whole > 0:
                return [a, b, c, d], [0, 1], [2, 3], (whole, grouped)
        return None

    def test_fused_parameter_is_split_like_the_reference(self):
        found = self._find_differing_grouping()
        assert found is not None, "no grouping produced a differing global norm"
        values, g1, g2, (whole, grouped) = found

        # unsplit: one BF16 norm over the whole fused tensor
        clip = _clip(1.0)
        p = _param("fused", groups=None)
        g = paddle.to_tensor([values], dtype="float32")
        clip._dygraph_clip([(p, g)])
        assert clip.last_global_norm == whole

        # split: two BF16 norms, squared and summed, like torch's per-nn.Linear
        clip2 = _clip(1.0)
        p2 = _param("fused", groups=[g1, g2])
        g2t = paddle.to_tensor([values], dtype="float32")
        clip2._dygraph_clip([(p2, g2t)])
        assert clip2.last_global_norm == grouped
        # and the two recipes genuinely disagree for this input
        assert whole != grouped


class TestMoEHybridParallelClipHf:
    """The MoE distributed wrapper must keep the HF recipe, not replace it."""

    def _wrapped_clip(self, norm=1.0):
        inner = _clip(norm)
        wrapper = MoEHybridParallelClipGrad(inner, hcg=None, timers=None)
        # no distributed job in unit tests: the reduction step is a no-op, which
        # keeps the *local* part identical to the single-card recipe
        wrapper._global_norm = lambda *a, **k: None
        return inner, wrapper

    def test_distributed_wrapper_keeps_hf_global_norm(self):
        inner, wrapper = self._wrapped_clip()
        g = paddle.to_tensor([1.0, 2.0, 3.0], dtype="float32")
        p = _param("w")
        inner._dygraph_clip([(p, g)])
        single_norm = inner.last_global_norm
        wrapper._dygraph_clip([(p, paddle.to_tensor([1.0, 2.0, 3.0], dtype="float32"))])
        # identical to the single-card clip on the same input
        assert wrapper.stat["global_grad_norm"] == single_norm
        ref = _bf16(paddle.sqrt(paddle.to_tensor([1.0 + 4.0 + 9.0], dtype="float32")))
        assert wrapper.stat["global_grad_norm"] == float(ref)

    def test_distributed_wrapper_hf_uses_bf16_norm_squares(self):
        """The buckets must carry squared *rounded* norms; with a single param
        the sum is the square of the reference's per-tensor norm."""
        inner, wrapper = self._wrapped_clip()
        g = paddle.to_tensor([1.0, 2.0], dtype="float32")
        p = _param("w")
        wrapper._dygraph_clip([(p, g)])
        ref_norm = _bf16(paddle.sqrt(paddle.to_tensor([5.0], dtype="float32")))
        # global = bf16(sqrt(ref_norm^2)) == ref_norm
        assert float(ref_norm) == 2.234375
        assert wrapper.stat["global_grad_norm"] == float(ref_norm)

    def test_distributed_wrapper_scales_bf16_rounded(self):
        inner, wrapper = self._wrapped_clip(norm=0.5)
        g = paddle.to_tensor([4.0, 0.0], dtype="float32")
        p = _param("w")
        out = wrapper._dygraph_clip([(p, g)])
        # coef = bf16(0.5 / bf16(4 + 1e-6)) = 0.125 exactly; scaled = bf16(4*0.125)
        assert float(out[0][1].sum()) == 0.5

    def test_two_params_global_norm_sums_squares_of_rounded_norms(self):
        inner, wrapper = self._wrapped_clip()
        g1 = paddle.to_tensor([1.0, 0.0], dtype="float32")
        g2 = paddle.to_tensor([0.0, 2.0], dtype="float32")
        wrapper._dygraph_clip([(_param("a"), g1), (_param("b"), g2)])
        expected = float(_bf16(paddle.sqrt(paddle.to_tensor([1.0 + 4.0], dtype="float32"))))
        assert wrapper.stat["global_grad_norm"] == expected
