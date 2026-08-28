#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unit tests for P9 batch 4: custom op registration, fake output shapes,
# capture size parsing, and the non-staged early return of graph_pre/post.
# CPU-only; the ops register into torch.ops.vllm via PrivateUse1 dispatch
# (the NPU kernel itself is not exercised).
#

import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import torch

# Break the device_op circular import (see test_staged_functions.py).
import vllm_ascend.ops  # noqa: F401
import vllm_ascend.ops.mla as mla_module
from vllm_ascend.attention.sfa_v1 import AscendSFAImpl


class TestCustomOpRegistration(unittest.TestCase):
    def test_three_staged_ops_registered(self):
        for name in ("sfa_forward_pre", "sfa_lmcache_retrieve", "sfa_forward_post"):
            op = getattr(torch.ops.vllm, name, None)
            self.assertIsNotNone(op, f"torch.ops.vllm.{name} is not registered")

    def test_fake_output_shapes_match_bridge(self):
        hidden = torch.zeros(2, 16)
        bridge = mla_module.sfa_forward_pre_fake(
            hidden, False, torch.zeros(2, 4), "layer0",
            local_num_heads=2, kv_lora_rank=4, qk_rope_head_dim=2,
            index_topk=8, token_capacity=8, request_capacity=4,
            scratch_capacity=16,
        )
        self.assertEqual(
            [tuple(t.shape) for t in bridge],
            [(8, 2, 4), (8, 2, 2), (8, 1, 8), (4, 16), (4,), (4, 16)],
        )
        self.assertEqual(
            [t.dtype for t in bridge],
            [torch.float32, torch.float32, torch.int32, torch.int32,
             torch.int32, torch.int64],
        )

    def test_retrieve_and_post_fake_return_none(self):
        self.assertIsNone(
            mla_module.sfa_lmcache_retrieve_fake(
                torch.zeros(1, 8, dtype=torch.int32),
                torch.zeros(1, 16, dtype=torch.int32),
                torch.zeros(1, 8, dtype=torch.int64),
                torch.zeros(1, 4), "layer0", "layer1",
            )
        )
        self.assertIsNone(
            mla_module.sfa_forward_post_fake(
                torch.zeros(1, 2, 4), torch.zeros(1, 2, 2),
                torch.zeros(1, 1, 8, dtype=torch.int32),
                torch.zeros(1, 16, dtype=torch.int32),
                torch.zeros(1, 16, dtype=torch.int32),
                torch.zeros(1, 16, dtype=torch.int64),
                torch.zeros(1, 4), "layer0",
            )
        )


def _staged_vllm_config(**overrides):
    """A minimal config on which the staged predicate evaluates enabled."""
    from vllm.config import CUDAGraphMode

    config = SimpleNamespace(
        model_config=SimpleNamespace(
            use_mla=True,
            hf_text_config=SimpleNamespace(index_topk=8),
            hf_config=SimpleNamespace(),
        ),
        parallel_config=SimpleNamespace(
            data_parallel_size=1,
            pipeline_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
        ),
        scheduler_config=SimpleNamespace(
            max_num_seqs=64,
            max_num_batched_tokens=256,
        ),
        speculative_config=None,
        lora_config=None,
        kv_transfer_config=SimpleNamespace(),
        compilation_config=SimpleNamespace(
            cudagraph_mode=CUDAGraphMode.PIECEWISE
        ),
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def _staged_envs(**overrides):
    envs = {
        "VLLM_ASCEND_SFA_STAGED_GRAPH": True,
        "VLLM_ASCEND_SFA_STAGED_GRAPH_CAPTURE_SIZES": "4",
        "VLLM_ASCEND_DSA_UNBUNDLE": True,
        "VLLM_ASCEND_DSA_TWO_GROUPS": True,
        "VLLM_ASCEND_DSA_SHRINK_LATENT": 2,
    }
    envs.update(overrides)
    return envs


class TestCaptureSizeParsing(unittest.TestCase):
    """staged_sfa_graph_capture_sizes: request-count env -> token output."""

    def _sizes(self, raw, **config_overrides):
        from vllm_ascend.utils import staged_sfa_graph_capture_sizes

        envs = _staged_envs(
            VLLM_ASCEND_SFA_STAGED_GRAPH_CAPTURE_SIZES=raw
        )
        patches = [
            patch(f"vllm_ascend.envs.{name}", value)
            for name, value in envs.items()
        ]
        with ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            return staged_sfa_graph_capture_sizes(
                _staged_vllm_config(**config_overrides)
            )

    def test_single_size_width_one(self):
        self.assertEqual(self._sizes("4"), (4,))

    def test_multiple_sizes_deduped_and_sorted(self):
        self.assertEqual(self._sizes("32,8,16,8"), (8, 16, 32))

    def test_request_counts_scale_by_query_width(self):
        spec = SimpleNamespace(num_speculative_tokens=1, method="mtp")
        # 4 requests x width 2 = 8 tokens; 5 requests x 2 = 10 tokens.
        self.assertEqual(
            self._sizes("4,5", speculative_config=spec), (8, 10)
        )

    def test_empty_rejected(self):
        with self.assertRaisesRegex(ValueError, "positive integer"):
            self._sizes(",")

    def test_non_positive_rejected(self):
        with self.assertRaisesRegex(ValueError, "positive integer"):
            self._sizes("0,8")

    def test_non_numeric_rejected(self):
        with self.assertRaisesRegex(ValueError, "comma-separated"):
            self._sizes("abc")

    def test_oversize_requests_rejected(self):
        spec = SimpleNamespace(num_speculative_tokens=1, method="mtp")
        with self.assertRaisesRegex(ValueError, "exceed"):
            # 100 requests > max_num_seqs=64.
            self._sizes("100", speculative_config=spec)

    def test_oversize_tokens_rejected(self):
        spec = SimpleNamespace(num_speculative_tokens=1, method="mtp")
        with self.assertRaisesRegex(ValueError, "exceed"):
            # 200 requests x 2 = 400 tokens > max_num_batched_tokens=256.
            self._sizes("200", speculative_config=spec)


class TestGraphPreNonStagedEarlyReturn(unittest.TestCase):
    def test_no_attn_metadata_calls_forward_and_returns_empty_bridge(self):
        impl = AscendSFAImpl.__new__(AscendSFAImpl)
        impl._staged_sfa_bridge_buffers = tuple(
            torch.zeros(2, s) for s in (2, 2, 2, 2, 2, 2)
        )
        called = []

        def mock_forward(*args, **kwargs):
            called.append(args)

        impl.forward = mock_forward
        impl._cross_layer_empty_outputs = lambda hs: impl._staged_sfa_bridge_buffers
        with patch(
            "vllm_ascend.attention.sfa_v1.get_forward_context",
            return_value=SimpleNamespace(),
        ):
            result = impl.cross_layer_graph_pre(
                "layer0", torch.zeros(1, 8), (torch.zeros(1),) * 3,
                None, False, torch.zeros(1, 4),
            )
        self.assertEqual(len(called), 1)
        self.assertEqual(
            len(result), 6
        )

    def test_no_graph_key_calls_forward_and_returns_empty_bridge(self):
        impl = AscendSFAImpl.__new__(AscendSFAImpl)
        impl._staged_sfa_bridge_buffers = tuple(
            torch.zeros(2, s) for s in (2, 2, 2, 2, 2, 2)
        )
        impl.dsa_offload_unbundle = False
        called = []

        def mock_forward(*args, **kwargs):
            called.append(args)

        impl.forward = mock_forward
        impl._cross_layer_empty_outputs = lambda hs: impl._staged_sfa_bridge_buffers
        metadata = SimpleNamespace()
        with patch(
            "vllm_ascend.attention.sfa_v1.get_forward_context",
            return_value=SimpleNamespace(),
        ):
            result = impl.cross_layer_graph_pre(
                "layer0", torch.zeros(1, 8), (torch.zeros(1),) * 3,
                metadata, False, torch.zeros(1, 4),
            )
        self.assertEqual(len(called), 1)
        self.assertEqual(len(result), 6)


class TestGraphPostEarlyReturn(unittest.TestCase):
    def test_no_graph_key_returns_silently(self):
        impl = AscendSFAImpl.__new__(AscendSFAImpl)
        with patch(
            "vllm_ascend.attention.sfa_v1.get_forward_context",
            return_value=SimpleNamespace(),
        ):
            result = impl.cross_layer_graph_post(
                "layer0", torch.zeros(1), torch.zeros(1), torch.zeros(1),
                (torch.zeros(1),) * 3, None, torch.zeros(1, 4),
            )
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
