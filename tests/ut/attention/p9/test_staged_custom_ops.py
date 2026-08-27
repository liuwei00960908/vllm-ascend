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


class TestCaptureSizeParsing(unittest.TestCase):
    def test_single_size(self):
        with patch(
            "vllm_ascend.envs.VLLM_ASCEND_SFA_STAGED_GRAPH_CAPTURE_SIZES", "4"
        ):
            self.assertEqual(
                AscendSFAImpl._parse_staged_capture_sizes(), (4,)
            )

    def test_multiple_sizes_deduped_and_sorted(self):
        with patch(
            "vllm_ascend.envs.VLLM_ASCEND_SFA_STAGED_GRAPH_CAPTURE_SIZES",
            "32,8,16,8",
        ):
            self.assertEqual(
                AscendSFAImpl._parse_staged_capture_sizes(), (8, 16, 32)
            )

    def test_empty_rejected(self):
        with patch(
            "vllm_ascend.envs.VLLM_ASCEND_SFA_STAGED_GRAPH_CAPTURE_SIZES", ","
        ):
            with self.assertRaisesRegex(ValueError, "positive integer"):
                AscendSFAImpl._parse_staged_capture_sizes()

    def test_non_positive_rejected(self):
        with patch(
            "vllm_ascend.envs.VLLM_ASCEND_SFA_STAGED_GRAPH_CAPTURE_SIZES", "0,8"
        ):
            with self.assertRaisesRegex(ValueError, "positive integer"):
                AscendSFAImpl._parse_staged_capture_sizes()

    def test_non_numeric_rejected(self):
        with patch(
            "vllm_ascend.envs.VLLM_ASCEND_SFA_STAGED_GRAPH_CAPTURE_SIZES", "abc"
        ):
            with self.assertRaisesRegex(ValueError, "Invalid"):
                AscendSFAImpl._parse_staged_capture_sizes()


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
