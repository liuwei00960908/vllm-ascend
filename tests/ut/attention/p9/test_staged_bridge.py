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
# Unit tests for P9 batch 2: the six fixed-address bridge slots and the
# immutable per-layer capture contract. Tests use CPU tensors deliberately;
# this batch is pure Python and has no NPU-kernel dependency.
#

import unittest
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import patch

import torch
from vllm.config import CUDAGraphMode

# Break the device_op <-> ops.fused_moe circular import (see
# test_staged_functions.py for the full explanation).
import vllm_ascend.ops  # noqa: F401

import vllm_ascend.attention.sfa_v1 as sfa_module
from vllm_ascend.attention.sfa_v1 import (
    AscendSFAImpl,
    _StagedSFACaptureState,
    _TensorBinding,
)


@dataclass(frozen=True)
class _Key:
    request_capacity: int


class _Event:
    pass


def _impl(*, capture_size=8, spec_tokens=1, index_topk=8):
    impl = AscendSFAImpl.__new__(AscendSFAImpl)
    impl._staged_sfa_graph_capture_sizes = (capture_size,)
    impl._staged_sfa_capture_state = _StagedSFACaptureState()
    impl._staged_sfa_bridge_buffers = None
    impl._dsa_idx_cache_t = object()
    impl.local_num_heads = 2
    impl.kv_lora_rank = 4
    impl.qk_rope_head_dim = 2
    impl.dsa_index_topk = index_topk
    speculative_config = (
        SimpleNamespace(num_speculative_tokens=spec_tokens)
        if spec_tokens is not None
        else None
    )
    impl.vllm_config = SimpleNamespace(speculative_config=speculative_config)
    return impl


class TestTensorBinding(unittest.TestCase):
    def test_records_address_and_full_layout(self):
        tensor = torch.empty((3, 4), dtype=torch.float32).t()
        binding = _TensorBinding.from_tensor(tensor)
        self.assertEqual(binding.address, tensor.data_ptr())
        self.assertEqual(binding.shape, tuple(tensor.shape))
        self.assertEqual(binding.stride, tuple(tensor.stride()))
        self.assertEqual(binding.dtype, tensor.dtype)
        self.assertEqual(binding.device, tensor.device)


class TestStagedBridgeBuffers(unittest.TestCase):
    @staticmethod
    def _warmup_context():
        return SimpleNamespace(cudagraph_runtime_mode=CUDAGraphMode.NONE)

    def test_six_slot_geometry_and_address_reuse(self):
        impl = _impl(capture_size=8, spec_tokens=1, index_topk=8)
        hidden = torch.empty((2, 16), dtype=torch.float32)
        with patch.object(
            sfa_module, "get_forward_context", return_value=self._warmup_context()
        ):
            buffers = impl._ensure_staged_sfa_bridge_buffers(hidden)
            reused = impl._ensure_staged_sfa_bridge_buffers(hidden)

        self.assertEqual(
            [tuple(tensor.shape) for tensor in buffers],
            [
                (8, 2, 4),    # ql_nope
                (8, 2, 2),    # q_pe
                (8, 1, 8),    # topk_indices
                (4, 16),      # selected_packed: 8 tokens / 2 rows
                (4,),         # selected_counts
                (4, 16),      # target_slots
            ],
        )
        self.assertEqual(
            [tensor.dtype for tensor in buffers],
            [
                torch.float32,
                torch.float32,
                torch.int32,
                torch.int32,
                torch.int32,
                torch.int64,
            ],
        )
        self.assertEqual(
            [tensor.data_ptr() for tensor in buffers],
            [tensor.data_ptr() for tensor in reused],
        )
        self.assertTrue(all(a is b for a, b in zip(buffers, reused, strict=True)))

    def test_allocation_requires_capture_sizes(self):
        impl = _impl()
        impl._staged_sfa_graph_capture_sizes = ()
        with self.assertRaisesRegex(RuntimeError, "capture sizes are unavailable"):
            impl._ensure_staged_sfa_bridge_buffers(torch.empty((1, 8)))

    def test_max_tokens_must_be_divisible_by_decode_width(self):
        impl = _impl(capture_size=7, spec_tokens=1)
        with self.assertRaisesRegex(RuntimeError, "must be divisible"):
            impl._ensure_staged_sfa_bridge_buffers(torch.empty((1, 8)))

    def test_capture_mode_cannot_allocate_late(self):
        impl = _impl()
        context = SimpleNamespace(cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE)
        with (
            patch.object(sfa_module, "get_forward_context", return_value=context),
            self.assertRaisesRegex(RuntimeError, "not allocated by eager warmup"),
        ):
            impl._ensure_staged_sfa_bridge_buffers(torch.empty((1, 8)))

    def test_copy_writes_prefix_and_reuses_fixed_storage(self):
        impl = _impl()
        hidden = torch.empty((2, 16), dtype=torch.float32)
        with patch.object(
            sfa_module, "get_forward_context", return_value=self._warmup_context()
        ):
            buffers = impl._ensure_staged_sfa_bridge_buffers(hidden)
            outputs = tuple(
                torch.full(
                    (2, *destination.shape[1:]),
                    fill_value=index + 1,
                    dtype=destination.dtype,
                )
                for index, destination in enumerate(buffers)
            )
            copied = impl._copy_to_staged_sfa_bridge(hidden, outputs)

        self.assertTrue(all(a is b for a, b in zip(buffers, copied, strict=True)))
        for source, destination in zip(outputs, buffers, strict=True):
            torch.testing.assert_close(destination[:2], source)

    def test_copy_rejects_arity_and_shape_mismatch(self):
        impl = _impl()
        hidden = torch.empty((2, 16), dtype=torch.float32)
        with patch.object(
            sfa_module, "get_forward_context", return_value=self._warmup_context()
        ):
            buffers = impl._ensure_staged_sfa_bridge_buffers(hidden)
            with self.assertRaisesRegex(RuntimeError, "unexpected bridge arity"):
                impl._copy_to_staged_sfa_bridge(hidden, buffers[:5])

            outputs = list(buffers)
            outputs[0] = torch.empty((9, *buffers[0].shape[1:]))
            with self.assertRaisesRegex(RuntimeError, "exceeds its fixed storage"):
                impl._copy_to_staged_sfa_bridge(hidden, tuple(outputs))


class TestStagedCaptureState(unittest.TestCase):
    def _resources(self):
        bridge = (torch.empty((4, 2)), torch.empty((4, 3)))
        kv_cache = (torch.empty((8, 4)), torch.empty((8, 2)))
        boundary = torch.empty(8, dtype=torch.int32)
        return bridge, kv_cache, boundary

    def test_register_requires_event_and_boundary(self):
        state = _StagedSFACaptureState()
        bridge, kv_cache, _ = self._resources()
        with self.assertRaisesRegex(RuntimeError, "storage is incomplete"):
            state.register(_Key(2), bridge, kv_cache)

    def test_register_rejects_duplicate_and_changed_bindings(self):
        state = _StagedSFACaptureState()
        bridge, kv_cache, boundary = self._resources()
        state.producer_event = _Event()
        state.remap_boundary = boundary[:4]
        state.register(_Key(2), bridge, kv_cache)

        with self.assertRaisesRegex(RuntimeError, "captured twice"):
            state.register(_Key(2), bridge, kv_cache)

        changed_bridge = (torch.empty((4, 2)), bridge[1])
        with self.assertRaisesRegex(RuntimeError, "bindings changed"):
            state.register(_Key(3), changed_bridge, kv_cache)

    def test_register_allows_boundary_outer_shape_change_same_storage(self):
        state = _StagedSFACaptureState()
        bridge, kv_cache, boundary = self._resources()
        state.producer_event = _Event()
        state.remap_boundary = boundary[:4]
        state.register(_Key(2), bridge, kv_cache)
        state.remap_boundary = boundary[:6]
        state.register(_Key(3), bridge, kv_cache)
        self.assertEqual(set(state.bindings), {_Key(2), _Key(3)})

    def test_seal_requires_runtime_and_exact_key_set(self):
        state = _StagedSFACaptureState()
        bridge, kv_cache, boundary = self._resources()
        state.producer_event = _Event()
        state.remap_boundary = boundary
        state.register(_Key(2), bridge, kv_cache)

        with self.assertRaisesRegex(RuntimeError, "capture state is incomplete"):
            state.seal((_Key(2),))

        state.runtime = ("layer", kv_cache, "index", True)
        with self.assertRaisesRegex(RuntimeError, "missing_keys"):
            state.seal((_Key(2), _Key(3)))

        state.register(_Key(3), bridge, kv_cache)
        state.seal((_Key(2), _Key(3)))


if __name__ == "__main__":
    unittest.main()
