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
# Unit tests for P9 batch 3: staged metadata fields, builder injection,
# module-level boundary/capacity helpers, and the non-staged early-return
# of the retrieve window. CPU-only; no NPU kernel dependency.
#

import unittest
from types import SimpleNamespace

import numpy as np
import torch

# Break the device_op <-> ops.fused_moe circular import that fires when
# sfa_v1 is imported directly by unittest (serving loads ops first and
# hides this). Importing the ops package here completes its initialization
# before sfa_v1 pulls in device_op.
import vllm_ascend.ops  # noqa: F401

from vllm_ascend.attention.sfa_v1 import (
    AscendSFAMetadata,
    _lmcache_sparse_wait_sync_once_enabled,
    _prepare_sfa_remap_boundary,
    _validate_dsa_scratch_capacity,
)


def _metadata(**overrides):
    """Create a minimal AscendSFAMetadata with staged fields populated."""
    base = {
        "num_actual_tokens": 4,
        "slot_mapping": torch.zeros(4, dtype=torch.int64),
        "seq_lens": torch.tensor([100, 200], dtype=torch.int64),
        "seq_lens_cpu": torch.tensor([100, 200], dtype=torch.int64),
        "cum_query_lens": torch.tensor([100, 200], dtype=torch.int64),
        "block_table": torch.zeros((2, 10), dtype=torch.int64),
        "sin": torch.zeros(4),
        "cos": torch.zeros(4),
    }
    base.update(overrides)
    return AscendSFAMetadata(**base)


def _staged_metadata(**overrides):
    """Create an AscendSFAMetadata with staged boundary fields populated."""
    overrides.setdefault(
        "decode_remap_boundary", torch.zeros(4, dtype=torch.int32)
    )
    overrides.setdefault("decode_remap_boundary_ready", False)
    overrides.setdefault("prompt_lens_cpu_rows", np.array([50, 50, 60, 60], dtype=np.int32))
    overrides.setdefault(
        "decode_req_indices_cpu", np.array([0, 0, 1, 1], dtype=np.int64)
    )
    overrides.setdefault("decode_scratch_capacity", 16)
    return _metadata(**overrides)


class TestStagedMetadataDefaults(unittest.TestCase):
    def test_new_fields_default_to_none(self):
        metadata = _metadata()
        self.assertIsNone(metadata.decode_union_mapping_workspace)
        self.assertIsNone(metadata.prompt_lens_cpu_rows)
        self.assertIsNone(metadata.decode_remap_boundary)
        self.assertFalse(metadata.decode_remap_boundary_ready)
        self.assertIsNone(metadata.decode_request_ids_compact)
        self.assertFalse(metadata.staged_sfa_payload_validated)


class TestValidateScratchCapacity(unittest.TestCase):
    def _boundary(self, values):
        return np.array(values, dtype=np.int32)

    def _req(self, values):
        return np.array(values, dtype=np.int64)

    def test_rejects_non_positive_index_topk(self):
        with self.assertRaisesRegex(RuntimeError, "positive index_topk"):
            _validate_dsa_scratch_capacity(
                self._boundary([100, 100]),
                self._req([0, 1]),
                None,
                index_topk=0,
            )

    def test_rejects_missing_or_small_capacity(self):
        with self.assertRaisesRegex(RuntimeError, "missing or too small"):
            _validate_dsa_scratch_capacity(
                self._boundary([100]),
                self._req([0]),
                None,
                index_topk=8,
                scratch_capacity=None,
            )

    def test_rejects_row_overcapacity(self):
        # 2 rows × width 8 = 16 > capacity 8
        with self.assertRaisesRegex(RuntimeError, "too small"):
            _validate_dsa_scratch_capacity(
                self._boundary([100, 100]),
                self._req([0, 0]),
                None,
                index_topk=8,
                scratch_capacity=8,
            )

    def test_rejects_boundary_aliasing_scratch(self):
        # boundary=4 < capacity=8 with non-zero → would alias
        with self.assertRaisesRegex(RuntimeError, "alias live KV"):
            _validate_dsa_scratch_capacity(
                self._boundary([4]),
                self._req([0]),
                None,
                index_topk=8,
                scratch_capacity=8,
            )

    def test_accepts_valid_layout(self):
        # boundary >= capacity → scratch [0,8) doesn't touch live [8,100)
        _validate_dsa_scratch_capacity(
            self._boundary([100]),
            self._req([0]),
            None,
            index_topk=8,
            scratch_capacity=8,
        )

    def test_accepts_zero_boundary(self):
        # boundary=0 means "nothing selected" — no alias by definition
        _validate_dsa_scratch_capacity(
            self._boundary([0]),
            self._req([0]),
            None,
            index_topk=8,
            scratch_capacity=8,
        )


class TestPrepareSfaRemapBoundary(unittest.TestCase):
    def test_idempotent_when_ready(self):
        metadata = _staged_metadata()
        metadata.decode_remap_boundary_ready = True
        result = _prepare_sfa_remap_boundary(
            metadata, ["req-0", "req-1"], is_dummy_run=True, index_topk=8
        )
        self.assertIs(result, metadata.decode_remap_boundary)
        # No computation happened — boundary still zeros
        self.assertTrue(torch.all(result == 0))

    def test_rejects_missing_cpu_metadata(self):
        metadata = _staged_metadata()
        metadata.prompt_lens_cpu_rows = None
        with self.assertRaisesRegex(RuntimeError, "CPU metadata is incomplete"):
            _prepare_sfa_remap_boundary(
                metadata, ["req-0", "req-1"], is_dummy_run=True, index_topk=8
            )

    def test_rejects_missing_boundary_storage(self):
        metadata = _staged_metadata()
        metadata.decode_remap_boundary = None
        with self.assertRaisesRegex(RuntimeError, "boundary storage"):
            _prepare_sfa_remap_boundary(
                metadata, ["req-0", "req-1"], is_dummy_run=True, index_topk=8
            )

    def test_dummy_run_fills_prompt_boundary(self):
        metadata = _staged_metadata()
        result = _prepare_sfa_remap_boundary(
            metadata, ["req-0", "req-1"], is_dummy_run=True, index_topk=8
        )
        # Dummy run: boundary = prompt_lens (no cache frontiers)
        expected = np.array([50, 50, 60, 60], dtype=np.int32)
        np.testing.assert_array_equal(
            result.numpy(), expected
        )
        self.assertTrue(metadata.decode_remap_boundary_ready)

    def test_cached_tokens_override_boundary(self):
        metadata = _staged_metadata()
        result = _prepare_sfa_remap_boundary(
            metadata,
            ["req-0", "req-1"],
            is_dummy_run=False,
            index_topk=8,
            cached_tokens=(30, 40),
        )
        # row 0,1 → req 0 → cached=30; row 2,3 → req 1 → cached=40
        np.testing.assert_array_equal(
            result.numpy(), np.array([30, 30, 40, 40], dtype=np.int32)
        )


class TestRetrieveWindowEarlyReturn(unittest.TestCase):
    def test_non_staged_returns_immediately(self):
        """Non-staged path: graph_key is None → zero side effects."""
        from vllm_ascend.attention.sfa_v1 import AscendSFAImpl

        impl = AscendSFAImpl.__new__(AscendSFAImpl)
        context = SimpleNamespace()  # No staged_sfa_graph_key
        metadata = _metadata()
        result = impl.cross_layer_lmcache_retrieve(
            "layer0",
            "layer1",
            torch.zeros(1, 8, dtype=torch.int32),
            torch.zeros(1, 16, dtype=torch.int32),
            torch.zeros(1, 8, dtype=torch.int64),
            metadata,
            context,
        )
        self.assertIsNone(result)

    def test_dummy_run_returns_after_boundary_prep(self):
        """Dummy run: only prepares boundary, no LMCache call."""
        from vllm_ascend.attention.sfa_v1 import AscendSFAImpl

        impl = AscendSFAImpl.__new__(AscendSFAImpl)
        next_metadata = _staged_metadata()
        next_metadata.req_ids = ["req-0", "req-1"]
        context = SimpleNamespace(
            staged_sfa_graph_key=object(),
            staged_sfa_graph_dummy_run=True,
            attn_metadata={"layer1": next_metadata},
        )
        metadata = _metadata()
        impl.cross_layer_lmcache_retrieve(
            "layer0",
            "layer1",
            torch.zeros(1, 8, dtype=torch.int32),
            torch.zeros(1, 16, dtype=torch.int32),
            torch.zeros(1, 8, dtype=torch.int64),
            metadata,
            context,
        )
        # Next layer boundary should now be ready
        self.assertTrue(next_metadata.decode_remap_boundary_ready)


class TestWaitSyncOnceGate(unittest.TestCase):
    def test_env_gate_reads_config(self):
        # Just verify it returns a bool without crashing
        self.assertIsInstance(_lmcache_sparse_wait_sync_once_enabled(), bool)


if __name__ == "__main__":
    unittest.main()
