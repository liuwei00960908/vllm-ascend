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
# Unit tests for the shrink forward behavior switches (replay B2d) and the
# remap oracle. The oracle test pins the worked example from 03-9 §4.2 so
# the doc and the tests cannot drift apart.
#

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from vllm_ascend.attention.sfa_v1 import AscendSFAImpl


class TestSkipDenseWait(unittest.TestCase):
    def _impl(self, shrink):
        impl = AscendSFAImpl.__new__(AscendSFAImpl)
        impl.dsa_shrink_latent = shrink
        return impl

    def test_skip_only_with_shrink_and_decode_rows(self):
        impl = self._impl(shrink=2)
        meta = SimpleNamespace(
            num_decode_tokens=4, need_sparse_lmcache_payload=True
        )
        self.assertTrue(impl._dsa_skip_dense_layer_wait(meta))
        meta = SimpleNamespace(
            num_decode_tokens=0, need_sparse_lmcache_payload=True
        )
        self.assertFalse(impl._dsa_skip_dense_layer_wait(meta))  # prefill-only batch
        impl = self._impl(shrink=0)
        meta = SimpleNamespace(
            num_decode_tokens=4, need_sparse_lmcache_payload=True
        )
        self.assertFalse(impl._dsa_skip_dense_layer_wait(meta))  # shrink off

    def test_dense_wait_preserved_without_sparse_contract(self):
        impl = self._impl(shrink=2)
        meta = SimpleNamespace(
            num_decode_tokens=4, need_sparse_lmcache_payload=False
        )
        self.assertFalse(impl._dsa_skip_dense_layer_wait(meta))

    def test_stage3_intentionally_skips_dense_wait_without_contract(self):
        # Isolation diagnostic: remap+FA over garbage scratch, no LMCache.
        impl = self._impl(shrink=3)
        meta = SimpleNamespace(
            num_decode_tokens=4, need_sparse_lmcache_payload=False
        )
        self.assertTrue(impl._dsa_skip_dense_layer_wait(meta))


class TestSkipDecodeSave(unittest.TestCase):
    def _impl(self, shrink):
        from vllm_ascend.attention import sfa_v1
        impl = AscendSFAImpl.__new__(AscendSFAImpl)
        impl.dsa_shrink_latent = shrink
        impl.dsa_unbundle = False
        return impl, sfa_v1

    def _meta(self, state):
        from vllm_ascend.attention.attention_v1 import AscendAttentionState
        return SimpleNamespace(attn_state=state)

    def test_pure_decode_states_skip_regular_save(self):
        from vllm_ascend.attention.attention_v1 import AscendAttentionState
        impl, _ = self._impl(shrink=2)
        kv = (torch.zeros(1), torch.zeros(1))
        for state in (
            AscendAttentionState.DecodeOnly,
            AscendAttentionState.SpecDecoding,  # MTP verify counts as pure decode
        ):
            with patch(
                "vllm_ascend.attention.sfa_v1._decode_window_save_window_size",
                return_value=0,
            ), patch(
                "vllm_ascend.attention.sfa_v1.maybe_save_kv_layer_to_connector"
            ) as mock_save:
                impl._maybe_save_unbundled_kv_cache(
                    "model.layers.0.self_attn.attn", kv, self._meta(state)
                )
            mock_save.assert_not_called()

    def test_decode_window_keeps_save_hook_active(self):
        from vllm_ascend.attention.attention_v1 import AscendAttentionState

        impl, _ = self._impl(shrink=2)
        kv = (torch.zeros(1), torch.zeros(1))
        with patch(
            "vllm_ascend.attention.sfa_v1._decode_window_save_window_size",
            return_value=256,
        ), patch(
            "vllm_ascend.attention.sfa_v1.maybe_save_kv_layer_to_connector"
        ) as mock_save:
            impl._maybe_save_unbundled_kv_cache(
                "model.layers.0.self_attn.attn",
                kv,
                self._meta(AscendAttentionState.DecodeOnly),
            )
        mock_save.assert_called_once()

    def test_prefill_and_shrink_off_still_save(self):
        from vllm_ascend.attention.attention_v1 import AscendAttentionState
        kv = (torch.zeros(1), torch.zeros(1))
        impl, _ = self._impl(shrink=2)
        with patch(
            "vllm_ascend.attention.sfa_v1.maybe_save_kv_layer_to_connector"
        ) as mock_save:
            impl._maybe_save_unbundled_kv_cache(
                "model.layers.0.self_attn.attn",
                kv,
                self._meta(AscendAttentionState.ChunkedPrefill),
            )
        mock_save.assert_called_once()

        impl, _ = self._impl(shrink=0)
        with patch(
            "vllm_ascend.attention.sfa_v1.maybe_save_kv_layer_to_connector"
        ) as mock_save:
            impl._maybe_save_unbundled_kv_cache(
                "model.layers.0.self_attn.attn",
                kv,
                self._meta(AscendAttentionState.DecodeOnly),
            )
        mock_save.assert_called_once()


class TestRemapOracle(unittest.TestCase):
    """Pin the worked example from 03-9 §4.2 (topk=8, block=4, boundary=10)."""

    def test_documented_example(self):
        from vllm_ascend.distributed.kv_transfer.sparse_offload.prepare_sparse_indices import (
            _prepare_sparse_indices_torch as oracle,
        )

        topk = torch.tensor([[2, 9, 5, 11, 3, 2, 9, 7]], dtype=torch.int32)
        boundary = torch.tensor([10], dtype=torch.int32)
        row_req = torch.tensor([0], dtype=torch.int32)
        block_table = torch.tensor([[77, 42]], dtype=torch.long)
        new_indices, packed, counts, targets = oracle(
            topk, boundary, row_req, block_table, block_size=4
        )
        # History (<10): {2,3,5,7,9} -> scratch rows 0..4; tail (>=10): 11 kept.
        self.assertEqual(new_indices[0].tolist(), [0, 4, 2, 11, 1, 0, 4, 3])
        self.assertEqual(packed[0, :5].tolist(), [2, 3, 5, 7, 9])
        self.assertEqual(int(counts[0]), 5)
        # scratch row 0-3 -> logical block 0 -> physical block 77 -> slots
        # 308..311; scratch row 4 -> logical block 1 -> physical 42 -> 168.
        self.assertEqual(targets[0, :5].tolist(), [308, 309, 310, 311, 168])

    def test_zero_boundary_selects_nothing(self):
        from vllm_ascend.distributed.kv_transfer.sparse_offload.prepare_sparse_indices import (
            _prepare_sparse_indices_torch as oracle,
        )

        topk = torch.tensor([[1, 2, 3]], dtype=torch.int32)
        boundary = torch.tensor([0], dtype=torch.int32)
        row_req = torch.tensor([0], dtype=torch.int32)
        block_table = torch.tensor([[5]], dtype=torch.long)
        new_indices, packed, counts, targets = oracle(
            topk, boundary, row_req, block_table, block_size=4
        )
        self.assertEqual(new_indices[0].tolist(), [1, 2, 3])  # all absolute
        self.assertEqual(int(counts[0]), 0)

    def test_union_dedupes_across_rows(self):
        from vllm_ascend.distributed.kv_transfer.sparse_offload.prepare_sparse_indices import (
            _prepare_sparse_indices_torch as oracle,
        )

        # Two rows of the same request selecting overlapping history: the
        # union stores each token once; both rows remap through it.
        topk = torch.tensor([[2, 5], [5, 7]], dtype=torch.int32)
        boundary = torch.tensor([10, 10], dtype=torch.int32)
        row_req = torch.tensor([0, 0], dtype=torch.int32)
        block_table = torch.tensor([[3]], dtype=torch.long)
        new_indices, packed, counts, targets = oracle(
            topk, boundary, row_req, block_table, block_size=4
        )
        self.assertEqual(new_indices[0].tolist(), [0, 1])
        self.assertEqual(new_indices[1].tolist(), [1, 2])
        self.assertEqual(packed[0, :3].tolist(), [2, 5, 7])
        self.assertEqual(int(counts[0]), 3)


if __name__ == "__main__":
    unittest.main()
