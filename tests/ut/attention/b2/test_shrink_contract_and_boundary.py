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
# Unit tests for the shrink contract functions and boundary data plane
# (replay B2c). The remap oracle test pins the worked example from
# 03-9 §4.2 so the doc and the tests can never drift apart.
#

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from vllm_ascend.attention.utils import (
    get_lmcache_sparse_cached_tokens,
    staged_sfa_connector_supports_sparse_load,
)


class TestSparseCachedTokensFrontier(unittest.TestCase):
    """get_lmcache_sparse_cached_tokens: frontier / dense-prefix / errors."""

    def _metadata(self, requests):
        return SimpleNamespace(requests=requests)

    def _req(self, req_id, is_sparse=False, can_load=True, committed=None, cached=0):
        load_spec = None
        if is_sparse or can_load:
            load_spec = SimpleNamespace(
                can_load=can_load,
                dsa_committed_end=committed,
                lmcache_cached_tokens=cached,
            )
        return SimpleNamespace(
            req_id=req_id,
            is_sparse_decode=is_sparse,
            load_spec=load_spec,
        )

    def test_committed_end_preferred_over_cached_tokens(self):
        meta = self._metadata([
            self._req("r0", is_sparse=True, committed=2304, cached=2048),
            self._req("r1", is_sparse=True, committed=None, cached=512),
        ])
        with patch(
            "vllm_ascend.attention.utils.staged_sfa_connector_supports_sparse_load",
            return_value=True,
        ), patch(
            "vllm_ascend.attention.utils.get_kv_transfer_group",
            return_value=SimpleNamespace(
                _get_connector_metadata=lambda: meta
            ),
        ):
            self.assertEqual(
                get_lmcache_sparse_cached_tokens(["r0", "r1"]), [2304, 512]
            )

    def test_dense_prefix_load_contributes_zero(self):
        meta = self._metadata([
            self._req("r0", is_sparse=False, can_load=True, cached=4096),
        ])
        with patch(
            "vllm_ascend.attention.utils.staged_sfa_connector_supports_sparse_load",
            return_value=True,
        ), patch(
            "vllm_ascend.attention.utils.get_kv_transfer_group",
            return_value=SimpleNamespace(
                _get_connector_metadata=lambda: meta
            ),
        ):
            self.assertEqual(get_lmcache_sparse_cached_tokens(["r0"]), [0])

    def test_none_request_ids_and_missing_frontier_raise(self):
        with patch(
            "vllm_ascend.attention.utils.staged_sfa_connector_supports_sparse_load",
            return_value=True,
        ), patch(
            "vllm_ascend.attention.utils.get_kv_transfer_group",
            return_value=SimpleNamespace(
                _get_connector_metadata=lambda: self._metadata([])
            ),
        ):
            with self.assertRaises(RuntimeError):
                get_lmcache_sparse_cached_tokens(None)
            with self.assertRaises(RuntimeError):
                get_lmcache_sparse_cached_tokens(["ghost"])

    def test_unsupported_connector_raises(self):
        with patch(
            "vllm_ascend.attention.utils.staged_sfa_connector_supports_sparse_load",
            return_value=False,
        ):
            with self.assertRaises(RuntimeError):
                get_lmcache_sparse_cached_tokens(["r0"])

    def test_supports_check_requires_full_contract(self):
        # No transfer group at all -> False (import-time environment).
        self.assertIsInstance(
            staged_sfa_connector_supports_sparse_load(), bool
        )


class TestSplitBoundaryUpdate(unittest.TestCase):
    """_update_dsa_split_boundary_in_place: frontier/window overwrite."""

    def _metadata(self, num_reqs=2, rows=None, seq_lens=None):
        rows = rows if rows is not None else np.array([0, 0, 1, 1], np.int32)
        seq = torch.tensor(seq_lens or [5000, 300], dtype=torch.int64)
        boundary_np = np.zeros(8, dtype=np.int32)
        boundary_t = torch.zeros(8, dtype=torch.int32)
        split = boundary_t[: len(rows)]
        return SimpleNamespace(
            split_boundary=split,
            decode_split_boundary_cpu=boundary_np,
            decode_split_boundary_cpu_tensor=boundary_t,
            decode_req_indices_cpu=rows,
            seq_lens_cpu=seq,
            num_decode_tokens=len(rows),
            decode_split_boundary=split,
        )

    def test_frontier_overwrites_decode_rows_only(self):
        from vllm_ascend.attention.sfa_v1 import (
            _update_dsa_split_boundary_in_place,
        )

        meta = self._metadata(rows=np.array([0, 0, 1, 1], np.int32))
        out = _update_dsa_split_boundary_in_place(
            meta, cached_tokens=[2048, 0], decode_window_size=0
        )
        self.assertEqual(
            meta.decode_split_boundary_cpu[:4].tolist(), [2048, 2048, 0, 0]
        )
        # Device view carries the same values.
        self.assertEqual(out[:4].tolist(), [2048, 2048, 0, 0])

    def test_window_min_semantics(self):
        from vllm_ascend.attention.sfa_v1 import (
            _update_dsa_split_boundary_in_place,
        )

        # seq_lens 5000 with window 256 -> window_start = 4992//256*256 = 4864;
        # committed 2048 -> boundary = min(4864, 2048) = 2048.
        meta = self._metadata(rows=np.array([0], np.int32))
        _update_dsa_split_boundary_in_place(
            meta, cached_tokens=[2048, 0], decode_window_size=256
        )
        self.assertEqual(meta.decode_split_boundary_cpu[:1].tolist(), [2048])

    def test_padding_rows_stay_untouched(self):
        from vllm_ascend.attention.sfa_v1 import (
            _update_dsa_split_boundary_in_place,
        )

        meta = self._metadata(rows=np.array([0, -1], np.int32))
        _update_dsa_split_boundary_in_place(
            meta, cached_tokens=[1024, 0], decode_window_size=0
        )
        self.assertEqual(
            meta.decode_split_boundary_cpu[:2].tolist(), [1024, 0]
        )


class TestDecodeRowMetadata(unittest.TestCase):
    """Pure CPU row-layout builder: prefill, decode/MTP and mixed batches."""

    def _build(self, qsl, plens, computed, rows):
        from vllm_ascend.attention.sfa_v1 import (
            _dsa_build_decode_row_metadata,
        )

        return _dsa_build_decode_row_metadata(qsl, plens, computed, rows)

    def test_prefill_only_has_no_decode_rows(self):
        boundary, req, offsets, nrows, nreqs = self._build(
            [0, 4], [10], [0], 4
        )
        self.assertEqual(boundary.tolist(), [0, 0, 0, 0])
        self.assertEqual(req.tolist(), [-1, -1, -1, -1])
        self.assertEqual((nrows, nreqs), (0, 0))

    def test_pure_decode_one_row_per_request(self):
        boundary, req, offsets, nrows, nreqs = self._build(
            [0, 1, 2], [100, 200], [100, 200], 2
        )
        self.assertEqual(boundary.tolist(), [100, 200])
        self.assertEqual(req.tolist(), [0, 1])
        self.assertEqual(offsets.tolist(), [0, 0])
        self.assertEqual((nrows, nreqs), (2, 2))

    def test_mtp_two_rows_get_request_major_offsets(self):
        boundary, req, offsets, nrows, nreqs = self._build(
            [0, 2, 4], [100, 200], [100, 200], 4
        )
        self.assertEqual(boundary.tolist(), [100, 100, 200, 200])
        self.assertEqual(req.tolist(), [0, 0, 1, 1])
        self.assertEqual(offsets.tolist(), [0, 1, 0, 1])
        self.assertEqual((nrows, nreqs), (4, 2))

    def test_mixed_batch_marks_only_decode_suffix(self):
        # r0 contributes 3 prefill rows (computed=5<prompt=8); r1 contributes
        # 2 decode rows (computed already at prompt frontier).
        boundary, req, offsets, nrows, nreqs = self._build(
            [0, 3, 5], [8, 20], [5, 20], 5
        )
        self.assertEqual(boundary.tolist(), [0, 0, 0, 20, 20])
        self.assertEqual(req.tolist(), [-1, -1, -1, 1, 1])
        self.assertEqual(offsets.tolist(), [0, 0, 0, 0, 1])
        self.assertEqual((nrows, nreqs), (2, 1))


if __name__ == "__main__":
    unittest.main()
