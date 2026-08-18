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
# Unit tests for the DSA shared pool slab reshape (DSA replay Step 5 / 5b-1).
# One raw slab must produce the latent (k_nope, k_pe) 2-tuple view and the
# indexer 1-tuple view with the exact bundle geometry (2 latent blocks +
# 9 indexer blocks per bundle slot, slot 0 included).
#

import math
import unittest

import torch

from vllm_ascend.worker.dsa_shared_pool import reshape_dsa_shared_pool_raw


BLOCK = 128
KV_LORA_RANK = 512
QK_ROPE = 64
INDEX_DIM = 128
LATENT_PAGE = BLOCK * (KV_LORA_RANK + QK_ROPE) * 2
INDEXER_PAGE = BLOCK * INDEX_DIM * 2
BUNDLE_PAGE = math.lcm(LATENT_PAGE, INDEXER_PAGE)


class TestSharedSlabReshape(unittest.TestCase):
    def _slab(self, bundles: int) -> torch.Tensor:
        # (bundles + 1) slots of bundle_page bytes as raw int8.
        return torch.zeros((bundles + 1) * BUNDLE_PAGE, dtype=torch.int8)

    def test_bundle_geometry(self):
        self.assertEqual(BUNDLE_PAGE, 294912)
        self.assertEqual(BUNDLE_PAGE // LATENT_PAGE, 2)
        self.assertEqual(BUNDLE_PAGE // INDEXER_PAGE, 9)

    def test_latent_view_shapes(self):
        bundles = 3
        k_nope, k_pe = reshape_dsa_shared_pool_raw(
            self._slab(bundles),
            torch.bfloat16,
            BLOCK,
            1,
            KV_LORA_RANK,
            QK_ROPE,
            INDEX_DIM,
            is_indexer=False,
        )
        S = bundles + 1
        self.assertEqual(k_nope.shape, (2 * S, BLOCK, 1, KV_LORA_RANK))
        self.assertEqual(k_pe.shape, (2 * S, BLOCK, 1, QK_ROPE))
        # Byte accounting: nope region + pe region must cover the full slab.
        self.assertEqual(
            k_nope.numel() * 2 + k_pe.numel() * 2, (bundles + 1) * BUNDLE_PAGE
        )

    def test_indexer_view_shape(self):
        bundles = 3
        (indexer,) = reshape_dsa_shared_pool_raw(
            self._slab(bundles),
            torch.bfloat16,
            BLOCK,
            1,
            KV_LORA_RANK,
            QK_ROPE,
            INDEX_DIM,
            is_indexer=True,
        )
        S = bundles + 1
        self.assertEqual(indexer.shape, (9 * S, BLOCK, 1, INDEX_DIM))
        self.assertEqual(indexer.numel() * 2, (bundles + 1) * BUNDLE_PAGE)

    def test_views_cover_same_physical_memory(self):
        raw = self._slab(2)
        k_nope, k_pe = reshape_dsa_shared_pool_raw(
            raw, torch.bfloat16, BLOCK, 1, KV_LORA_RANK, QK_ROPE, INDEX_DIM,
            is_indexer=False,
        )
        (indexer,) = reshape_dsa_shared_pool_raw(
            raw, torch.bfloat16, BLOCK, 1, KV_LORA_RANK, QK_ROPE, INDEX_DIM,
            is_indexer=True,
        )
        # Both views alias the same storage: total bytes are identical and
        # mutating through one view is visible in the other.
        self.assertEqual(k_nope.numel() * 2 + k_pe.numel() * 2, indexer.numel() * 2)
        raw[0] = 7
        self.assertEqual(indexer.view(torch.int8).flatten()[0].item(), 7)


if __name__ == "__main__":
    unittest.main()
