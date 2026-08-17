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
# Unit tests for the Ascend indexer kernel-block-size adaptation
# (replay Step 2 / A2c). Upstream DeepseekV32IndexerBackend advertises
# CUDA/ROCm sizes ([64]); on Ascend it must report 128-token kernel
# blocks so select_common_block_size accepts block size 128 once the
# unbundle slice registers indexer specs.
#

import unittest

from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerBackend
from vllm.v1.worker.utils import select_common_block_size

from vllm_ascend.attention.sfa_v1 import AscendSFABackend
from vllm_ascend.worker.model_runner_v1 import _patch_ascend_indexer_kernel_block_size


class TestIndexerKernelBlockSize(unittest.TestCase):
    def tearDown(self):
        # Restore the upstream implementation so other tests are unaffected.
        DeepseekV32IndexerBackend.get_supported_kernel_block_sizes = staticmethod(
            lambda: [64]
        )

    def test_patch_reports_128_on_ascend(self):
        _patch_ascend_indexer_kernel_block_size()
        self.assertEqual(DeepseekV32IndexerBackend.get_supported_kernel_block_sizes(), [128])

    def test_select_common_block_size_accepts_128(self):
        _patch_ascend_indexer_kernel_block_size()
        self.assertEqual(
            select_common_block_size(128, [AscendSFABackend, DeepseekV32IndexerBackend]),
            128,
        )

    def test_unpatched_upstream_still_reports_64(self):
        # Guard: without the adaptation the upstream sizes remain the CUDA/ROCm
        # defaults, which is the regression this adaptation fixes.
        self.assertEqual(DeepseekV32IndexerBackend.get_supported_kernel_block_sizes(), [64])


if __name__ == "__main__":
    unittest.main()
