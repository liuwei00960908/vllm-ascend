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
# Unit tests for the shrink data-plane injection (replay B2b).
#
# Verifies that AscendCommonAttentionMetadata carries request_ids /
# prompt_lens_cpu with None defaults (official paths unchanged) and that
# unpadded() slices both lists together with the other per-request fields
# (the fork left these unsliced in its own unpadded copy — a known fork
# defect the baseline pattern already fixes for the indexer mirror).
#

import unittest
from dataclasses import fields

from vllm_ascend.attention.utils import AscendCommonAttentionMetadata


def _metadata(**overrides):
    """Minimal AscendCommonAttentionMetadata for slicing tests."""
    required = {
        "query_start_loc": None,
        "query_start_loc_cpu": None,
        "seq_lens": None,
        "seq_lens_cpu": None,
        "num_computed_tokens_cpu": None,
        "num_reqs": 4,
        "num_actual_tokens": 16,
        "max_query_len": 4,
        "block_table_tensor": None,
        "slot_mapping": None,
        "causal": True,
    }
    required.update(overrides)
    return AscendCommonAttentionMetadata(**required)


class TestShrinkMetadataFields(unittest.TestCase):
    def test_fields_exist_with_none_defaults(self):
        names = {f.name for f in fields(AscendCommonAttentionMetadata)}
        self.assertIn("request_ids", names)
        self.assertIn("prompt_lens_cpu", names)
        # Official construction path: defaults keep both channels closed.
        cm = _metadata()
        self.assertIsNone(cm.request_ids)
        self.assertIsNone(cm.prompt_lens_cpu)

    def test_unpadded_slices_request_scoped_lists(self):
        cm = _metadata(
            num_reqs=4,
            request_ids=["r0", "r1", "r2", "r3"],
            prompt_lens_cpu=[10, 20, 30, 40],
            seq_lens_cpu=[11, 21, 31, 41],
        )
        cut = cm.unpadded(num_actual_tokens=8, num_actual_reqs=2)
        self.assertEqual(cut.request_ids, ["r0", "r1"])
        self.assertEqual(cut.prompt_lens_cpu, [10, 20])
        self.assertEqual(cut.seq_lens_cpu, [11, 21])

    def test_unpadded_keeps_none_channels_none(self):
        cm = _metadata(num_reqs=4)
        cut = cm.unpadded(num_actual_tokens=8, num_actual_reqs=2)
        self.assertIsNone(cut.request_ids)
        self.assertIsNone(cut.prompt_lens_cpu)


if __name__ == "__main__":
    unittest.main()
