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
# Unit tests for the DSA unbundle consumer-side 3-tuple reassembly
# (replay Step 2 / A2b). Verifies that the MLA latent 2-tuple is
# re-assembled with the sibling indexer layer's key so the existing
# kv_cache[2] read/write paths stay unchanged.
#

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from vllm_ascend.attention.sfa_v1 import AscendSFAImpl, _dsa_indexer_layer_name


class TestDsaIndexerLayerName(unittest.TestCase):
    def test_maps_inner_mla_name_to_sibling_indexer_cache(self):
        self.assertEqual(
            _dsa_indexer_layer_name("model.layers.0.self_attn.attn"),
            "model.layers.0.self_attn.indexer.k_cache",
        )
        self.assertEqual(
            _dsa_indexer_layer_name("model.layers.77.self_attn.attn"),
            "model.layers.77.self_attn.indexer.k_cache",
        )


class TestDsaUnbundleReassemble(unittest.TestCase):
    def _build_impl(self, unbundle: bool):
        impl = AscendSFAImpl.__new__(AscendSFAImpl)
        impl.dsa_unbundle = unbundle
        impl._dsa_idx_cache_t = None
        return impl

    def test_disabled_unbundle_passthrough(self):
        impl = self._build_impl(unbundle=False)
        kv_cache = (torch.zeros(2), torch.zeros(3), torch.zeros(4))
        self.assertIs(impl._dsa_reassemble_kv_cache("model.layers.0.self_attn.attn", kv_cache), kv_cache)

    def test_three_tuple_passthrough(self):
        impl = self._build_impl(unbundle=True)
        kv_cache = (torch.zeros(2), torch.zeros(3), torch.zeros(4))
        self.assertIs(impl._dsa_reassemble_kv_cache("model.layers.0.self_attn.attn", kv_cache), kv_cache)

    def test_unbundle_reassembles_two_tuple_with_sibling_indexer(self):
        impl = self._build_impl(unbundle=True)
        k_nope = torch.zeros(2)
        k_pe = torch.zeros(3)
        indexer_t = torch.zeros(5)
        indexer_layer = SimpleNamespace(kv_cache=(indexer_t,))
        with patch(
            "vllm.forward_context.get_forward_context",
            return_value=SimpleNamespace(
                no_compile_layers={"model.layers.0.self_attn.indexer.k_cache": indexer_layer}
            ),
        ):
            reassembled = impl._dsa_reassemble_kv_cache("model.layers.0.self_attn.attn", (k_nope, k_pe))

        self.assertEqual(len(reassembled), 3)
        self.assertIs(reassembled[0], k_nope)
        self.assertIs(reassembled[1], k_pe)
        self.assertIs(reassembled[2], indexer_t)

    def test_sibling_cache_reference_is_cached(self):
        impl = self._build_impl(unbundle=True)
        indexer_t = torch.zeros(5)
        indexer_layer = SimpleNamespace(kv_cache=(indexer_t,))
        with patch(
            "vllm.forward_context.get_forward_context",
            return_value=SimpleNamespace(
                no_compile_layers={"model.layers.0.self_attn.indexer.k_cache": indexer_layer}
            ),
        ) as mock_get_fc:
            impl._dsa_reassemble_kv_cache("model.layers.0.self_attn.attn", (torch.zeros(2), torch.zeros(3)))
            mock_get_fc.assert_called_once()
            impl._dsa_reassemble_kv_cache("model.layers.0.self_attn.attn", (torch.zeros(2), torch.zeros(3)))
            mock_get_fc.assert_called_once()

    def test_bare_tensor_sibling_cache_used_directly(self):
        impl = self._build_impl(unbundle=True)
        indexer_t = torch.zeros(5)
        indexer_layer = SimpleNamespace(kv_cache=indexer_t)
        with patch(
            "vllm.forward_context.get_forward_context",
            return_value=SimpleNamespace(
                no_compile_layers={"model.layers.0.self_attn.indexer.k_cache": indexer_layer}
            ),
        ):
            reassembled = impl._dsa_reassemble_kv_cache("model.layers.0.self_attn.attn", (torch.zeros(2), torch.zeros(3)))

        self.assertIs(reassembled[2], indexer_t)


class TestDsaTwoGroupsMetadataFields(unittest.TestCase):
    """Step 4c: indexer group table/slot mirror fields default to None so the
    unbundle-only (shared block id space) path is unchanged; consumers fall
    back to the shared tables when None."""

    def test_sfa_metadata_indexer_fields_default_none(self):
        from dataclasses import fields

        from vllm_ascend.attention.sfa_v1 import AscendSFAMetadata

        field_names = {f.name for f in fields(AscendSFAMetadata)}
        self.assertIn("indexer_block_table", field_names)
        self.assertIn("indexer_slot_mapping", field_names)

    def test_common_metadata_indexer_fields_default_none(self):
        from dataclasses import fields

        from vllm_ascend.attention.utils import AscendCommonAttentionMetadata

        field_names = {f.name for f in fields(AscendCommonAttentionMetadata)}
        self.assertIn("indexer_block_table_tensor", field_names)
        self.assertIn("indexer_slot_mapping", field_names)

    def test_builder_mirrors_indexer_tables_when_present(self):
        # The builder slices common indexer tables with the same bounds as
        # the latent tables. Construct a minimal common metadata via
        # dataclass defaults + explicit tensors and a stubbed builder.
        from vllm_ascend.attention.sfa_v1 import AscendSFAMetadataBuilder

        builder = AscendSFAMetadataBuilder.__new__(AscendSFAMetadataBuilder)
        builder.metadata_cls = None  # not used; we intercept via a fake cls
        builder.kernel_block_size = 128
        released = {}

        class _FakeMeta:
            def __init__(self, **kwargs):
                released.update(kwargs)

        builder.metadata_cls = _FakeMeta

        class _MaskBuilder:
            def get_attention_mask(self, causal, model_config):
                return None

        builder.attn_mask_builder = _MaskBuilder()
        builder.model_config = SimpleNamespace(get_head_size=lambda: 128)
        builder.enable_dsa_cp = False
        builder.spec_actual_seq_lengths_query = None
        builder.spec_actual_seq_lengths_key = None

        import torch.nn as nn

        num_reqs, num_tokens = 2, 16
        latent_bt = torch.arange(num_reqs * 4, dtype=torch.int32).reshape(num_reqs, 4)
        indexer_bt = torch.arange(100, 100 + num_reqs * 4, dtype=torch.int32).reshape(num_reqs, 4)
        latent_sm = torch.arange(num_tokens, dtype=torch.int64)
        indexer_sm = torch.arange(1000, 1000 + num_tokens, dtype=torch.int64)
        common = SimpleNamespace(
            num_reqs=num_reqs,
            num_actual_tokens=num_tokens,
            num_input_tokens=num_tokens,
            block_table_tensor=latent_bt,
            slot_mapping=latent_sm,
            indexer_block_table_tensor=indexer_bt,
            indexer_slot_mapping=indexer_sm,
            positions=torch.arange(num_tokens),
            query_start_loc=torch.tensor([0, 8, 16], dtype=torch.int32),
            seq_lens=torch.tensor([8, 8], dtype=torch.int32),
            _seq_lens_cpu=None,
            seq_lens_cpu=None,
            causal=True,
            attn_state=None,
        )
        fake_cos_sin = (
            torch.zeros(num_tokens, 1, 64),
            torch.zeros(num_tokens, 1, 64),
        )
        with patch(
            "vllm_ascend.attention.sfa_v1.get_cos_and_sin_mla",
            return_value=fake_cos_sin,
        ), patch(
            "vllm_ascend.attention.sfa_v1.get_ascend_config",
            return_value=SimpleNamespace(c8_enable_reshape_optim=False),
        ):
            builder._build(common)
        self.assertIs(released["indexer_block_table"], indexer_bt)
        self.assertIs(released["indexer_slot_mapping"], indexer_sm)

        # Unbundle-only: indexer tables absent -> metadata keeps None.
        released.clear()
        common.indexer_block_table_tensor = None
        common.indexer_slot_mapping = None
        with patch(
            "vllm_ascend.attention.sfa_v1.get_cos_and_sin_mla",
            return_value=fake_cos_sin,
        ), patch(
            "vllm_ascend.attention.sfa_v1.get_ascend_config",
            return_value=SimpleNamespace(c8_enable_reshape_optim=False),
        ):
            builder._build(common)
        self.assertIsNone(released["indexer_block_table"])
        self.assertIsNone(released["indexer_slot_mapping"])


if __name__ == "__main__":
    unittest.main()
