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


if __name__ == "__main__":
    unittest.main()
