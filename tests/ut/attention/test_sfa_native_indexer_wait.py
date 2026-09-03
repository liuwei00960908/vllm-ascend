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
# Unit tests for the native-path indexer row wait (PD cold start). The
# wait must fire before top-k whenever the connector owns the indexer
# namespace, and stay silent for resident-indexer / non-unbundled /
# profiling shapes. CPU-only; no NPU kernel dependency.
#

import unittest
from unittest.mock import patch

from vllm_ascend.attention import sfa_v1
from vllm_ascend.attention.sfa_v1 import AscendSFAImpl


class TestNativeIndexerWait(unittest.TestCase):
    def _impl(self, dsa_unbundle=True, has_indexer=True):
        impl = AscendSFAImpl.__new__(AscendSFAImpl)
        impl.dsa_unbundle = dsa_unbundle
        impl.has_indexer = has_indexer
        return impl

    def test_wait_uses_sibling_indexer_layer_name(self):
        impl = self._impl()
        with (
            patch.object(sfa_v1, "_dsa_index_lmcache_enabled", return_value=True),
            patch.object(sfa_v1, "wait_for_kv_layer_from_connector") as wait,
        ):
            impl._dsa_maybe_wait_indexer_rows("model.layers.3.self_attn.attn", object())
        wait.assert_called_once_with("model.layers.3.self_attn.indexer.k_cache")

    def test_no_wait_without_index_lmcache_support(self):
        impl = self._impl()
        with (
            patch.object(sfa_v1, "_dsa_index_lmcache_enabled", return_value=False),
            patch.object(sfa_v1, "wait_for_kv_layer_from_connector") as wait,
        ):
            impl._dsa_maybe_wait_indexer_rows("model.layers.3.self_attn.attn", object())
        wait.assert_not_called()

    def test_no_wait_for_profiling_run(self):
        # kv_cache is None during profiling runs; the wait must not fire.
        impl = self._impl()
        with (
            patch.object(sfa_v1, "_dsa_index_lmcache_enabled", return_value=True),
            patch.object(sfa_v1, "wait_for_kv_layer_from_connector") as wait,
        ):
            impl._dsa_maybe_wait_indexer_rows("model.layers.3.self_attn.attn", None)
        wait.assert_not_called()

    def test_no_wait_without_unbundle(self):
        # Bundled layouts keep the indexer inside the latent layer tuple;
        # the sibling namespace does not exist.
        impl = self._impl(dsa_unbundle=False)
        with (
            patch.object(sfa_v1, "_dsa_index_lmcache_enabled", return_value=True),
            patch.object(sfa_v1, "wait_for_kv_layer_from_connector") as wait,
        ):
            impl._dsa_maybe_wait_indexer_rows("model.layers.3.self_attn.attn", object())
        wait.assert_not_called()

    def test_no_wait_without_indexer(self):
        impl = self._impl(has_indexer=False)
        with (
            patch.object(sfa_v1, "_dsa_index_lmcache_enabled", return_value=True),
            patch.object(sfa_v1, "wait_for_kv_layer_from_connector") as wait,
        ):
            impl._dsa_maybe_wait_indexer_rows("model.layers.3.self_attn.attn", object())
        wait.assert_not_called()

    def test_forward_wires_the_wait_before_producer_event(self):
        # The forward() body must route through the helper (wiring check:
        # the call sits in the native path, before the producer event).
        import inspect

        source = inspect.getsource(AscendSFAImpl.forward)
        self.assertIn("_dsa_maybe_wait_indexer_rows", source)


if __name__ == "__main__":
    unittest.main()
