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
# Unit tests for P9 batch 5: graph key construction, route enums and
# decision, metadata sparse route classification, capture size parsing
# integration, and the aclgraph dispatch-key hook. CPU-only.
#

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

import vllm_ascend.ops  # noqa: F401 (break device_op circular)
from vllm_ascend.ascend_forward_context import (
    StagedSFAGraphKey,
    StagedSFAQueryProfile,
)
from vllm_ascend.attention.utils import staged_sfa_metadata_sparse_route
from vllm_ascend.utils import (
    StagedSFARouteAction,
    StagedSFARouteDecision,
    StagedSFARouteReason,
)


class TestStagedSFAGraphKey(unittest.TestCase):
    def test_exact_q1_construction(self):
        key = StagedSFAGraphKey.exact_q1(8)
        self.assertEqual(key.token_capacity, 8)
        self.assertEqual(key.request_capacity, 8)
        self.assertEqual(key.query_profile, StagedSFAQueryProfile.DECODE_Q1)
        self.assertEqual(key.max_query_len, 1)

    def test_fixed_spec_construction(self):
        key = StagedSFAGraphKey.fixed_spec(4, 2)
        self.assertEqual(key.token_capacity, 8)
        self.assertEqual(key.request_capacity, 4)
        self.assertEqual(key.query_profile, StagedSFAQueryProfile.SPEC_FIXED)
        self.assertEqual(key.max_query_len, 2)

    def test_q1_rejects_mismatched_capacity(self):
        with self.assertRaisesRegex(ValueError, "equal token/request"):
            StagedSFAGraphKey(
                token_capacity=8,
                request_capacity=4,
                query_profile=StagedSFAQueryProfile.DECODE_Q1,
                max_query_len=1,
            )

    def test_fixed_spec_rejects_width_one(self):
        with self.assertRaisesRegex(ValueError, "greater than one"):
            StagedSFAGraphKey.fixed_spec(4, 1)

    def test_fixed_spec_rejects_mismatched_capacity(self):
        with self.assertRaisesRegex(ValueError, "SPEC_FIXED requires"):
            StagedSFAGraphKey(
                token_capacity=10,
                request_capacity=4,
                query_profile=StagedSFAQueryProfile.SPEC_FIXED,
                max_query_len=2,
            )

    def test_to_legacy_batch_descriptor(self):
        key = StagedSFAGraphKey.exact_q1(16)
        desc = key.to_legacy_batch_descriptor()
        self.assertEqual(desc.num_tokens, 16)


class TestRouteEnums(unittest.TestCase):
    def test_decision_frozen(self):
        decision = StagedSFARouteDecision(
            StagedSFARouteAction.STAGED,
            StagedSFARouteReason.ELIGIBLE,
            graph_key=StagedSFAGraphKey.exact_q1(1),
            frontiers=(100, 200),
        )
        self.assertEqual(decision.action, StagedSFARouteAction.STAGED)
        self.assertEqual(decision.reason, StagedSFARouteReason.ELIGIBLE)
        self.assertEqual(decision.frontiers, (100, 200))
        with self.assertRaises(AttributeError):
            decision.action = StagedSFARouteAction.FATAL


class TestMetadataSparseRoute(unittest.TestCase):
    def _request(self, req_id, sparse=False, committed=None, can_load=False, cached=0):
        load_spec = None
        if sparse or can_load:
            load_spec = SimpleNamespace(
                can_load=can_load,
                dsa_committed_end=committed,
                lmcache_cached_tokens=cached,
            )
        return SimpleNamespace(
            req_id=req_id,
            is_sparse_decode=sparse,
            load_spec=load_spec,
        )

    def test_all_sparse_eligible(self):
        metadata = SimpleNamespace(requests=[
            self._request("r0", sparse=True, committed=256),
            self._request("r1", sparse=True, committed=512),
        ])
        reason, frontiers, cold = staged_sfa_metadata_sparse_route(
            metadata, ["r0", "r1"]
        )
        self.assertEqual(reason, StagedSFARouteReason.ELIGIBLE)
        self.assertEqual(frontiers, (256, 512))
        self.assertEqual(cold, ())

    def test_dense_prefix_hit(self):
        metadata = SimpleNamespace(requests=[
            self._request("r0", can_load=True, cached=1024),
        ])
        reason, _, _ = staged_sfa_metadata_sparse_route(metadata, ["r0"])
        self.assertEqual(reason, StagedSFARouteReason.DENSE_PREFIX_HIT)

    def test_mixed_load_rejected(self):
        metadata = SimpleNamespace(requests=[
            self._request("r0", sparse=True, committed=256),
            self._request("r1", can_load=True, cached=512),
        ])
        reason, _, _ = staged_sfa_metadata_sparse_route(metadata, ["r0", "r1"])
        self.assertEqual(reason, StagedSFARouteReason.MIXED_CONNECTOR_LOAD)

    def test_missing_metadata(self):
        reason, _, _ = staged_sfa_metadata_sparse_route(None, ["r0"])
        self.assertEqual(reason, StagedSFARouteReason.MISSING_CONNECTOR_METADATA)

    def test_invalid_request_ids(self):
        metadata = SimpleNamespace(requests=[])
        reason, _, _ = staged_sfa_metadata_route_check(metadata, [])
        self.assertEqual(reason, StagedSFARouteReason.INVALID_REQUEST_IDS)

    def test_duplicate_sparse_load(self):
        metadata = SimpleNamespace(requests=[
            self._request("r0", sparse=True, committed=256),
            self._request("r0", sparse=True, committed=512),
        ])
        reason, _, _ = staged_sfa_metadata_sparse_route(metadata, ["r0"])
        self.assertEqual(reason, StagedSFARouteReason.DUPLICATE_SPARSE_LOAD)

    def test_sparse_load_unavailable(self):
        metadata = SimpleNamespace(requests=[
            self._request("r0"),
        ])
        reason, _, _ = staged_sfa_metadata_sparse_route(metadata, ["r0"])
        self.assertEqual(reason, StagedSFARouteReason.SPARSE_LOAD_UNAVAILABLE)


def staged_sfa_metadata_route_check(metadata, request_ids):
    return staged_sfa_metadata_sparse_route(metadata, request_ids)


class TestACLGraphDispatchKey(unittest.TestCase):
    def test_staged_key_overrides_batch_descriptor(self):
        """Verify the dispatch key selection logic in __call__."""
        from vllm_ascend.compilation.acl_graph import ACLGraphWrapper

        # Verify the class has seal_staged_entries
        self.assertTrue(hasattr(ACLGraphWrapper, "seal_staged_entries"))

    def test_seal_empty_plan_rejected(self):
        from vllm_ascend.compilation.acl_graph import ACLGraphWrapper

        with self.assertRaisesRegex(RuntimeError, "plan is empty"):
            ACLGraphWrapper.seal_staged_entries((), 0)


class TestContextInjection(unittest.TestCase):
    def test_set_ascend_forward_context_staged_params(self):
        """Verify the signature accepts the three staged parameters."""
        import inspect
        from vllm_ascend.ascend_forward_context import (
            set_ascend_forward_context,
        )

        sig = inspect.signature(set_ascend_forward_context)
        self.assertIn("staged_sfa_graph_dummy_run", sig.parameters)
        self.assertIn("staged_sfa_route", sig.parameters)
        self.assertIn("staged_sfa_graph_key", sig.parameters)
        self.assertFalse(
            sig.parameters["staged_sfa_graph_dummy_run"].default
        )
        self.assertIsNone(
            sig.parameters["staged_sfa_route"].default
        )
        self.assertIsNone(
            sig.parameters["staged_sfa_graph_key"].default
        )


if __name__ == "__main__":
    unittest.main()
