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
# Unit tests for the P9 batch-5 wiring fixes: route gates (staged=0 zero
# impact, kv_scales / DP / capture-pending / cascade), capture lifecycle
# guards (sentinel, attempt-once, reset-before-capture), data-plane repairs
# (frontier propagation, payload event, dsa_unbundle naming, prompt-row
# expansion), the ops production branch, and the async replay fence.
# CPU-only; no NPU kernel dependency.
#

import inspect
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import torch

import vllm_ascend.ops  # noqa: F401 (break device_op circular)
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.sfa_v1 import AscendSFAImpl
from vllm_ascend.utils import (
    StagedSFARouteAction,
    StagedSFARouteReason,
    staged_sfa_graph_capture_sizes,
    staged_sfa_graph_configured,
)


def _load_model_runner():
    from vllm_ascend.worker import model_runner_v1

    return model_runner_v1


def _route_runner(**overrides):
    """A bare NPUModelRunner carrying only the routing-chain attributes."""
    model_runner_v1 = _load_model_runner()
    runner = model_runner_v1.NPUModelRunner.__new__(
        model_runner_v1.NPUModelRunner
    )
    defaults = dict(
        _staged_sfa_graph_capture_sizes=(4, 8),
        _staged_sfa_impls=(("model.layers.0.self_attn.attn", object()),),
        calculate_kv_scales=False,
        speculative_config=None,
        vllm_config=SimpleNamespace(lora_config=None),
        attn_state=AscendAttentionState.DecodeOnly,
        parallel_config=SimpleNamespace(data_parallel_size=1),
    )
    defaults.update(overrides)
    for key, value in defaults.items():
        setattr(runner, key, value)
    return runner


class TestLocalRouteGates(unittest.TestCase):
    """Every gate must return SAFE_NATIVE before any staged authorization."""

    def _route(self, runner, **kwargs):
        model_runner_v1 = _load_model_runner()
        call_kwargs = dict(
            num_tokens_unpadded=4,
            num_reqs=4,
            num_scheduled_tokens=np.ones(4, dtype=np.int32),
            index_topk=8,
            request_ids=["r0", "r1", "r2", "r3"],
            kv_connector_metadata=None,
        )
        call_kwargs.update(kwargs)
        return model_runner_v1.NPUModelRunner._staged_sfa_local_route(
            runner, **call_kwargs
        )

    def test_not_configured_when_sizes_empty(self):
        runner = _route_runner(_staged_sfa_graph_capture_sizes=())
        decision = self._route(runner)
        self.assertEqual(decision.action, StagedSFARouteAction.SAFE_NATIVE)
        self.assertEqual(decision.reason, StagedSFARouteReason.NOT_CONFIGURED)

    def test_kv_scales_routes_native(self):
        runner = _route_runner(calculate_kv_scales=True)
        decision = self._route(runner)
        self.assertEqual(decision.reason, StagedSFARouteReason.RUNTIME_MODE)

    def test_dp_greater_than_one_fails_closed(self):
        runner = _route_runner(
            parallel_config=SimpleNamespace(data_parallel_size=2)
        )
        decision = self._route(runner)
        self.assertEqual(
            decision.reason, StagedSFARouteReason.RUNTIME_PARALLELISM
        )

    def test_capture_pending_until_impls_sealed(self):
        runner = _route_runner(_staged_sfa_impls=None)
        decision = self._route(runner)
        self.assertEqual(decision.reason, StagedSFARouteReason.CAPTURE_PENDING)
        runner2 = _route_runner(_staged_sfa_impls=())
        decision2 = self._route(runner2)
        self.assertEqual(decision2.reason, StagedSFARouteReason.CAPTURE_PENDING)

    def test_cascade_routes_native(self):
        runner = _route_runner()
        decision = self._route(runner, has_cascade_attention=True)
        self.assertEqual(decision.reason, StagedSFARouteReason.CASCADE)

    def test_lora_routes_native(self):
        runner = _route_runner(
            vllm_config=SimpleNamespace(lora_config=object())
        )
        decision = self._route(runner)
        self.assertEqual(decision.reason, StagedSFARouteReason.LORA)


class TestRunnerWiringInvariants(unittest.TestCase):
    """AST/source invariants for the fixes that integration tests cannot
    reach without a full NPU model (staged=0 zero impact, sentinel, dummy
    capture context, bootstrap/save hooks)."""

    def _method_source(self, name):
        model_runner_v1 = _load_model_runner()
        return inspect.getsource(
            getattr(model_runner_v1.NPUModelRunner, name)
        )

    def test_init_uses_none_sentinel_for_impls(self):
        source = self._method_source("__init__")
        self.assertIn("self._staged_sfa_impls: tuple | None = None", source)
        self.assertIn("self.dsa_index_topk = 0", source)

    def test_execute_model_gates_routing_on_enable_flag(self):
        source = self._method_source("execute_model")
        self.assertIn("if self.enable_staged_sfa_graph", source)
        # Context still receives the staged channels.
        self.assertIn("staged_sfa_route=staged_sfa_route", source)
        self.assertIn("staged_sfa_graph_key=staged_sfa_graph_key", source)

    def test_execute_model_bootstraps_and_saves_around_forward(self):
        source = self._method_source("execute_model")
        self.assertIn("bootstrap_cross_layer", source)
        self.assertIn("submit_cross_layer_save", source)
        self.assertLess(
            source.index("bootstrap_cross_layer"),
            source.index("_model_forward"),
        )
        self.assertLess(
            source.index("_model_forward"),
            source.rindex("submit_cross_layer_save"),
        )

    def test_capture_model_collects_unconditionally_and_resets(self):
        source = self._method_source("capture_model")
        self.assertIn("_reset_staged_sfa_startup_capture", source)
        self.assertIn("_staged_sfa_startup_capture_attempted", source)
        self.assertIn("no local SFA layers were", source)
        self.assertNotIn(
            'getattr(self, "_staged_sfa_impls", None) is None', source
        )

    def test_dummy_run_threads_staged_context(self):
        source = self._method_source("_dummy_run")
        self.assertIn("staged_sfa_graph_dummy_run=staged_sfa_graph_dummy_run", source)
        self.assertIn("staged_sfa_graph_key=staged_dummy_key", source)
        self.assertIn("_prepare_staged_sfa_dummy_block_tables", source)
        self.assertIn("_staged_sfa_query_start_locs", source)

    def test_build_attention_metadata_accepts_dummy_flag(self):
        model_runner_v1 = _load_model_runner()
        sig = inspect.signature(
            model_runner_v1.NPUModelRunner._build_attention_metadata
        )
        self.assertIn("staged_sfa_graph_dummy_run", sig.parameters)

    def test_dummy_helpers_exist(self):
        model_runner_v1 = _load_model_runner()
        for name in (
            "_staged_sfa_query_start_locs",
            "_staged_sfa_dummy_request_ids",
            "_staged_sfa_dummy_batch_size",
            "_prepare_staged_sfa_dummy_block_tables",
        ):
            self.assertTrue(
                hasattr(model_runner_v1.NPUModelRunner, name),
                f"missing {name}",
            )
        self.assertTrue(
            hasattr(model_runner_v1, "_staged_sfa_dummy_remap_boundaries")
        )


class TestOpsProductionBranch(unittest.TestCase):
    def test_forward_dispatches_staged_ops_in_order(self):
        import vllm_ascend.ops.mla as mla_mod

        source = inspect.getsource(mla_mod.AscendMultiHeadLatentAttention.forward)
        self.assertIn("if self.use_cross_layer_sfa:", source)
        pre = source.index("torch.ops.vllm.sfa_forward_pre(")
        retrieve = source.index("torch.ops.vllm.sfa_lmcache_retrieve(")
        post = source.index("torch.ops.vllm.sfa_forward_post(")
        native = source.index("torch.ops.vllm.mla_forward(")
        self.assertLess(pre, retrieve)
        self.assertLess(retrieve, post)
        # Native call stays in the else branch (after the staged block).
        self.assertLess(post, native)

    def test_init_resolves_next_layer_and_staged_flag(self):
        import vllm_ascend.ops.mla as mla_mod

        source = inspect.getsource(mla_mod.AscendMultiHeadLatentAttention.__init__)
        self.assertIn("self.next_layer_name = (", source)
        self.assertIn("self.use_cross_layer_sfa = (", source)


class TestDataPlaneRepairs(unittest.TestCase):
    def _impl(self, **overrides):
        impl = AscendSFAImpl.__new__(AscendSFAImpl)
        defaults = dict(
            dsa_unbundle=False,
            dsa_index_topk=8,
            layer_name="model.layers.0.self_attn.attn",
        )
        defaults.update(overrides)
        for key, value in defaults.items():
            setattr(impl, key, value)
        return impl

    def test_cross_layer_kv_cache_reads_dsa_unbundle(self):
        impl = self._impl()
        kv = (torch.zeros(1), torch.zeros(1), torch.zeros(1))
        out, index_layer, index_enabled = impl._cross_layer_kv_cache(
            "model.layers.0.self_attn.attn", kv
        )
        # The historical bug raised AttributeError on the fork-only
        # dsa_offload_unbundle name; reaching here means the fix holds.
        self.assertEqual(len(out), 3)
        self.assertIsNone(index_layer)
        self.assertFalse(index_enabled)

    def test_cross_layer_kv_cache_unbundle_keeps_three_tensors(self):
        impl = self._impl(dsa_unbundle=True)
        kv = (torch.zeros(1), torch.zeros(1), torch.zeros(1))
        with patch(
            "vllm_ascend.attention.sfa_v1.staged_sfa_connector_supports_sparse_load",
            return_value=False,
        ):
            out, index_layer, index_enabled = impl._cross_layer_kv_cache(
                "model.layers.0.self_attn.attn", kv
            )
        self.assertEqual(len(out), 3)
        self.assertIsNotNone(index_layer)
        self.assertFalse(index_enabled)

    def test_retrieve_propagates_route_frontiers(self):
        impl = self._impl()
        producer_event = object()
        state = SimpleNamespace(runtime=None, producer_event=producer_event)
        impl._staged_sfa_capture_state = state
        next_metadata = SimpleNamespace(
            req_ids=["r0", "r1"],
        )
        context = SimpleNamespace(
            staged_sfa_graph_key=object(),
            staged_sfa_graph_dummy_run=False,
            staged_sfa_route=SimpleNamespace(frontiers=(100, 200)),
            attn_metadata={"model.layers.1.self_attn.attn": next_metadata},
        )
        attn_metadata = SimpleNamespace(
            decode_request_ids_compact=["r0", "r1"],
        )
        captured_wait = {}
        captured_boundary = {}

        def fake_wait(layer_name, **kwargs):
            captured_wait[layer_name] = kwargs

        def fake_boundary(metadata, req_ids, *, is_dummy_run, index_topk,
                          cached_tokens):
            captured_boundary["cached_tokens"] = cached_tokens
            return MagicMock()

        with patch(
            "vllm_ascend.attention.sfa_v1.wait_for_kv_layer_from_connector",
            side_effect=fake_wait,
        ), patch(
            "vllm_ascend.attention.sfa_v1._prepare_sfa_remap_boundary",
            side_effect=fake_boundary,
        ), patch(
            "vllm_ascend.attention.sfa_v1._lmcache_sparse_wait_sync_once_enabled",
            return_value=False,
        ):
            impl.cross_layer_lmcache_retrieve(
                "model.layers.0.self_attn.attn",
                "model.layers.1.self_attn.attn",
                torch.zeros((2, 8), dtype=torch.int32),
                torch.zeros(2, dtype=torch.int32),
                torch.zeros((2, 8), dtype=torch.int64),
                attn_metadata,
                context,
            )
        # The single-lookup fix must hand the route's tuple through.
        self.assertEqual(captured_boundary["cached_tokens"], (100, 200))
        # The producer event rides the selective wait as payload_event.
        wait_kwargs = captured_wait["model.layers.0.self_attn.attn"]
        self.assertIs(wait_kwargs["payload_event"], producer_event)

    def test_prompt_lens_rows_expand_per_token_row(self):
        from vllm_ascend.attention.sfa_v1 import _staged_prompt_lens_rows

        rows = _staged_prompt_lens_rows(np.array([50, 60]), 2)
        self.assertEqual(rows.tolist(), [50, 50, 60, 60])
        rows_q1 = _staged_prompt_lens_rows(np.array([50, 60]), 1)
        self.assertEqual(rows_q1.tolist(), [50, 60])

    def test_dummy_remap_boundaries_clamp_to_scratch(self):
        from vllm_ascend.worker.model_runner_v1 import (
            _staged_sfa_dummy_remap_boundaries,
        )

        boundaries = _staged_sfa_dummy_remap_boundaries(
            np.array([100, 10, 1]), 1, 8
        )
        self.assertEqual(boundaries.tolist(), [99, 9, 0])


class TestPayloadEventForwarding(unittest.TestCase):
    def test_wait_helper_forwards_payload_event(self):
        from vllm_ascend.attention import utils as attn_utils

        connector = MagicMock()
        sentinel = object()
        with patch.object(
            attn_utils, "has_kv_transfer_group", return_value=True
        ), patch.object(
            attn_utils, "is_v1_kv_transfer_group", return_value=True
        ), patch.object(
            attn_utils, "get_kv_transfer_group", return_value=connector
        ):
            attn_utils.wait_for_kv_layer_from_connector(
                "layer",
                selected_tokens=torch.zeros(1),
                request_ids=["r0"],
                payload_event=sentinel,
            )
        kwargs = connector.wait_for_layer_load.call_args.kwargs
        self.assertIs(kwargs["payload_event"], sentinel)


class TestStagedConfigPredicate(unittest.TestCase):
    def _vllm_config(self, *, use_mla=True, dp=1, spec_tokens=0,
                     max_seqs=8, max_tokens=16):
        hf_text = SimpleNamespace(
            index_topk=8,
        )
        return SimpleNamespace(
            model_config=SimpleNamespace(
                use_mla=use_mla,
                hf_text_config=hf_text,
                hf_config=SimpleNamespace(),
            ),
            parallel_config=SimpleNamespace(data_parallel_size=dp),
            scheduler_config=SimpleNamespace(
                max_num_seqs=max_seqs,
                max_num_batched_tokens=max_tokens,
            ),
            speculative_config=(
                SimpleNamespace(num_speculative_tokens=spec_tokens)
                if spec_tokens
                else None
            ),
        )

    def test_not_configured_when_env_off(self):
        with patch(
            "vllm_ascend.envs.VLLM_ASCEND_SFA_STAGED_GRAPH", False
        ):
            self.assertFalse(
                staged_sfa_graph_configured(self._vllm_config())
            )
            self.assertEqual(
                staged_sfa_graph_capture_sizes(self._vllm_config()), ()
            )

    def test_not_configured_for_non_mla_or_multi_dp(self):
        with patch(
            "vllm_ascend.envs.VLLM_ASCEND_SFA_STAGED_GRAPH", True
        ):
            self.assertFalse(
                staged_sfa_graph_configured(
                    self._vllm_config(use_mla=False)
                )
            )
            self.assertFalse(
                staged_sfa_graph_configured(self._vllm_config(dp=2))
            )

    def test_capture_sizes_validated(self):
        with patch(
            "vllm_ascend.envs.VLLM_ASCEND_SFA_STAGED_GRAPH", True
        ), patch(
            "vllm_ascend.envs.VLLM_ASCEND_SFA_STAGED_GRAPH_CAPTURE_SIZES",
            "4,8",
        ):
            self.assertEqual(
                staged_sfa_graph_capture_sizes(self._vllm_config()),
                (4, 8),
            )
        with patch(
            "vllm_ascend.envs.VLLM_ASCEND_SFA_STAGED_GRAPH", True
        ), patch(
            "vllm_ascend.envs.VLLM_ASCEND_SFA_STAGED_GRAPH_CAPTURE_SIZES",
            "4,5",
        ), self.assertRaisesRegex(ValueError, "divisible"):
            # query width 2 (MTP1): 5 is not divisible.
            staged_sfa_graph_capture_sizes(
                self._vllm_config(spec_tokens=1)
            )
        with patch(
            "vllm_ascend.envs.VLLM_ASCEND_SFA_STAGED_GRAPH", True
        ), patch(
            "vllm_ascend.envs.VLLM_ASCEND_SFA_STAGED_GRAPH_CAPTURE_SIZES",
            "32",
        ), self.assertRaisesRegex(ValueError, "exceed"):
            staged_sfa_graph_capture_sizes(self._vllm_config())


class TestPlatformAndFenceInvariants(unittest.TestCase):
    def _source(self, relative):
        import vllm_ascend

        return (Path(vllm_ascend.__file__).parent / relative).read_text()

    def test_platform_splits_on_retrieve_when_staged(self):
        source = self._source("platform.py")
        self.assertIn("staged_sfa_graph_capture_sizes(vllm_config)", source)
        self.assertIn('"vllm::sfa_lmcache_retrieve"', source)

    def test_acl_graph_fences_first_staged_island(self):
        source = self._source("compilation/acl_graph.py")
        self.assertIn("staged_sfa_replay_fenced", source)
        self.assertIn("staged_piecewise_replay", source)

    def test_forward_context_resets_fence_each_forward(self):
        source = self._source("ascend_forward_context.py")
        self.assertIn(
            "forward_context.staged_sfa_replay_fenced = False", source
        )

    def test_impl_eligibility_rejects_c8_mlapo_dsa_cp(self):
        source = inspect.getsource(
            AscendSFAImpl.cross_layer_graph_pre
        )
        self.assertIn("enable_sparse_sfa_c8", source)
        self.assertIn("enable_mlapo", source)
        self.assertIn("enable_dsa_cp", source)


if __name__ == "__main__":
    unittest.main()
