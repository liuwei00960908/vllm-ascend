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

    def setUp(self):
        model_runner_v1 = _load_model_runner()
        # enable_sp reads the real parallel/additional config; the routing
        # tests exercise unrelated gates, so pin it off.
        sp_patcher = patch.object(
            model_runner_v1, "enable_sp", return_value=False
        )
        sp_patcher.start()
        self.addCleanup(sp_patcher.stop)

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

    def test_dp_greater_than_one_no_longer_fails_closed_locally(self):
        # DP divergence is resolved by the route row of the DP metadata
        # all-reduce, not by a local verdict: with metadata reporting an
        # eligible sparse route, a dp=2 rank still authorizes STAGED and
        # leaves the downgrade to the collective. Provenance: fork
        # model_runner_v1.py:2980+ (no local DP gate).
        from vllm_ascend.attention import utils as attn_utils

        model_runner_v1 = _load_model_runner()
        runner = _route_runner(
            parallel_config=SimpleNamespace(data_parallel_size=2)
        )
        with (
            patch.object(
                model_runner_v1, "enable_sp", return_value=False
            ),
            # _staged_sfa_local_route imports the helper from its source
            # module at call time, so the patch target is the source
            # module, not model_runner_v1.
            patch.object(
                attn_utils,
                "staged_sfa_metadata_sparse_route",
                return_value=(
                    StagedSFARouteReason.ELIGIBLE,
                    (0, 0, 0, 0),
                    None,
                ),
            ),
        ):
            decision = self._route(runner)
        self.assertEqual(decision.action, StagedSFARouteAction.STAGED)

    def test_sequence_parallelism_fails_closed(self):
        model_runner_v1 = _load_model_runner()
        runner = _route_runner()
        with patch.object(
            model_runner_v1, "enable_sp", return_value=True
        ):
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

    def test_init_consumes_shared_predicate(self):
        # The runner must never gate on the raw env flag: platform, runner,
        # and impl all key off staged_sfa_graph_configured/_capture_sizes.
        source = self._method_source("__init__")
        self.assertIn("staged_sfa_graph_configured", source)
        self.assertIn("staged_sfa_graph_capture_sizes", source)
        self.assertIn("staged_sfa_graph_configuration_errors", source)

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
        # The capture lifecycle gates on the shared predicate, never the
        # raw env flag.
        self.assertIn("staged_sfa_graph_configured", source)

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
    def _vllm_config(self, **overrides):
        from vllm.config import CUDAGraphMode

        config = SimpleNamespace(
            model_config=SimpleNamespace(
                use_mla=True,
                hf_text_config=SimpleNamespace(index_topk=8),
                hf_config=SimpleNamespace(),
            ),
            parallel_config=SimpleNamespace(
                data_parallel_size=1,
                pipeline_parallel_size=1,
                prefill_context_parallel_size=1,
                decode_context_parallel_size=1,
            ),
            scheduler_config=SimpleNamespace(
                max_num_seqs=64,
                max_num_batched_tokens=256,
            ),
            speculative_config=None,
            lora_config=None,
            kv_transfer_config=SimpleNamespace(),
            compilation_config=SimpleNamespace(
                cudagraph_mode=CUDAGraphMode.PIECEWISE
            ),
        )
        for key, value in overrides.items():
            setattr(config, key, value)
        return config

    def _env(self, **overrides):
        env = {
            "VLLM_ASCEND_SFA_STAGED_GRAPH": True,
            "VLLM_ASCEND_SFA_STAGED_GRAPH_CAPTURE_SIZES": "4,8",
            "VLLM_ASCEND_DSA_UNBUNDLE": True,
            "VLLM_ASCEND_DSA_TWO_GROUPS": True,
            "VLLM_ASCEND_DSA_SHRINK_LATENT": 2,
        }
        env.update(overrides)
        return env

    def _configured(self, config=None, **env_overrides):
        from contextlib import ExitStack

        with ExitStack() as stack:
            for name, value in self._env(**env_overrides).items():
                stack.enter_context(
                    patch(f"vllm_ascend.envs.{name}", value)
                )
            return staged_sfa_graph_configured(
                config if config is not None else self._vllm_config()
            )

    def test_configured_on_the_production_shape(self):
        self.assertTrue(self._configured())

    def test_not_configured_when_env_off(self):
        self.assertFalse(
            self._configured(VLLM_ASCEND_SFA_STAGED_GRAPH=False)
        )
        self.assertEqual(
            self._sizes_with_env(VLLM_ASCEND_SFA_STAGED_GRAPH=False),
            (),
        )

    def _sizes_with_env(self, raw="4,8", config=None, **env_overrides):
        from contextlib import ExitStack

        with ExitStack() as stack:
            for name, value in self._env(
                VLLM_ASCEND_SFA_STAGED_GRAPH_CAPTURE_SIZES=raw,
                **env_overrides,
            ).items():
                stack.enter_context(
                    patch(f"vllm_ascend.envs.{name}", value)
                )
            return staged_sfa_graph_capture_sizes(
                config if config is not None else self._vllm_config()
            )

    def test_reasons_matrix(self):
        from contextlib import ExitStack

        from vllm.config import CUDAGraphMode

        from vllm_ascend.utils import (
            StagedSFAConfigReason,
            staged_sfa_graph_configuration_reasons,
        )

        def reasons(config, **env_overrides):
            with ExitStack() as stack:
                for name, value in self._env(**env_overrides).items():
                    stack.enter_context(
                        patch(f"vllm_ascend.envs.{name}", value)
                    )
                return set(staged_sfa_graph_configuration_reasons(config))

        self.assertIn(
            StagedSFAConfigReason.DSA_UNBUNDLE,
            reasons(self._vllm_config(), VLLM_ASCEND_DSA_UNBUNDLE=False),
        )
        self.assertIn(
            StagedSFAConfigReason.DSA_TWO_GROUPS,
            reasons(self._vllm_config(), VLLM_ASCEND_DSA_TWO_GROUPS=False),
        )
        self.assertIn(
            StagedSFAConfigReason.SHRINK_LATENT,
            reasons(
                self._vllm_config(), VLLM_ASCEND_DSA_SHRINK_LATENT=0
            ),
        )
        non_piecewise = self._vllm_config()
        non_piecewise.compilation_config = SimpleNamespace(
            cudagraph_mode=CUDAGraphMode.FULL
        )
        self.assertIn(
            StagedSFAConfigReason.CUDAGRAPH_MODE, reasons(non_piecewise)
        )
        self.assertIn(
            StagedSFAConfigReason.MODEL_NOT_MLA,
            reasons(self._vllm_config(use_mla=False)),
        )
        self.assertIn(
            StagedSFAConfigReason.CONNECTOR_MISSING,
            reasons(self._vllm_config(kv_transfer_config=None)),
        )
        eagle = self._vllm_config(
            speculative_config=SimpleNamespace(
                num_speculative_tokens=2, method="eagle"
            )
        )
        self.assertIn(
            StagedSFAConfigReason.SPECULATIVE_DECODE, reasons(eagle)
        )
        mtp = self._vllm_config(
            speculative_config=SimpleNamespace(
                num_speculative_tokens=1, method="mtp"
            )
        )
        self.assertNotIn(
            StagedSFAConfigReason.SPECULATIVE_DECODE, reasons(mtp)
        )
        self.assertIn(
            StagedSFAConfigReason.LORA,
            reasons(self._vllm_config(lora_config=object())),
        )
        # Plain DP2 coordinates the staged verdict through the DP route
        # sync, so it stays allowed; only the external-launcher backend
        # (a DP-global rank/world the route chain cannot qualify) is
        # rejected. Provenance: fork utils.py:562-575.
        plain_dp2 = reasons(
            self._vllm_config(
                parallel_config=SimpleNamespace(
                    data_parallel_size=2,
                    pipeline_parallel_size=1,
                    prefill_context_parallel_size=1,
                    decode_context_parallel_size=1,
                )
            )
        )
        self.assertNotIn(StagedSFAConfigReason.DATA_PARALLEL, plain_dp2)
        self.assertIn(
            StagedSFAConfigReason.DATA_PARALLEL,
            reasons(
                self._vllm_config(
                    parallel_config=SimpleNamespace(
                        data_parallel_size=2,
                        pipeline_parallel_size=1,
                        prefill_context_parallel_size=1,
                        decode_context_parallel_size=1,
                        distributed_executor_backend="external_launcher",
                    )
                )
            ),
        )
        self.assertIn(
            StagedSFAConfigReason.PIPELINE_PARALLEL,
            reasons(
                self._vllm_config(
                    parallel_config=SimpleNamespace(
                        data_parallel_size=1,
                        pipeline_parallel_size=2,
                        prefill_context_parallel_size=1,
                        decode_context_parallel_size=1,
                    )
                )
            ),
        )
        self.assertIn(
            StagedSFAConfigReason.CONTEXT_PARALLEL,
            reasons(
                self._vllm_config(
                    parallel_config=SimpleNamespace(
                        data_parallel_size=1,
                        pipeline_parallel_size=1,
                        prefill_context_parallel_size=2,
                        decode_context_parallel_size=1,
                    )
                )
            ),
        )
        # MTP keeps the production shape eligible.
        self.assertEqual(reasons(mtp), set())

    def test_capture_sizes_request_units(self):
        # Width 1: env request counts pass through as token capacities.
        self.assertEqual(self._sizes_with_env("4,8"), (4, 8))
        # Width 2 (MTP1): request counts scale by the query width.
        spec = SimpleNamespace(num_speculative_tokens=1, method="mtp")
        self.assertEqual(
            self._sizes_with_env(
                "4,5",
                config=self._vllm_config(speculative_config=spec),
            ),
            (8, 10),
        )

    def test_capture_sizes_bounds(self):
        spec = SimpleNamespace(num_speculative_tokens=1, method="mtp")
        with self.assertRaisesRegex(ValueError, "exceed"):
            # 100 requests > max_num_seqs=64.
            self._sizes_with_env(
                "100", config=self._vllm_config(speculative_config=spec)
            )
        with self.assertRaisesRegex(ValueError, "exceed"):
            # 200 requests x 2 = 400 tokens > max_num_batched_tokens=256.
            self._sizes_with_env(
                "200", config=self._vllm_config(speculative_config=spec)
            )


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
        self.assertIn("two-group indexer tables", source)
        self.assertIn("block-table logical capacity", source)


    def test_impl_eligibility_requires_two_group_tables(self):
        source = inspect.getsource(
            AscendSFAImpl.cross_layer_graph_pre
        )
        self.assertIn("indexer_block_table", source)
        self.assertIn("indexer_slot_mapping", source)


class TestStagedFixedLayout(unittest.TestCase):
    """Executable matrix for the padded fixed-layout predicate (P0 fix)."""

    def test_exact_capacity_q1(self):
        from vllm_ascend.attention.sfa_v1 import _staged_fixed_layout

        self.assertTrue(_staged_fixed_layout(4, 4, 4, 4, 1, True))

    def test_padded_live_batch_q1(self):
        # 3 active requests padded to capacity 4 (the common live case the
        # historical num_decode_rows == num_reqs * width check rejected).
        from vllm_ascend.attention.sfa_v1 import _staged_fixed_layout

        self.assertTrue(_staged_fixed_layout(4, 4, 3, 3, 1, True))

    def test_exact_capacity_mtp1(self):
        from vllm_ascend.attention.sfa_v1 import _staged_fixed_layout

        # 4 requests x 2 rows = 8 tokens.
        self.assertTrue(_staged_fixed_layout(8, 4, 8, 4, 2, True))

    def test_padded_live_batch_mtp1(self):
        from vllm_ascend.attention.sfa_v1 import _staged_fixed_layout

        # 3 active requests on capacity 4: 6 active rows, 8 padded tokens.
        self.assertTrue(_staged_fixed_layout(8, 4, 6, 3, 2, True))

    def test_mixed_batch_rejected(self):
        from vllm_ascend.attention.sfa_v1 import _staged_fixed_layout

        # A decode row is missing (mixed prefill/decode): not fixed layout.
        self.assertFalse(_staged_fixed_layout(4, 4, 3, 4, 1, True))

    def test_non_multiple_token_view_rejected(self):
        from vllm_ascend.attention.sfa_v1 import _staged_fixed_layout

        self.assertFalse(_staged_fixed_layout(5, 4, 4, 4, 1, True))

    def test_prefix_hit_prefill_tail_rejected(self):
        # log32 regression: a cached-prompt tail scheduling exactly
        # width tokens satisfies the token-view equation coincidentally,
        # but carries zero decode rows — must never enter the staged
        # branch (the prompt-row expansion crashes on the shape mismatch).
        from vllm_ascend.attention.sfa_v1 import _staged_fixed_layout

        # 1 request, 2 remaining prompt tokens, width 2: decode rows 0
        # while ALL-active counting (1) rejects 0 != 1 * 2.
        self.assertFalse(
            _staged_fixed_layout(2, 1, 0, 1, 2, False)
        )

    def test_mixed_prefill_decode_coincidence_rejected(self):
        # 1 decode request (2 rows) + 1 prefill request (2 prompt tokens):
        # token view 4 == 2 * 2 holds, decode rows 2 == decode-reqs * 2
        # would hold under the weaker count — the all-active count (2)
        # rejects 2 != 2 * 2.
        from vllm_ascend.attention.sfa_v1 import _staged_fixed_layout

        self.assertFalse(_staged_fixed_layout(4, 2, 2, 2, 2, True))

    def test_uncomputed_prompt_rejected(self):
        # Fork's third condition: even a shape-perfect decode batch is
        # rejected while any prompt is not fully computed.
        from vllm_ascend.attention.sfa_v1 import _staged_fixed_layout

        self.assertFalse(_staged_fixed_layout(4, 4, 4, 4, 1, False))


class TestIndexerProductionCallSite(unittest.TestCase):
    """The staged pre must run the production top-k (weights + q_c query),
    not the historical key-projection-as-query shortcut."""

    def _impl(self):
        impl = SimpleNamespace(
            fused_qkv_a_proj=lambda h: (torch.zeros(4, 12),),
            q_a_layernorm=lambda t: t,
            indexer_select_pre_process=lambda **kw: (
                torch.zeros(4, 1, 8),
                None,
            ),
            exec_kv=MagicMock(),
            _q_proj_and_k_up_proj=lambda q: (
                torch.zeros(4, 2, 4),
                torch.zeros(4, 2, 2),
            ),
            rope_single=lambda q, cos, sin: q,
            indexer_select_post_process=MagicMock(
                return_value=torch.zeros(4, 1, 8, dtype=torch.int32)
            ),
            dsa_index_topk=2048,
            q_lora_rank=6,
            kv_lora_rank=4,
            qk_rope_head_dim=2,
            vllm_config=SimpleNamespace(
                cache_config=SimpleNamespace(block_size=128)
            ),
        )
        return impl

    def test_pre_compute_uses_production_topk(self):
        import vllm_ascend.attention.sfa_v1 as sfa_mod

        impl = self._impl()
        hidden = torch.zeros(4, 16)
        kv_cache = (
            torch.zeros(2, 4, 1, 4),
            torch.zeros(2, 4, 1, 2),
            torch.zeros(2, 4, 1, 8),
        )
        indexer_block_table = torch.zeros((4, 10), dtype=torch.int64)

        def fake_prepare(topk, boundary, *args, **kwargs):
            # Pure stand-in: return shapes matching the dispatch contract.
            return (
                topk,
                torch.zeros((4, 8), dtype=torch.int32),
                torch.zeros((4, 8), dtype=torch.int32),
                torch.zeros((4, 8), dtype=torch.int64),
            )

        with patch.object(
            sfa_mod.torch_npu, "npu_scatter_nd_update_", MagicMock()
        ), patch(
            "vllm_ascend.distributed.kv_transfer.sparse_offload."
            "prepare_sparse_indices.prepare_sparse_indices",
            side_effect=fake_prepare,
        ):
            outputs = AscendSFAImpl._cross_layer_pre_compute(
                impl,
                hidden,
                kv_cache[0],
                kv_cache[1],
                kv_cache[2],
                torch.zeros(4, 2),
                torch.zeros(4, 2),
                torch.zeros(4, dtype=torch.int32),
                torch.zeros(4, dtype=torch.int32),
                torch.zeros(4, dtype=torch.int32),
                torch.zeros(4, dtype=torch.int32),
                indexer_block_table,
                torch.zeros(4, dtype=torch.int32),
                torch.zeros(4, dtype=torch.int32),
                indexer_block_table,
                torch.zeros((4, 8), dtype=torch.int32),
                torch.zeros((4, 8), dtype=torch.int32),
                torch.zeros(4, dtype=torch.int32),
                torch.zeros((4, 8), dtype=torch.int64),
                SimpleNamespace(),
            )
        call = impl.indexer_select_post_process.call_args
        self.assertEqual(call.kwargs["x"].data_ptr(), hidden.data_ptr())
        # q_c is the layer-normed compressed query, NOT the key projection.
        self.assertEqual(call.kwargs["q_c"].shape, (4, 6))
        self.assertIsNone(call.kwargs["attn_metadata"])
        self.assertIs(
            call.kwargs["indexer_block_table_override"], indexer_block_table
        )
        self.assertEqual(call.kwargs["sparse_count"], 2048)
        # Six bridge outputs came back from the pre-compute.
        self.assertEqual(len(outputs), 6)

    def test_dead_helper_fails_loudly(self):
        impl = AscendSFAImpl.__new__(AscendSFAImpl)
        with self.assertRaises(NotImplementedError):
            impl._indexer_topk_for_staged(
                torch.zeros(1), torch.zeros(1), (), None, None,
                None, None, None,
            )

    def test_impl_post_process_substitutes_override_table(self):
        import vllm_ascend.attention.sfa_v1 as sfa_mod

        impl = SimpleNamespace(
            has_indexer=True,
            wk_weights_proj=lambda x: (torch.zeros(4, 16),),
            wq_b=lambda q: (torch.zeros(4, 64),),
            n_head=2,
            head_dim=8,
            qk_rope_head_dim=4,
            is_rope_neox_style=True,
            enable_sparse_li_c8=False,
            use_torch_npu_lightning_indexer=False,
            layer_name="l0",
        )
        override = torch.zeros((4, 10), dtype=torch.int64)
        with patch.object(sfa_mod, "HAS_TRITON", False), patch.object(
            sfa_mod.torch_npu,
            "npu_rotary_mul",
            lambda pe, cos, sin: pe,
        ), patch.object(
            sfa_mod, "record_attention_compute_start", lambda: None
        ), patch.object(
            sfa_mod.DeviceOperator,
            "indexer_select_post_process",
            return_value=torch.zeros(4, 1, 8, dtype=torch.int32),
        ) as device_op:
            AscendSFAImpl.indexer_select_post_process(
                impl,
                x=torch.zeros(4, 16),
                q_c=torch.zeros(4, 6),
                kv_cache=(torch.zeros(1), torch.zeros(1), torch.zeros(1)),
                attn_metadata=None,
                cos=torch.zeros(4, 4),
                sin=torch.zeros(4, 4),
                actual_seq_lengths_query=torch.zeros(4, dtype=torch.int32),
                actual_seq_lengths_key=torch.zeros(4, dtype=torch.int32),
                indexer_block_table_override=override,
                sparse_count=2048,
            )
        device_op_kwargs = device_op.call_args
        metadata = device_op_kwargs.args[6]
        self.assertIs(metadata.indexer_block_table, override)
        self.assertIs(metadata.block_table, override)
        self.assertEqual(device_op_kwargs.kwargs["sparse_count"], 2048)


class TestSealStagedEntries(unittest.TestCase):
    """Executable seal rework: minimum island count + per-key completeness."""

    def _key(self, capacity):
        from vllm_ascend.ascend_forward_context import StagedSFAGraphKey

        return StagedSFAGraphKey.exact_q1(capacity)

    def _entry(self, complete=True):
        entry = SimpleNamespace(
            aclgraph=object() if complete else None,
            input_addresses=[1] if complete else None,
        )
        return entry

    def test_extra_islands_tolerated(self):
        from vllm_ascend.compilation.acl_graph import ACLGraphWrapper

        keys = (self._key(1), self._key(2))
        # Three islands (layers+1 = 3 minimum) plus one EXTRA island from a
        # kv-cache-update split — tolerated, every island complete.
        wrappers = {
            SimpleNamespace(
                concrete_aclgraph_entries={
                    key: self._entry() for key in keys
                }
            )
            for _ in range(4)
        }
        with patch(
            "vllm_ascend.compilation.acl_graph._acl_graph_wrappers",
            wrappers,
        ):
            count = ACLGraphWrapper.seal_staged_entries(keys, 3)
        self.assertEqual(count, 8)

    def test_legacy_entries_coexist(self):
        from vllm_ascend.compilation.acl_graph import ACLGraphWrapper

        keys = (self._key(1),)
        legacy = SimpleNamespace(num_tokens=8)
        wrappers = {
            SimpleNamespace(
                concrete_aclgraph_entries={
                    keys[0]: self._entry(),
                    legacy: self._entry(),
                }
            )
        }
        with patch(
            "vllm_ascend.compilation.acl_graph._acl_graph_wrappers",
            wrappers,
        ):
            # Legacy BatchDescriptor coexistence is no longer "unexpected".
            self.assertEqual(
                ACLGraphWrapper.seal_staged_entries(keys, 1), 1
            )

    def test_missing_key_still_fails(self):
        from vllm_ascend.compilation.acl_graph import ACLGraphWrapper

        keys = (self._key(1), self._key(2))
        wrappers = {
            SimpleNamespace(
                concrete_aclgraph_entries={keys[0]: self._entry()}
            )
        }
        with patch(
            "vllm_ascend.compilation.acl_graph._acl_graph_wrappers",
            wrappers,
        ), self.assertRaisesRegex(RuntimeError, "incomplete"):
            ACLGraphWrapper.seal_staged_entries(keys, 1)

    def test_too_few_islands_fails(self):
        from vllm_ascend.compilation.acl_graph import ACLGraphWrapper

        keys = (self._key(1),)
        wrappers = {
            SimpleNamespace(
                concrete_aclgraph_entries={keys[0]: self._entry()}
            )
        }
        with patch(
            "vllm_ascend.compilation.acl_graph._acl_graph_wrappers",
            wrappers,
        ), self.assertRaisesRegex(RuntimeError, "expected_at_least"):
            ACLGraphWrapper.seal_staged_entries(keys, 2)


class TestDPRouteSync(unittest.TestCase):
    """The route row of the DP metadata all-reduce converges on the
    strongest downgrade across ranks, landed on the runner state.

    The return ABI stays the v0.23 3-tuple; the synced verdict is read
    from _staged_sfa_dp_route_action. Provenance: fork
    model_runner_v1.py:2435-2470 (row layout), :2587 (verdict landing),
    :3099-3108 (live-route merge).
    """

    def _runner(self, dp_rank=0):
        model_runner_v1 = _load_model_runner()
        runner = model_runner_v1.NPUModelRunner.__new__(
            model_runner_v1.NPUModelRunner
        )
        runner.dp_size = 2
        runner.dp_rank = dp_rank
        runner.vllm_config = SimpleNamespace()
        runner.ascend_config = SimpleNamespace(dp_allreduce_on_npu=False)
        runner._staged_sfa_dp_route_action = None
        return runner

    def _sync(self, runner, action, peer_action):
        from vllm.config import CUDAGraphMode

        model_runner_v1 = _load_model_runner()
        peer_route_index = list(StagedSFARouteAction).index(peer_action)

        def fake_all_reduce(tensor, group=None):
            # Emulate a SUM all-reduce over the two-rank group: this rank
            # already wrote its column; rank 1 contributes its values into
            # the other column. No row-level collapsing — the production
            # unpack keeps the per-row max/min semantics under test.
            tensor[0][1] += 4
            tensor[1][1] += int(CUDAGraphMode.PIECEWISE.value)
            if tensor.shape[0] > 2:
                tensor[2][1] += peer_route_index

        cpu_group = SimpleNamespace(cpu_group=object())
        with (
            patch.object(
                model_runner_v1,
                "should_skip_allreduce_across_dp_group",
                return_value=False,
            ),
            patch.object(
                model_runner_v1, "get_dp_group", return_value=cpu_group
            ),
            patch.object(
                model_runner_v1.dist, "all_reduce", side_effect=fake_all_reduce
            ),
        ):
            result = runner._sync_metadata_across_dp(
                num_tokens=4,
                cudagraph_mode=CUDAGraphMode.PIECEWISE,
                staged_sfa_route_action=action,
            )
        # The return ABI is the v0.23 3-tuple (spec-decode proposers
        # unpack it positionally).
        self.assertEqual(len(result), 3)
        return result, runner._staged_sfa_dp_route_action

    def test_sync_downgrades_to_strongest(self):
        runner = self._runner(dp_rank=0)
        _, landed = self._sync(
            runner,
            StagedSFARouteAction.STAGED,
            peer_action=StagedSFARouteAction.SAFE_NATIVE,
        )
        self.assertIs(landed, StagedSFARouteAction.SAFE_NATIVE)

    def test_sync_keeps_staged_when_all_ranks_agree(self):
        runner = self._runner(dp_rank=0)
        _, landed = self._sync(
            runner,
            StagedSFARouteAction.STAGED,
            peer_action=StagedSFARouteAction.STAGED,
        )
        self.assertIs(landed, StagedSFARouteAction.STAGED)

    def test_sync_without_route_row_leaves_state_alone(self):
        from vllm.config import CUDAGraphMode

        model_runner_v1 = _load_model_runner()
        runner = self._runner(dp_rank=0)

        def fake_all_reduce(tensor, group=None):
            tensor[0][1] += 4
            tensor[1][1] += int(CUDAGraphMode.PIECEWISE.value)

        with (
            patch.object(
                model_runner_v1,
                "should_skip_allreduce_across_dp_group",
                return_value=False,
            ),
            patch.object(
                model_runner_v1,
                "get_dp_group",
                return_value=SimpleNamespace(cpu_group=object()),
            ),
            patch.object(
                model_runner_v1.dist, "all_reduce", side_effect=fake_all_reduce
            ),
        ):
            result = runner._sync_metadata_across_dp(
                num_tokens=4,
                cudagraph_mode=CUDAGraphMode.PIECEWISE,
            )
        self.assertEqual(len(result), 3)
        self.assertIsNone(runner._staged_sfa_dp_route_action)

    def test_live_route_honors_dp_downgrade(self):
        from vllm_ascend.utils import StagedSFARouteDecision

        runner = self._runner(dp_rank=0)
        local = StagedSFARouteDecision(
            StagedSFARouteAction.STAGED,
            StagedSFARouteReason.ELIGIBLE,
            frontiers=(0,),
        )
        decision = runner._staged_sfa_live_route(
            local_route=local,
            dp_route_action=StagedSFARouteAction.SAFE_NATIVE,
            cudagraph_mode=None,
            batch_descriptor=SimpleNamespace(),
            num_tokens_unpadded=4,
            num_tokens_padded=4,
            num_reqs=4,
            should_ubatch=False,
        )
        self.assertIs(decision.action, StagedSFARouteAction.SAFE_NATIVE)
        self.assertEqual(
            decision.reason, StagedSFARouteReason.RUNTIME_PARALLELISM
        )

    def test_live_route_identity_when_actions_match(self):
        from vllm_ascend.utils import StagedSFARouteDecision

        runner = self._runner(dp_rank=0)
        local = StagedSFARouteDecision(
            StagedSFARouteAction.SAFE_NATIVE,
            StagedSFARouteReason.NOT_DECODE,
        )
        decision = runner._staged_sfa_live_route(
            local_route=local,
            dp_route_action=StagedSFARouteAction.SAFE_NATIVE,
            cudagraph_mode=None,
            batch_descriptor=SimpleNamespace(),
            num_tokens_unpadded=4,
            num_tokens_padded=4,
            num_reqs=4,
            should_ubatch=False,
        )
        self.assertIs(decision, local)


class TestDPIdleDummyStagedGuard(unittest.TestCase):
    """A DP-padded idle dummy must not engage the staged bootstrap unless
    its row geometry fully fills the captured capacity (log53: the staged
    bootstrap raised "boundary storage is unavailable" on the idle rank).

    Provenance: fork model_runner_v1.py:3426-3434/:3499-3514 (dp_idle STAGED
    declaration + eager fallback), plus the full-capacity requirement the
    P9-simplified builder branch imposes.
    """

    def _runner(self, **overrides):
        model_runner_v1 = _load_model_runner()
        runner = model_runner_v1.NPUModelRunner.__new__(
            model_runner_v1.NPUModelRunner
        )
        defaults = dict(
            _staged_sfa_graph_capture_sizes=(10, 20),
            speculative_config=SimpleNamespace(num_speculative_tokens=1),
            dp_size=4,
            _staged_sfa_dp_route_action=None,
        )
        defaults.update(overrides)
        for key, value in defaults.items():
            setattr(runner, key, value)
        return runner

    def _dummy_size(self, runner, *, num_reqs, batch_size, unpadded):
        from vllm.config import CUDAGraphMode
        from vllm.forward_context import BatchDescriptor

        return runner._staged_sfa_dummy_batch_size(
            is_profile=False,
            cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE,
            allow_eager=False,
            num_active_loras=0,
            num_tokens_unpadded=unpadded,
            num_tokens_padded=batch_size,
            num_reqs=num_reqs,
            num_scheduled_tokens=np.full(num_reqs, 2, dtype=np.int32),
            batch_descriptor=BatchDescriptor(num_tokens=batch_size),
        )

    def test_partial_fill_idle_dummy_rejected(self):
        # DP padding raised a 1-request idle dummy (2 tokens) to capacity
        # 10: the row geometry no longer fills the capacity, so the staged
        # dummy must be refused (the builder cannot attach the boundary).
        runner = self._runner()
        self.assertIsNone(
            self._dummy_size(runner, num_reqs=1, batch_size=10, unpadded=2)
        )

    def test_full_fill_capture_dummy_accepted(self):
        # Capture warmups carry num_tokens == capacity: 5 requests x 2.
        runner = self._runner()
        self.assertEqual(
            self._dummy_size(runner, num_reqs=5, batch_size=10, unpadded=10),
            10,
        )

    def test_fallback_none_when_staged_or_undeclared(self):
        runner = self._runner()
        self.assertIsNone(
            runner._staged_sfa_dummy_fallback_action(None, False)
        )
        self.assertIsNone(
            runner._staged_sfa_dummy_fallback_action(
                StagedSFARouteAction.STAGED, True
            )
        )

    def test_fallback_safe_native_when_declared_but_not_staged(self):
        runner = self._runner()
        self.assertIs(
            runner._staged_sfa_dummy_fallback_action(
                StagedSFARouteAction.STAGED, False
            ),
            StagedSFARouteAction.SAFE_NATIVE,
        )

    def test_fallback_honors_stronger_dp_verdict(self):
        runner = self._runner(
            _staged_sfa_dp_route_action=StagedSFARouteAction.RECOMPUTE
        )
        self.assertIs(
            runner._staged_sfa_dummy_fallback_action(
                StagedSFARouteAction.STAGED, False
            ),
            StagedSFARouteAction.RECOMPUTE,
        )


if __name__ == "__main__":
    unittest.main()
