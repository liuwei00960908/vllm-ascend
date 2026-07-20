import unittest
from contextlib import nullcontext
from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from vllm.config import CUDAGraphMode
from vllm.forward_context import BatchDescriptor
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec, KVCacheTensor

import vllm_ascend.worker.model_runner_v1 as model_runner_module
from vllm_ascend.ascend_forward_context import (
    STAGED_SFA_SINGLETON_GRAPH_KEY,
    StagedSFAGraphKey,
    StagedSFAQueryProfile,
)
from vllm_ascend.worker.block_table import MultiGroupBlockTable
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner


class TestNPUModelRunnerKVCache(unittest.TestCase):
    def _build_runner(self):
        runner = NPUModelRunner.__new__(NPUModelRunner)
        runner.device = torch.device("cpu")
        runner.use_sparse = False
        runner.dsa_shared_pool = False
        runner.dsa_unbundle = False
        runner.use_sparse_c8_indexer = False
        runner.use_hybrid_blocks = False
        runner.hybrid_with_attn_and_mamba = False
        runner.runner_only_attn_layers = set()
        runner.is_kv_consumer = False
        runner.vllm_config = MagicMock()
        runner.vllm_config.kv_transfer_config = None
        runner.model_config = MagicMock()
        runner.model_config.use_mla = True
        backend = MagicMock()
        backend.get_kv_cache_shape.side_effect = lambda num_blocks, block_size, num_kv_heads, head_size: (
            2,
            num_blocks,
            block_size,
            num_kv_heads,
            head_size,
        )
        runner.attn_backend = backend
        return runner

    def test_allocate_kv_cache_uses_layer_spec_for_draft_gqa(self):
        runner = self._build_runner()
        kv_cache_spec = FullAttentionSpec(
            block_size=16,
            num_kv_heads=8,
            head_size=64,
            head_size_v=64,
            dtype=torch.float16,
        )
        kv_cache_config = KVCacheConfig(
            num_blocks=2,
            kv_cache_tensors=[KVCacheTensor(size=kv_cache_spec.page_size_bytes * 2, shared_by=["draft_attn"])],
            kv_cache_groups=[KVCacheGroupSpec(layer_names=["draft_attn"], kv_cache_spec=kv_cache_spec)],
        )

        kv_cache_raw_tensors = runner._allocate_kv_cache_tensors(kv_cache_config)
        k_cache_raw, v_cache_raw = kv_cache_raw_tensors["draft_attn"]

        self.assertEqual(k_cache_raw.numel(), kv_cache_spec.page_size_bytes)
        self.assertEqual(v_cache_raw.numel(), kv_cache_spec.page_size_bytes)

    def test_reshape_kv_cache_uses_layer_spec_for_draft_gqa(self):
        runner = self._build_runner()
        kv_cache_spec = FullAttentionSpec(
            block_size=16,
            num_kv_heads=8,
            head_size=64,
            head_size_v=64,
            dtype=torch.float16,
        )
        kv_cache_config = KVCacheConfig(
            num_blocks=2,
            kv_cache_tensors=[KVCacheTensor(size=kv_cache_spec.page_size_bytes * 2, shared_by=["draft_attn"])],
            kv_cache_groups=[KVCacheGroupSpec(layer_names=["draft_attn"], kv_cache_spec=kv_cache_spec)],
        )
        kv_cache_raw_tensors = runner._allocate_kv_cache_tensors(kv_cache_config)
        runner._kv_cache_spec_attn_group_iterator = lambda: [
            SimpleNamespace(
                kv_cache_spec=kv_cache_spec,
                backend=runner.attn_backend,
                layer_names=["draft_attn"],
            )
        ]

        kv_caches = runner._reshape_kv_cache_tensors(kv_cache_config, kv_cache_raw_tensors)
        k_cache, v_cache = kv_caches["draft_attn"]

        self.assertEqual(k_cache.shape, (2, 16, 8, 64))
        self.assertEqual(v_cache.shape, (2, 16, 8, 64))


class TestStagedSFAGraphKey(unittest.TestCase):
    def test_structural_dimensions_do_not_collide(self):
        base = STAGED_SFA_SINGLETON_GRAPH_KEY
        variants = (
            StagedSFAGraphKey(
                token_capacity=1,
                request_capacity=2,
                query_profile=StagedSFAQueryProfile.DECODE_Q1,
                max_query_len=1,
            ),
            StagedSFAGraphKey(
                token_capacity=1,
                request_capacity=1,
                query_profile=StagedSFAQueryProfile.SPEC_FIXED,
                max_query_len=1,
            ),
            StagedSFAGraphKey(
                token_capacity=1,
                request_capacity=1,
                query_profile=StagedSFAQueryProfile.DECODE_Q1,
                max_query_len=2,
            ),
        )
        self.assertEqual(
            len({base, StagedSFAGraphKey(**base.__dict__)}),
            1,
        )
        self.assertTrue(all(variant != base for variant in variants))
        self.assertEqual(len(set(variants)), len(variants))

    def test_key_is_frozen(self):
        with self.assertRaises(FrozenInstanceError):
            STAGED_SFA_SINGLETON_GRAPH_KEY.token_capacity = 2

    def test_only_singleton_adapts_to_legacy_descriptor(self):
        descriptor = STAGED_SFA_SINGLETON_GRAPH_KEY.to_legacy_batch_descriptor()
        self.assertEqual(descriptor, BatchDescriptor(num_tokens=1))
        self.assertIsNone(descriptor.num_reqs)
        self.assertFalse(descriptor.uniform)
        self.assertFalse(descriptor.has_lora)

        batch_descriptor = StagedSFAGraphKey(
            token_capacity=2,
            request_capacity=2,
            query_profile=StagedSFAQueryProfile.DECODE_Q1,
            max_query_len=1,
        ).to_legacy_batch_descriptor()
        self.assertEqual(batch_descriptor, BatchDescriptor(num_tokens=2))

        invalid_keys = (
            StagedSFAGraphKey(
                token_capacity=2,
                request_capacity=1,
                query_profile=StagedSFAQueryProfile.DECODE_Q1,
                max_query_len=1,
            ),
            StagedSFAGraphKey(
                token_capacity=2,
                request_capacity=2,
                query_profile=StagedSFAQueryProfile.SPEC_FIXED,
                max_query_len=2,
            ),
        )
        for key in invalid_keys:
            with (
                self.subTest(key=key),
                self.assertRaises(NotImplementedError),
            ):
                key.to_legacy_batch_descriptor()


class TestStagedSFADummyBatch(unittest.TestCase):
    @staticmethod
    def _build_runner():
        runner = NPUModelRunner.__new__(NPUModelRunner)
        runner.vllm_config = object()
        runner.speculative_config = None
        runner.parallel_config = SimpleNamespace(data_parallel_size=1)
        return runner

    @staticmethod
    def _eligibility_kwargs(batch_size=4):
        return {
            "is_profile": False,
            "cudagraph_runtime_mode": CUDAGraphMode.PIECEWISE,
            "uniform_decode": False,
            "skip_eplb": True,
            "remove_lora": False,
            "num_active_loras": 0,
            "num_tokens_unpadded": batch_size,
            "num_tokens_padded": batch_size,
            "num_reqs": batch_size,
            "num_reqs_padded": batch_size,
            "num_scheduled_tokens": np.ones(batch_size, dtype=np.int32),
            "batch_descriptor": BatchDescriptor(num_tokens=batch_size),
            "num_tokens_across_dp": None,
        }

    def test_exact_q1_capture_sizes_are_staged(self):
        runner = self._build_runner()
        with (
            patch.object(
                model_runner_module,
                "staged_sfa_graph_configured",
                return_value=True,
            ),
            patch.object(
                model_runner_module,
                "staged_sfa_graph_capture_sizes",
                return_value=(1, 4),
            ),
        ):
            for runtime_mode in (
                CUDAGraphMode.NONE,
                CUDAGraphMode.PIECEWISE,
            ):
                kwargs = self._eligibility_kwargs()
                kwargs["cudagraph_runtime_mode"] = runtime_mode
                with self.subTest(runtime_mode=runtime_mode):
                    self.assertEqual(
                        runner._staged_sfa_dummy_batch_size(**kwargs),
                        4,
                    )

    def test_padded_non_q1_and_unsupported_batches_fall_back(self):
        runner = self._build_runner()
        cases = {
            "unsupported_size": {
                "num_tokens_unpadded": 2,
                "num_tokens_padded": 2,
                "num_reqs": 2,
                "num_reqs_padded": 2,
                "num_scheduled_tokens": np.ones(2, dtype=np.int32),
                "batch_descriptor": BatchDescriptor(num_tokens=2),
            },
            "token_padding": {
                "num_tokens_padded": 8,
                "batch_descriptor": BatchDescriptor(num_tokens=8),
            },
            "request_padding": {"num_reqs_padded": 8},
            "non_q1": {
                "num_scheduled_tokens": np.array([1, 1, 2, 0]),
            },
            "uniform_descriptor": {
                "batch_descriptor": BatchDescriptor(
                    num_tokens=4,
                    uniform=True,
                ),
            },
            "dp_padding": {"num_tokens_across_dp": torch.ones(1)},
            "profile": {"is_profile": True},
        }
        with (
            patch.object(
                model_runner_module,
                "staged_sfa_graph_configured",
                return_value=True,
            ),
            patch.object(
                model_runner_module,
                "staged_sfa_graph_capture_sizes",
                return_value=(1, 4),
            ),
        ):
            for case_name, overrides in cases.items():
                kwargs = self._eligibility_kwargs()
                kwargs.update(overrides)
                with self.subTest(case=case_name):
                    self.assertIsNone(runner._staged_sfa_dummy_batch_size(**kwargs))

    def test_live_key_requires_exact_unpadded_q1_before_mutation(self):
        runner = self._build_runner()
        kwargs = {
            "cudagraph_mode": CUDAGraphMode.PIECEWISE,
            "batch_descriptor": BatchDescriptor(num_tokens=4),
            "num_tokens_unpadded": 4,
            "num_tokens_padded": 4,
            "num_reqs": 4,
            "num_scheduled_tokens": np.ones(4, dtype=np.int32),
            "num_tokens_across_dp": None,
            "should_ubatch": False,
            "has_cascade_attention": False,
        }
        with (
            patch.object(
                model_runner_module,
                "staged_sfa_graph_configured",
                return_value=True,
            ),
            patch.object(
                model_runner_module,
                "staged_sfa_graph_capture_sizes",
                return_value=(1, 4),
            ),
        ):
            self.assertEqual(
                runner._staged_sfa_live_graph_key(**kwargs),
                StagedSFAGraphKey.exact_q1(4),
            )
            for name, overrides in {
                "padding": {
                    "num_tokens_padded": 8,
                    "batch_descriptor": BatchDescriptor(num_tokens=8),
                },
                "multi_token": {
                    "num_scheduled_tokens": np.array(
                        [1, 1, 2, 0],
                        dtype=np.int32,
                    ),
                },
                "ubatch": {"should_ubatch": True},
                "cascade": {"has_cascade_attention": True},
                "dp_padding": {
                    "num_tokens_across_dp": torch.ones(1),
                },
            }.items():
                rejected = dict(kwargs)
                rejected.update(overrides)
                with self.subTest(name=name):
                    self.assertIsNone(runner._staged_sfa_live_graph_key(**rejected))

    def test_native_q1_rows_have_unique_ids_and_query_starts(self):
        request_ids = NPUModelRunner._staged_sfa_dummy_request_ids(4)
        query_start_locs = NPUModelRunner._staged_sfa_q1_query_start_locs(
            4,
            dtype=np.dtype(np.int32),
        )

        self.assertEqual(
            request_ids,
            [
                "staged-sfa-graph-dummy-0",
                "staged-sfa-graph-dummy-1",
                "staged-sfa-graph-dummy-2",
                "staged-sfa-graph-dummy-3",
            ],
        )
        self.assertEqual(len(set(request_ids)), 4)
        np.testing.assert_array_equal(
            query_start_locs,
            np.arange(5, dtype=np.int32),
        )

    def test_dp_and_speculative_dummy_batches_fall_back(self):
        runner = self._build_runner()
        kwargs = self._eligibility_kwargs()
        with (
            patch.object(
                model_runner_module,
                "staged_sfa_graph_configured",
                return_value=True,
            ),
            patch.object(
                model_runner_module,
                "staged_sfa_graph_capture_sizes",
                return_value=(1, 4),
            ),
        ):
            runner.parallel_config.data_parallel_size = 2
            self.assertIsNone(runner._staged_sfa_dummy_batch_size(**kwargs))

            runner.parallel_config.data_parallel_size = 1
            runner.speculative_config = SimpleNamespace(method="mtp")
            self.assertIsNone(runner._staged_sfa_dummy_batch_size(**kwargs))

    def test_two_group_dummy_rows_use_noncolliding_physical_slots(self):
        runner = self._build_runner()
        runner.kv_cache_config = SimpleNamespace(
            num_blocks=8,
            num_blocks_per_group=[8, 8],
        )
        runner.input_batch = SimpleNamespace(
            block_table=MultiGroupBlockTable(
                max_num_reqs=4,
                max_model_len=16,
                max_num_batched_tokens=4,
                pin_memory=False,
                device=torch.device("cpu"),
                block_sizes=[4, 8],
                kernel_sizes=[[4], [8]],
                max_num_blocks=[4, 2],
            )
        )

        positions = np.array([8, 9, 10, 11], dtype=np.int64)
        runner._prepare_staged_sfa_dummy_block_tables(
            batch_size=4,
            positions=positions,
        )

        expected_slots = (
            np.array([0, 5, 10, 15], dtype=np.int64),
            np.array([0, 9, 18, 27], dtype=np.int64),
        )
        for group_index, block_table in enumerate(runner.input_batch.block_table.block_tables):
            expected_rows = np.broadcast_to(
                np.arange(4, dtype=np.int32).reshape(-1, 1),
                (4, block_table.max_num_blocks_per_req),
            )
            np.testing.assert_array_equal(
                block_table.block_table.np[:4],
                expected_rows,
            )
            np.testing.assert_array_equal(
                block_table.slot_mapping.np[:4],
                expected_slots[group_index],
            )
            self.assertEqual(
                np.unique(block_table.slot_mapping.np[:4]).size,
                4,
            )

    def test_dummy_block_rows_require_enough_physical_blocks(self):
        runner = self._build_runner()
        runner.kv_cache_config = SimpleNamespace(num_blocks=3)
        runner.input_batch = SimpleNamespace(
            block_table=MultiGroupBlockTable(
                max_num_reqs=4,
                max_model_len=16,
                max_num_batched_tokens=4,
                pin_memory=False,
                device=torch.device("cpu"),
                block_sizes=[4, 8],
                kernel_sizes=[[4], [8]],
                max_num_blocks=[4, 2],
            )
        )

        with self.assertRaisesRegex(RuntimeError, "one physical block"):
            runner._prepare_staged_sfa_dummy_block_tables(
                batch_size=4,
                positions=np.arange(4),
            )

    def test_dummy_block_rows_reject_asymmetric_group_pool(self):
        runner = self._build_runner()
        runner.kv_cache_config = SimpleNamespace(
            num_blocks=8,
            num_blocks_per_group=[3, 8],
        )
        runner.input_batch = SimpleNamespace(
            block_table=MultiGroupBlockTable(
                max_num_reqs=4,
                max_model_len=16,
                max_num_batched_tokens=4,
                pin_memory=False,
                device=torch.device("cpu"),
                block_sizes=[4, 8],
                kernel_sizes=[[4], [8]],
                max_num_blocks=[4, 2],
            )
        )

        with self.assertRaisesRegex(
            RuntimeError,
            r"KV group 0: .*available_blocks=3",
        ):
            runner._prepare_staged_sfa_dummy_block_tables(
                batch_size=4,
                positions=np.arange(4),
            )

    def test_dummy_position_rejects_logical_row_overflow(self):
        runner = self._build_runner()
        runner.kv_cache_config = SimpleNamespace(
            num_blocks=8,
            num_blocks_per_group=[8, 8],
        )
        runner.input_batch = SimpleNamespace(
            block_table=MultiGroupBlockTable(
                max_num_reqs=4,
                max_model_len=16,
                max_num_batched_tokens=4,
                pin_memory=False,
                device=torch.device("cpu"),
                block_sizes=[4, 8],
                kernel_sizes=[[4], [8]],
                max_num_blocks=[4, 2],
            )
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "logical block-table capacity for KV group 0",
        ):
            runner._prepare_staged_sfa_dummy_block_tables(
                batch_size=4,
                positions=np.array([16, 1, 2, 3], dtype=np.int64),
            )

    def test_dummy_position_capacity_includes_cp_world_size(self):
        runner = self._build_runner()
        runner.kv_cache_config = SimpleNamespace(
            num_blocks=8,
            num_blocks_per_group=[8, 8],
        )
        block_table = MultiGroupBlockTable(
            max_num_reqs=4,
            max_model_len=16,
            max_num_batched_tokens=4,
            pin_memory=False,
            device=torch.device("cpu"),
            block_sizes=[4, 8],
            kernel_sizes=[[4], [8]],
            max_num_blocks=[4, 2],
        )
        for group_table in block_table.block_tables:
            group_table.dcp_world_size = 2
            group_table.dcp_rank = 0
            group_table.pcp_world_size = 1
            group_table.pcp_rank = 0
        runner.input_batch = SimpleNamespace(block_table=block_table)

        runner._prepare_staged_sfa_dummy_block_tables(
            batch_size=4,
            positions=np.array([16, 18, 20, 22], dtype=np.int64),
        )

        for group_table in block_table.block_tables:
            self.assertEqual(
                np.unique(group_table.slot_mapping.np[:4]).size,
                4,
            )


class TestSFALayerwiseGraphModeCompatibility(unittest.TestCase):
    @staticmethod
    def _build_runner(mode, *, use_sparse=True):
        runner = NPUModelRunner.__new__(NPUModelRunner)
        runner.use_sparse = use_sparse
        runner.compilation_config = SimpleNamespace(cudagraph_mode=mode)
        return runner

    @patch.object(
        model_runner_module,
        "staged_sfa_graph_configured",
        return_value=True,
    )
    def test_data_parallel_staged_graph_is_rejected(
        self,
        _staged_sfa_graph_configured,
    ):
        runner = self._build_runner(CUDAGraphMode.PIECEWISE)
        runner.vllm_config = object()
        runner.parallel_config = SimpleNamespace(data_parallel_size=2)

        with self.assertRaisesRegex(ValueError, "data parallel"):
            runner._validate_sfa_layerwise_connector_cudagraph_mode()

    def test_explicit_unsupported_staged_graph_request_is_rejected(self):
        runner = self._build_runner(CUDAGraphMode.PIECEWISE)
        runner.vllm_config = object()
        runner.parallel_config = SimpleNamespace(data_parallel_size=1)

        with (
            patch.object(
                model_runner_module.envs_ascend,
                "VLLM_ASCEND_SFA_STAGED_GRAPH",
                True,
            ),
            patch.object(
                model_runner_module,
                "staged_sfa_graph_configured",
                return_value=False,
            ),
            patch.object(
                model_runner_module,
                "staged_sfa_graph_configuration_errors",
                return_value=("speculative decoding/MTP is not implemented",),
            ),
            self.assertRaisesRegex(ValueError, "MTP"),
        ):
            runner._validate_sfa_layerwise_connector_cudagraph_mode()

    def test_staged_graph_requires_sparse_load_connector_capability(self):
        runner = self._build_runner(CUDAGraphMode.PIECEWISE)
        runner.vllm_config = object()
        runner.parallel_config = SimpleNamespace(data_parallel_size=1)

        with (
            patch.object(
                model_runner_module,
                "staged_sfa_graph_configured",
                return_value=True,
            ),
            patch.object(
                model_runner_module,
                "staged_sfa_connector_supports_sparse_load",
                return_value=False,
            ),
            self.assertRaisesRegex(
                ValueError,
                "batched sparse selective loads",
            ),
        ):
            runner._validate_sfa_layerwise_connector_cudagraph_mode()

        with (
            patch.object(
                model_runner_module,
                "staged_sfa_graph_configured",
                return_value=True,
            ),
            patch.object(
                model_runner_module,
                "staged_sfa_connector_supports_sparse_load",
                return_value=True,
            ),
        ):
            runner._validate_sfa_layerwise_connector_cudagraph_mode()

    def test_full_graph_modes_are_rejected_for_layerwise_connector(self):
        connector = SimpleNamespace(uses_layerwise_model_callbacks=True)
        for mode in (
            CUDAGraphMode.FULL,
            CUDAGraphMode.FULL_DECODE_ONLY,
        ):
            with (
                self.subTest(mode=mode),
                patch.object(
                    model_runner_module,
                    "has_kv_transfer_group",
                    return_value=True,
                ),
                patch.object(
                    model_runner_module,
                    "get_kv_transfer_group",
                    return_value=connector,
                ),
                self.assertRaisesRegex(ValueError, "PIECEWISE"),
            ):
                runner = self._build_runner(mode)
                runner.vllm_config = object()
                runner.parallel_config = SimpleNamespace(data_parallel_size=1)
                with patch.object(
                    model_runner_module,
                    "staged_sfa_graph_configured",
                    return_value=False,
                ):
                    runner._validate_sfa_layerwise_connector_cudagraph_mode()

    def test_compatible_modes_and_connectors_are_accepted(self):
        cases = (
            (
                CUDAGraphMode.PIECEWISE,
                True,
                True,
                True,
            ),
            (
                CUDAGraphMode.FULL,
                False,
                True,
                True,
            ),
            (
                CUDAGraphMode.FULL,
                True,
                False,
                True,
            ),
            (
                CUDAGraphMode.FULL,
                True,
                True,
                False,
            ),
        )
        for mode, use_sparse, has_connector, uses_layerwise in cases:
            with (
                self.subTest(
                    mode=mode,
                    use_sparse=use_sparse,
                    has_connector=has_connector,
                    uses_layerwise=uses_layerwise,
                ),
                patch.object(
                    model_runner_module,
                    "has_kv_transfer_group",
                    return_value=has_connector,
                ),
                patch.object(
                    model_runner_module,
                    "get_kv_transfer_group",
                    return_value=SimpleNamespace(
                        uses_layerwise_model_callbacks=uses_layerwise,
                    ),
                ),
            ):
                runner = self._build_runner(
                    mode,
                    use_sparse=use_sparse,
                )
                runner.vllm_config = object()
                runner.parallel_config = SimpleNamespace(data_parallel_size=1)
                with patch.object(
                    model_runner_module,
                    "staged_sfa_graph_configured",
                    return_value=False,
                ):
                    runner._validate_sfa_layerwise_connector_cudagraph_mode()


class TestStagedSFAStartupCaptureValidation(unittest.TestCase):
    def setUp(self):
        configured = patch.object(
            model_runner_module,
            "staged_sfa_graph_configured",
            return_value=True,
        )
        capture_sizes = patch.object(
            model_runner_module,
            "staged_sfa_graph_capture_sizes",
            return_value=(1,),
        )
        configured.start()
        capture_sizes.start()
        self.addCleanup(configured.stop)
        self.addCleanup(capture_sizes.stop)

    @staticmethod
    def _build_runner():
        runner = NPUModelRunner.__new__(NPUModelRunner)
        runner.compilation_config = SimpleNamespace(
            cudagraph_mode=CUDAGraphMode.PIECEWISE,
        )
        runner.vllm_config = SimpleNamespace(
            compilation_config=runner.compilation_config,
            model_config=SimpleNamespace(
                use_mla=True,
                hf_text_config=SimpleNamespace(index_topk=2048),
            ),
            kv_transfer_config=object(),
            speculative_config=None,
            lora_config=None,
        )
        runner._staged_sfa_startup_capture_attempted = False
        runner._staged_sfa_startup_capture_complete = False
        return runner

    @staticmethod
    def _signature(tensor):
        return (
            tensor.data_ptr(),
            tuple(tensor.shape),
            tuple(tensor.stride()),
            tensor.storage_offset(),
            tensor.dtype,
            tensor.device,
        )

    @classmethod
    def _make_captured_impl(cls):
        descriptor = BatchDescriptor(num_tokens=1)
        pre_input = torch.zeros(2)
        post_input = torch.zeros(3)
        pre_outputs = tuple(torch.zeros(2) for _ in range(4))
        post_output = torch.zeros(2)
        pre_canary = torch.ones(1, dtype=torch.int32)
        post_canary = torch.ones(1, dtype=torch.int32)
        pre_inputs = (pre_input, *pre_outputs, pre_canary)
        post_inputs = (post_input, post_output, post_canary)
        pre_entry = SimpleNamespace(
            aclgraph=object(),
            input_addresses=[value.data_ptr() for value in pre_inputs],
            output=pre_outputs,
        )
        post_entry = SimpleNamespace(
            aclgraph=object(),
            input_addresses=[value.data_ptr() for value in post_inputs],
            output=post_output,
        )
        return SimpleNamespace(
            enable_staged_sfa_graph=True,
            _staged_sfa_capture_phases={
                "pre:enter",
                "pre:exit",
                "post:enter",
                "post:exit",
            },
            _staged_sfa_capture_records={"pre", "post"},
            _staged_sfa_capture_failures=[],
            _staged_sfa_graph_input_signatures={
                "pre": tuple(cls._signature(value) for value in pre_inputs),
                "post": tuple(cls._signature(value) for value in post_inputs),
            },
            _staged_sfa_replay_proved=set(),
            _staged_sfa_pre_output_buffers=pre_outputs,
            _staged_sfa_replay_canaries={
                "pre": pre_canary,
                "post": post_canary,
            },
            _staged_sfa_pre_graph=SimpleNamespace(
                concrete_aclgraph_entries={descriptor: pre_entry},
            ),
            _staged_sfa_post_graph=SimpleNamespace(
                concrete_aclgraph_entries={descriptor: post_entry},
            ),
            _test_input_tensors=(*pre_inputs, *post_inputs),
        )

    @classmethod
    def _make_multi_key_captured_impl(cls, sizes=(1, 2)):
        pre_entries = {}
        post_entries = {}
        states = {}
        owned_tensors = []
        for size in sizes:
            key = StagedSFAGraphKey.exact_q1(size)
            descriptor = key.to_legacy_batch_descriptor()
            pre_input = torch.zeros(size + 1)
            post_input = torch.zeros(size + 2)
            pre_outputs = tuple(torch.zeros(size + 3) for _ in range(4))
            post_output = torch.zeros(size + 4)
            pre_canary = torch.ones(1, dtype=torch.int32)
            post_canary = torch.ones(1, dtype=torch.int32)
            pre_inputs = (pre_input, *pre_outputs, pre_canary)
            post_inputs = (post_input, post_output, post_canary)
            pre_entries[descriptor] = SimpleNamespace(
                aclgraph=object(),
                input_addresses=[value.data_ptr() for value in pre_inputs],
                output=pre_outputs,
            )
            post_entries[descriptor] = SimpleNamespace(
                aclgraph=object(),
                input_addresses=[value.data_ptr() for value in post_inputs],
                output=post_output,
            )
            states[key] = SimpleNamespace(
                key=key,
                capture_phases={
                    "pre:enter",
                    "pre:exit",
                    "post:enter",
                    "post:exit",
                },
                capture_records={"pre", "post"},
                capture_failures=[],
                graph_input_signatures={
                    "pre": tuple(cls._signature(value) for value in pre_inputs),
                    "post": tuple(cls._signature(value) for value in post_inputs),
                },
                replay_proved=set(),
                pre_output_buffers=pre_outputs,
                replay_canaries={
                    "pre": pre_canary,
                    "post": post_canary,
                },
                dummy_cache_initialized=True,
            )
            owned_tensors.extend((*pre_inputs, *post_inputs))
        impl = SimpleNamespace(
            enable_staged_sfa_graph=True,
            _staged_sfa_pre_graph=SimpleNamespace(
                concrete_aclgraph_entries=pre_entries,
            ),
            _staged_sfa_post_graph=SimpleNamespace(
                concrete_aclgraph_entries=post_entries,
            ),
            _staged_sfa_graph_states=states,
            _test_input_tensors=tuple(owned_tensors),
        )
        impl._iter_staged_sfa_graph_states = MagicMock(side_effect=lambda: tuple(states.items()))
        impl._activate_staged_sfa_graph_key = MagicMock()
        return impl

    @staticmethod
    def _layers(*impls):
        return {
            f"model.layers.{index}.self_attn.attn": SimpleNamespace(
                impl=impl,
            )
            for index, impl in enumerate(impls)
        }

    def test_validates_every_configured_layer_key_pair(self):
        runner = self._build_runner()
        impl = self._make_multi_key_captured_impl()
        with (
            patch.object(
                model_runner_module,
                "staged_sfa_graph_capture_sizes",
                return_value=(1, 2),
            ),
            patch.object(
                model_runner_module,
                "get_layers_from_vllm_config",
                return_value=self._layers(impl),
            ),
            patch.object(
                runner,
                "_staged_sfa_tp_consensus",
                return_value=(False, 2, 2),
            ) as consensus,
        ):
            runner._validate_staged_sfa_startup_capture()

        self.assertEqual(
            runner._staged_sfa_expected_graph_keys,
            (
                StagedSFAGraphKey.exact_q1(1),
                StagedSFAGraphKey.exact_q1(2),
            ),
        )
        consensus.assert_called_once_with(
            local_failed=False,
            local_count=2,
        )

    def test_validation_rejects_a_missing_configured_key(self):
        runner = self._build_runner()
        impl = self._make_multi_key_captured_impl()
        impl._staged_sfa_graph_states.pop(StagedSFAGraphKey.exact_q1(2))
        with (
            patch.object(
                model_runner_module,
                "staged_sfa_graph_capture_sizes",
                return_value=(1, 2),
            ),
            patch.object(
                model_runner_module,
                "get_layers_from_vllm_config",
                return_value=self._layers(impl),
            ),
            patch.object(
                runner,
                "_staged_sfa_tp_consensus",
                return_value=(True, 2, 2),
            ),
            self.assertRaisesRegex(RuntimeError, "missing_keys"),
        ):
            runner._validate_staged_sfa_startup_capture()

    def test_ordered_replay_proves_each_configured_key(self):
        runner = self._build_runner()
        impl = self._make_multi_key_captured_impl()
        keys = (
            StagedSFAGraphKey.exact_q1(1),
            StagedSFAGraphKey.exact_q1(2),
        )
        runner._staged_sfa_impls = (("layer-0", impl),)
        runner._staged_sfa_expected_graph_keys = keys

        def replay_key(batch_size, **_kwargs):
            state = impl._staged_sfa_graph_states[StagedSFAGraphKey.exact_q1(batch_size)]
            state.replay_canaries["pre"].fill_(1)
            state.replay_canaries["post"].fill_(1)

        runner._dummy_run = MagicMock(side_effect=replay_key)
        with (
            patch.object(torch.npu, "synchronize"),
            patch.object(
                model_runner_module,
                "get_tp_group",
                return_value=SimpleNamespace(
                    world_size=1,
                    cpu_group=None,
                ),
            ),
            patch.object(model_runner_module.logger, "info"),
        ):
            runner._prove_staged_sfa_ordered_startup_replay()

        self.assertEqual(
            [call.args[0] for call in runner._dummy_run.call_args_list],
            [1, 2],
        )
        self.assertTrue(all(state.replay_proved == {"pre", "post"} for state in impl._staged_sfa_graph_states.values()))
        impl._activate_staged_sfa_graph_key.assert_called_once_with(STAGED_SFA_SINGLETON_GRAPH_KEY)

    @patch.object(
        model_runner_module,
        "staged_sfa_graph_configured",
        return_value=True,
    )
    def test_accepts_complete_structural_capture_for_every_local_layer(
        self,
        _staged_sfa_graph_configured,
    ):
        runner = self._build_runner()
        impls = (self._make_captured_impl(), self._make_captured_impl())
        with (
            patch.object(
                model_runner_module,
                "get_layers_from_vllm_config",
                return_value=self._layers(*impls),
            ),
            patch.object(
                model_runner_module,
                "get_tp_group",
                return_value=SimpleNamespace(world_size=1, cpu_group=None),
            ),
        ):
            runner._validate_staged_sfa_startup_capture()

        self.assertEqual(runner._staged_sfa_expected_layer_count, 2)
        self.assertEqual(len(runner._staged_sfa_impls), 2)
        self.assertTrue(all(not impl._staged_sfa_replay_proved for impl in impls))

    @patch.object(
        model_runner_module,
        "staged_sfa_graph_configured",
        return_value=True,
    )
    def test_zero_local_layers_fails_after_consensus(
        self,
        _staged_sfa_graph_configured,
    ):
        runner = self._build_runner()
        with (
            patch.object(
                model_runner_module,
                "get_layers_from_vllm_config",
                return_value={},
            ),
            patch.object(
                runner,
                "_staged_sfa_tp_consensus",
                return_value=(True, 0, 0),
            ) as consensus,
            self.assertRaisesRegex(
                RuntimeError,
                "did not find any local staged SFA implementations",
            ),
        ):
            runner._validate_staged_sfa_startup_capture()

        consensus.assert_called_once_with(local_failed=True, local_count=0)

    @patch.object(
        model_runner_module,
        "staged_sfa_graph_configured",
        return_value=True,
    )
    def test_reports_all_structural_defects_after_consensus(
        self,
        _staged_sfa_graph_configured,
    ):
        runner = self._build_runner()
        impl = self._make_captured_impl()
        impl._staged_sfa_capture_phases.remove("post:exit")
        impl._staged_sfa_capture_records.remove("post")
        impl._staged_sfa_capture_failures.append("pre capture failed")
        impl._staged_sfa_graph_input_signatures.pop("pre")
        impl._staged_sfa_pre_output_buffers = None
        impl._staged_sfa_replay_canaries.pop("post")
        impl._staged_sfa_post_graph.concrete_aclgraph_entries.clear()
        with (
            patch.object(
                model_runner_module,
                "get_layers_from_vllm_config",
                return_value=self._layers(impl),
            ),
            patch.object(
                runner,
                "_staged_sfa_tp_consensus",
                return_value=(True, 1, 1),
            ) as consensus,
            self.assertRaisesRegex(
                RuntimeError,
                r"entries=\['post'\].*pre capture failed",
            ),
        ):
            runner._validate_staged_sfa_startup_capture()

        consensus.assert_called_once_with(local_failed=True, local_count=1)

    @patch.object(
        model_runner_module,
        "staged_sfa_graph_configured",
        return_value=True,
    )
    def test_rejects_owned_storage_tail_and_post_output_binding_drift(
        self,
        _staged_sfa_graph_configured,
    ):
        runner = self._build_runner()
        impl = self._make_captured_impl()
        descriptor = BatchDescriptor(num_tokens=1)
        pre_entry = impl._staged_sfa_pre_graph.concrete_aclgraph_entries[descriptor]
        pre_signatures = list(impl._staged_sfa_graph_input_signatures["pre"])
        pre_signatures[-5] = (
            pre_signatures[-4][0],
            *pre_signatures[-5][1:],
        )
        impl._staged_sfa_graph_input_signatures["pre"] = tuple(pre_signatures)
        pre_entry.input_addresses[-5] = pre_signatures[-5][0]

        post_entry = impl._staged_sfa_post_graph.concrete_aclgraph_entries[descriptor]
        post_signatures = list(impl._staged_sfa_graph_input_signatures["post"])
        post_signatures[-1] = (
            post_signatures[-2][0],
            *post_signatures[-1][1:],
        )
        impl._staged_sfa_graph_input_signatures["post"] = tuple(post_signatures)
        post_entry.input_addresses[-1] = post_signatures[-1][0]
        impl._test_drift_output = torch.empty_like(post_entry.output)
        post_entry.output = impl._test_drift_output

        with (
            patch.object(
                model_runner_module,
                "get_layers_from_vllm_config",
                return_value=self._layers(impl),
            ),
            patch.object(
                runner,
                "_staged_sfa_tp_consensus",
                return_value=(True, 1, 1),
            ) as consensus,
            self.assertRaisesRegex(
                RuntimeError,
                "pre_signature_tail_mismatch=True.*"
                "post_signature_tail_mismatch=True.*"
                "post_output_binding_mismatch=True",
            ),
        ):
            runner._validate_staged_sfa_startup_capture()

        consensus.assert_called_once_with(local_failed=True, local_count=1)

    @patch.object(
        model_runner_module,
        "staged_sfa_graph_configured",
        return_value=True,
    )
    def test_rejects_aliasing_strong_pre_output_buffers(
        self,
        _staged_sfa_graph_configured,
    ):
        runner = self._build_runner()
        impl = self._make_captured_impl()
        descriptor = BatchDescriptor(num_tokens=1)
        shared_storage = torch.zeros(4)
        pre_outputs = (
            shared_storage[:2],
            shared_storage[2:],
            torch.zeros(2),
            torch.zeros(2),
        )
        impl._staged_sfa_pre_output_buffers = pre_outputs
        pre_entry = impl._staged_sfa_pre_graph.concrete_aclgraph_entries[descriptor]
        pre_entry.output = pre_outputs
        pre_signatures = list(impl._staged_sfa_graph_input_signatures["pre"])
        expected_tail = (
            *pre_outputs,
            impl._staged_sfa_replay_canaries["pre"],
        )
        pre_signatures[-5:] = [self._signature(value) for value in expected_tail]
        impl._staged_sfa_graph_input_signatures["pre"] = tuple(pre_signatures)
        pre_entry.input_addresses[-5:] = [value.data_ptr() for value in expected_tail]

        with (
            patch.object(
                model_runner_module,
                "get_layers_from_vllm_config",
                return_value=self._layers(impl),
            ),
            patch.object(
                runner,
                "_staged_sfa_tp_consensus",
                return_value=(True, 1, 1),
            ),
            self.assertRaisesRegex(
                RuntimeError,
                "pre_output_storage_alias=True",
            ),
        ):
            runner._validate_staged_sfa_startup_capture()

    @patch.object(
        model_runner_module,
        "staged_sfa_graph_configured",
        return_value=True,
    )
    def test_rejects_distinct_canary_views_of_one_storage(
        self,
        _staged_sfa_graph_configured,
    ):
        runner = self._build_runner()
        impl = self._make_captured_impl()
        descriptor = BatchDescriptor(num_tokens=1)
        shared_storage = torch.ones(2, dtype=torch.int32)
        pre_canary = shared_storage[:1]
        post_canary = shared_storage[1:]
        impl._staged_sfa_replay_canaries = {
            "pre": pre_canary,
            "post": post_canary,
        }
        pre_signatures = list(impl._staged_sfa_graph_input_signatures["pre"])
        pre_signatures[-1] = self._signature(pre_canary)
        impl._staged_sfa_graph_input_signatures["pre"] = tuple(pre_signatures)
        post_signatures = list(impl._staged_sfa_graph_input_signatures["post"])
        post_signatures[-1] = self._signature(post_canary)
        impl._staged_sfa_graph_input_signatures["post"] = tuple(post_signatures)
        impl._staged_sfa_pre_graph.concrete_aclgraph_entries[descriptor].input_addresses[-1] = pre_canary.data_ptr()
        impl._staged_sfa_post_graph.concrete_aclgraph_entries[descriptor].input_addresses[-1] = post_canary.data_ptr()

        with (
            patch.object(
                model_runner_module,
                "get_layers_from_vllm_config",
                return_value=self._layers(impl),
            ),
            patch.object(
                runner,
                "_staged_sfa_tp_consensus",
                return_value=(True, 1, 1),
            ),
            self.assertRaisesRegex(
                RuntimeError,
                "shared_canary_storage=True",
            ),
        ):
            runner._validate_staged_sfa_startup_capture()

    @patch.object(
        model_runner_module,
        "staged_sfa_graph_configured",
        return_value=True,
    )
    def test_rejects_canary_view_of_graph_output_storage(
        self,
        _staged_sfa_graph_configured,
    ):
        runner = self._build_runner()
        impl = self._make_captured_impl()
        descriptor = BatchDescriptor(num_tokens=1)
        shared_storage = torch.zeros(3, dtype=torch.int32)
        pre_outputs = list(impl._staged_sfa_pre_output_buffers)
        pre_outputs[0] = shared_storage[:2]
        pre_outputs = tuple(pre_outputs)
        pre_canary = shared_storage[2:]
        impl._staged_sfa_pre_output_buffers = pre_outputs
        impl._staged_sfa_replay_canaries["pre"] = pre_canary
        pre_entry = impl._staged_sfa_pre_graph.concrete_aclgraph_entries[descriptor]
        pre_entry.output = pre_outputs
        expected_tail = (*pre_outputs, pre_canary)
        pre_signatures = list(impl._staged_sfa_graph_input_signatures["pre"])
        pre_signatures[-5:] = [self._signature(value) for value in expected_tail]
        impl._staged_sfa_graph_input_signatures["pre"] = tuple(pre_signatures)
        pre_entry.input_addresses[-5:] = [value.data_ptr() for value in expected_tail]

        with (
            patch.object(
                model_runner_module,
                "get_layers_from_vllm_config",
                return_value=self._layers(impl),
            ),
            patch.object(
                runner,
                "_staged_sfa_tp_consensus",
                return_value=(True, 1, 1),
            ),
            self.assertRaisesRegex(
                RuntimeError,
                "canary_output_storage_alias=True",
            ),
        ):
            runner._validate_staged_sfa_startup_capture()

    @patch.object(
        model_runner_module,
        "staged_sfa_graph_configured",
        return_value=True,
    )
    def test_rejects_non_int32_vector_canary_signature(
        self,
        _staged_sfa_graph_configured,
    ):
        runner = self._build_runner()
        impl = self._make_captured_impl()
        descriptor = BatchDescriptor(num_tokens=1)
        malformed_canary = torch.ones((), dtype=torch.float32)
        impl._staged_sfa_replay_canaries["pre"] = malformed_canary
        pre_signatures = list(impl._staged_sfa_graph_input_signatures["pre"])
        pre_signatures[-1] = self._signature(malformed_canary)
        impl._staged_sfa_graph_input_signatures["pre"] = tuple(pre_signatures)
        pre_entry = impl._staged_sfa_pre_graph.concrete_aclgraph_entries[descriptor]
        pre_entry.input_addresses[-1] = malformed_canary.data_ptr()

        with (
            patch.object(
                model_runner_module,
                "get_layers_from_vllm_config",
                return_value=self._layers(impl),
            ),
            patch.object(
                runner,
                "_staged_sfa_tp_consensus",
                return_value=(True, 1, 1),
            ),
            self.assertRaisesRegex(
                RuntimeError,
                r"malformed_canaries=\['pre'\]",
            ),
        ):
            runner._validate_staged_sfa_startup_capture()

    @patch.object(
        model_runner_module,
        "staged_sfa_graph_configured",
        return_value=True,
    )
    def test_remote_structural_failure_forces_local_raise(
        self,
        _staged_sfa_graph_configured,
    ):
        runner = self._build_runner()
        impl = self._make_captured_impl()
        with (
            patch.object(
                model_runner_module,
                "get_layers_from_vllm_config",
                return_value=self._layers(impl),
            ),
            patch.object(
                runner,
                "_staged_sfa_tp_consensus",
                return_value=(True, 1, 1),
            ),
            self.assertRaisesRegex(
                RuntimeError,
                "local failures: none on this TP rank",
            ),
        ):
            runner._validate_staged_sfa_startup_capture()

    def test_tp_consensus_uses_cpu_group_and_reports_count_range(self):
        runner = self._build_runner()
        cpu_group = object()

        def reduce_remote_status(status, **_kwargs):
            status[0] = 1
            status[1] = 4
            status[2] = -2

        with (
            patch.object(
                model_runner_module,
                "get_tp_group",
                return_value=SimpleNamespace(
                    world_size=2,
                    cpu_group=cpu_group,
                ),
            ),
            patch.object(
                model_runner_module.dist,
                "all_reduce",
                side_effect=reduce_remote_status,
            ) as all_reduce,
        ):
            result = runner._staged_sfa_tp_consensus(
                local_failed=False,
                local_count=3,
            )

        self.assertEqual(result, (True, 2, 4))
        self.assertIs(all_reduce.call_args.kwargs["group"], cpu_group)
        self.assertEqual(all_reduce.call_args.kwargs["op"], torch.distributed.ReduceOp.MAX)

    @patch.object(
        model_runner_module,
        "staged_sfa_graph_configured",
        return_value=False,
    )
    def test_validation_is_inert_when_poc_is_disabled(
        self,
        _staged_sfa_graph_configured,
    ):
        runner = self._build_runner()
        with patch.object(
            model_runner_module,
            "get_layers_from_vllm_config",
        ) as get_layers:
            runner._validate_staged_sfa_startup_capture()

        get_layers.assert_not_called()
        self.assertEqual(runner._staged_sfa_impls, ())
        self.assertEqual(runner._staged_sfa_expected_layer_count, 0)

    def test_reset_discards_stale_profiling_entries_and_impl_state(self):
        runner = self._build_runner()
        runner._staged_sfa_startup_capture_attempted = True
        runner._staged_sfa_live_parity_request_id = "stale"
        runner._staged_sfa_live_parity_validated_seq_lens = [4096]
        runner._staged_sfa_live_parity_last_seq_len = 4096
        runner._staged_sfa_live_parity_histories = {
            STAGED_SFA_SINGLETON_GRAPH_KEY: [(4096,)],
        }
        runner._staged_sfa_live_parity_last_batches = {
            STAGED_SFA_SINGLETON_GRAPH_KEY: (
                ("stale",),
                (4096,),
            ),
        }
        impl = self._make_captured_impl()
        impl._staged_sfa_replay_proved = {"pre", "post"}
        impl._staged_sfa_dummy_cache_initialized = True
        impl._staged_sfa_live_capture_validated = True
        impl._staged_sfa_live_validated_request_ids = ("stale",)
        impl._staged_sfa_parity_output = torch.ones(1)
        impl._staged_sfa_parity_latent_scratch = torch.ones(1)
        old_pre_entries = impl._staged_sfa_pre_graph.concrete_aclgraph_entries
        old_post_entries = impl._staged_sfa_post_graph.concrete_aclgraph_entries
        with patch.object(
            runner,
            "_collect_staged_sfa_impls",
            return_value=(("layer-0", impl),),
        ):
            runner._reset_staged_sfa_startup_capture()

        self.assertEqual(old_pre_entries, {})
        self.assertEqual(old_post_entries, {})
        self.assertIsNone(impl._staged_sfa_pre_graph)
        self.assertIsNone(impl._staged_sfa_post_graph)
        self.assertEqual(impl._staged_sfa_capture_phases, set())
        self.assertEqual(impl._staged_sfa_capture_records, set())
        self.assertEqual(impl._staged_sfa_capture_failures, [])
        self.assertEqual(impl._staged_sfa_graph_input_signatures, {})
        self.assertEqual(impl._staged_sfa_replay_proved, set())
        self.assertIsNone(impl._staged_sfa_pre_output_buffers)
        self.assertEqual(impl._staged_sfa_replay_canaries, {})
        self.assertFalse(impl._staged_sfa_dummy_cache_initialized)
        self.assertFalse(impl._staged_sfa_live_capture_validated)
        self.assertIsNone(impl._staged_sfa_live_validated_request_ids)
        self.assertIsNone(impl._staged_sfa_parity_output)
        self.assertIsNone(impl._staged_sfa_parity_latent_scratch)
        self.assertTrue(runner._staged_sfa_startup_capture_attempted)
        self.assertFalse(runner._staged_sfa_startup_capture_complete)
        self.assertIsNone(runner._staged_sfa_live_parity_request_id)
        self.assertEqual(
            runner._staged_sfa_live_parity_validated_seq_lens,
            [],
        )
        self.assertIsNone(runner._staged_sfa_live_parity_last_seq_len)
        self.assertEqual(runner._staged_sfa_live_parity_histories, {})
        self.assertEqual(runner._staged_sfa_live_parity_last_batches, {})

    @patch.object(
        model_runner_module,
        "staged_sfa_graph_configured",
        return_value=True,
    )
    def test_ordered_full_model_replay_proves_every_canary(
        self,
        _staged_sfa_graph_configured,
    ):
        runner = self._build_runner()
        with torch.inference_mode():
            impls = (
                self._make_captured_impl(),
                self._make_captured_impl(),
            )
        self.assertTrue(
            all(torch.is_inference(canary) for impl in impls for canary in impl._staged_sfa_replay_canaries.values())
        )
        runner._staged_sfa_impls = tuple((f"layer-{index}", impl) for index, impl in enumerate(impls))

        def ordered_dummy_replay(*args, **kwargs):
            self.assertEqual(args, (1,))
            self.assertTrue(torch.is_inference_mode_enabled())
            self.assertEqual(
                kwargs,
                {
                    "cudagraph_runtime_mode": CUDAGraphMode.PIECEWISE,
                    "uniform_decode": False,
                    "allow_microbatching": False,
                    "skip_eplb": True,
                    "remove_lora": False,
                    "num_active_loras": 0,
                },
            )
            for impl in impls:
                self.assertEqual(impl._staged_sfa_replay_canaries["pre"].item(), 0)
                self.assertEqual(impl._staged_sfa_replay_canaries["post"].item(), 0)
                impl._staged_sfa_replay_canaries["pre"].fill_(1)
                impl._staged_sfa_replay_canaries["post"].fill_(1)

        runner._dummy_run = MagicMock(side_effect=ordered_dummy_replay)
        with (
            patch.object(torch.npu, "synchronize") as synchronize,
            patch.object(
                model_runner_module,
                "get_tp_group",
                return_value=SimpleNamespace(world_size=1, cpu_group=None),
            ),
            patch.object(model_runner_module.logger, "info") as info,
        ):
            runner._prove_staged_sfa_ordered_startup_replay()

        self.assertEqual(synchronize.call_count, 2)
        runner._dummy_run.assert_called_once()
        self.assertTrue(runner._staged_sfa_startup_capture_complete)
        self.assertTrue(all(impl._staged_sfa_replay_proved == {"pre", "post"} for impl in impls))
        info.assert_called_once_with(
            "[SFA staged graph POC] startup capture and ordered replay-canary "
            "completeness check passed for %d local SFA layers, %d keys "
            "(%d staged graphs).",
            2,
            1,
            4,
        )

    @patch.object(
        model_runner_module,
        "staged_sfa_graph_configured",
        return_value=True,
    )
    def test_missing_canary_write_fails_after_tp_consensus(
        self,
        _staged_sfa_graph_configured,
    ):
        runner = self._build_runner()
        with torch.inference_mode():
            impl = self._make_captured_impl()
        self.assertTrue(all(torch.is_inference(canary) for canary in impl._staged_sfa_replay_canaries.values()))
        runner._staged_sfa_impls = (("layer-0", impl),)

        def replay_pre_only(*_args, **_kwargs):
            self.assertTrue(torch.is_inference_mode_enabled())
            impl._staged_sfa_replay_canaries["pre"].fill_(1)

        runner._dummy_run = MagicMock(side_effect=replay_pre_only)
        with (
            patch.object(torch.npu, "synchronize"),
            patch.object(
                runner,
                "_staged_sfa_tp_consensus",
                return_value=(True, 1, 1),
            ) as consensus,
            self.assertRaisesRegex(
                RuntimeError,
                "post replay canary is 0, expected 1",
            ),
        ):
            runner._prove_staged_sfa_ordered_startup_replay()

        consensus.assert_called_once_with(local_failed=True, local_count=1)
        self.assertFalse(runner._staged_sfa_startup_capture_complete)
        self.assertEqual(impl._staged_sfa_replay_proved, set())
        self.assertEqual(impl._staged_sfa_replay_canaries["pre"].item(), 1)
        self.assertEqual(impl._staged_sfa_replay_canaries["post"].item(), 0)

    def test_capture_model_preserves_result_and_orders_all_checks(self):
        runner = self._build_runner()
        calls = []
        with (
            patch.object(
                model_runner_module,
                "staged_sfa_graph_configured",
                return_value=True,
            ),
            patch.object(
                model_runner_module,
                "_torch_cuda_wrapper",
                return_value=nullcontext(),
            ),
            patch.object(
                model_runner_module,
                "_replace_gpu_model_runner_function_wrapper",
                return_value=nullcontext(),
            ),
            patch.object(
                runner,
                "_reset_staged_sfa_startup_capture",
                side_effect=lambda: calls.append("reset"),
            ),
            patch.object(
                model_runner_module.GPUModelRunner,
                "capture_model",
                side_effect=lambda _runner: calls.append("parent") or 123,
            ) as parent_capture,
            patch.object(
                runner,
                "_validate_staged_sfa_startup_capture",
                side_effect=lambda: calls.append("validate"),
            ),
            patch.object(
                runner,
                "_prove_staged_sfa_ordered_startup_replay",
                side_effect=lambda: calls.append("replay"),
            ),
        ):
            result = runner.capture_model()

        self.assertEqual(result, 123)
        self.assertEqual(calls, ["reset", "parent", "validate", "replay"])
        self.assertTrue(runner._staged_sfa_startup_capture_attempted)
        parent_capture.assert_called_once_with(runner)

    def test_failed_parent_capture_cannot_retry_stale_outer_graphs(self):
        runner = self._build_runner()
        with (
            patch.object(
                model_runner_module,
                "staged_sfa_graph_configured",
                return_value=True,
            ),
            patch.object(
                model_runner_module,
                "_torch_cuda_wrapper",
                return_value=nullcontext(),
            ),
            patch.object(
                model_runner_module,
                "_replace_gpu_model_runner_function_wrapper",
                return_value=nullcontext(),
            ),
            patch.object(
                runner,
                "_reset_staged_sfa_startup_capture",
            ),
            patch.object(
                model_runner_module.GPUModelRunner,
                "capture_model",
                side_effect=RuntimeError("capture failed"),
            ) as parent_capture,
        ):
            with self.assertRaisesRegex(RuntimeError, "capture failed"):
                runner.capture_model()
            with self.assertRaisesRegex(
                RuntimeError,
                "startup graph capture was already attempted",
            ):
                runner.capture_model()

        self.assertTrue(runner._staged_sfa_startup_capture_attempted)
        self.assertFalse(runner._staged_sfa_startup_capture_complete)
        parent_capture.assert_called_once_with(runner)

    def test_second_capture_attempt_is_rejected_before_parent(self):
        runner = self._build_runner()
        runner._staged_sfa_startup_capture_attempted = True
        runner._staged_sfa_startup_capture_complete = True
        with (
            patch.object(
                model_runner_module,
                "staged_sfa_graph_configured",
                return_value=True,
            ),
            patch.object(
                runner,
                "_reset_staged_sfa_startup_capture",
            ) as reset,
            patch.object(
                model_runner_module.GPUModelRunner,
                "capture_model",
            ) as parent_capture,
            self.assertRaisesRegex(
                RuntimeError,
                "startup graph capture was already attempted",
            ),
        ):
            runner.capture_model()

        reset.assert_not_called()
        parent_capture.assert_not_called()


class TestStagedSFALiveParity(unittest.TestCase):
    _PARITY_SUFFIXES = (
        "pre.ql_nope",
        "pre.q_pe",
        "pre.topk_indices",
        "pre.selected_packed",
        "pre.cache_nope",
        "pre.cache_pe",
        "pre.cache_index",
        "post.output",
    )

    @classmethod
    def _mark_layer_passed(
        cls,
        state,
        impl_id,
        layer_name,
        failed_suffix=None,
    ):
        state.checked_impl_ids.add(impl_id)
        state.checked_layer_names.append(layer_name)
        state.match_flags.extend(
            (
                f"{layer_name}: {suffix}",
                torch.tensor(suffix != failed_suffix),
            )
            for suffix in cls._PARITY_SUFFIXES
        )

    @classmethod
    def _mark_all_layers_passed(cls, state, failed_label=None):
        for layer_index in range(state.expected_layers):
            layer_name = f"layer-{layer_index}"
            failed_suffix = None
            if failed_label is not None:
                label_prefix = f"{layer_name}: "
                if failed_label.startswith(label_prefix):
                    failed_suffix = failed_label[len(label_prefix) :]
            cls._mark_layer_passed(
                state,
                101 * (layer_index + 1),
                layer_name,
                failed_suffix,
            )

    @staticmethod
    def _build_runner(expected_layers=2):
        runner = NPUModelRunner.__new__(NPUModelRunner)
        runner._staged_sfa_expected_layer_count = expected_layers
        runner._staged_sfa_live_parity_request_id = None
        runner._staged_sfa_live_parity_validated_seq_lens = []
        runner._staged_sfa_live_parity_last_seq_len = None
        runner.attn_state = model_runner_module.AscendAttentionState.DecodeOnly
        runner.speculative_config = None
        runner.vllm_config = SimpleNamespace(lora_config=None)
        runner.dsa_shrink_latent = 2
        runner.dsa_offload_manager = None
        runner.dsa_adapter_cache = None
        runner.dsa_index_topk = 2048
        runner.input_batch = SimpleNamespace(
            num_prompt_tokens=[4096],
            num_computed_tokens_cpu=[4096],
        )
        runner.seq_lens = SimpleNamespace(np=[4096])
        return runner

    @staticmethod
    def _prepare(runner, seq_len, request_id="request-0"):
        runner.seq_lens.np[0] = seq_len
        return runner._prepare_staged_sfa_live_parity(
            cudagraph_mode=CUDAGraphMode.PIECEWISE,
            batch_descriptor=BatchDescriptor(num_tokens=1),
            num_tokens_unpadded=1,
            num_reqs=1,
            request_ids=[request_id],
        )

    @classmethod
    def _commit(cls, runner, parity_state):
        cls._mark_all_layers_passed(parity_state)
        with (
            patch.object(
                model_runner_module,
                "get_tp_group",
                return_value=SimpleNamespace(world_size=1, cpu_group=object()),
            ),
            patch.object(model_runner_module.logger, "info"),
        ):
            runner._finalize_staged_sfa_live_parity(parity_state)

    def test_batched_key_tracks_sequence_tuples_independently(self):
        runner = self._build_runner()
        key_two = StagedSFAGraphKey.exact_q1(2)
        runner.input_batch.num_prompt_tokens = [4096, 4096]
        runner.input_batch.num_computed_tokens_cpu = [4096, 4096]
        runner.seq_lens.np = [4096, 5000]

        state = runner._prepare_staged_sfa_live_parity(
            cudagraph_mode=CUDAGraphMode.PIECEWISE,
            batch_descriptor=BatchDescriptor(num_tokens=2),
            num_tokens_unpadded=2,
            num_reqs=2,
            request_ids=["request-0", "request-1"],
            graph_key=key_two,
        )
        self.assertIsNotNone(state)
        self.assertEqual(
            state.request_ids,
            ("request-0", "request-1"),
        )
        self.assertEqual(state.seq_lens, (4096, 5000))
        self.assertEqual(state.graph_key, key_two)
        self._commit(runner, state)
        self.assertEqual(
            runner._staged_sfa_live_parity_histories[key_two],
            [(4096, 5000)],
        )
        runner.seq_lens.np = [4097, 5001]
        self.assertIsNone(
            runner._prepare_staged_sfa_live_parity(
                cudagraph_mode=CUDAGraphMode.PIECEWISE,
                batch_descriptor=BatchDescriptor(num_tokens=2),
                num_tokens_unpadded=2,
                num_reqs=2,
                request_ids=["request-0", "request-1"],
                graph_key=key_two,
            )
        )

        runner.input_batch.num_prompt_tokens = [4096]
        runner.input_batch.num_computed_tokens_cpu = [4096]
        runner.seq_lens.np = [4096]
        singleton = self._prepare(runner, 4096)
        self.assertIsNotNone(singleton)
        self.assertEqual(
            runner._staged_sfa_live_parity_histories[STAGED_SFA_SINGLETON_GRAPH_KEY],
            [],
        )

    def test_first_successful_check_completes_key(self):
        runner = self._build_runner()

        first = self._prepare(runner, 4096)
        self.assertIsNotNone(first)
        equal_retry = self._prepare(runner, 4096)
        self.assertIsNotNone(equal_retry)
        self.assertIsNot(first, equal_retry)

        self._commit(runner, first)
        self.assertEqual(
            runner._staged_sfa_live_parity_validated_seq_lens,
            [4096],
        )
        self.assertIsNone(self._prepare(runner, 4096))
        self.assertIsNone(self._prepare(runner, 4097))

    def test_completed_key_does_not_suppress_a_new_key(self):
        runner = self._build_runner()
        state = self._prepare(runner, 4096)
        self.assertIsNotNone(state)
        self._commit(runner, state)

        key_two = StagedSFAGraphKey.exact_q1(2)
        runner.input_batch.num_prompt_tokens = [4096, 4096]
        runner.input_batch.num_computed_tokens_cpu = [4096, 4096]
        runner.seq_lens.np = [5000, 6000]
        state = runner._prepare_staged_sfa_live_parity(
            cudagraph_mode=CUDAGraphMode.PIECEWISE,
            batch_descriptor=BatchDescriptor(num_tokens=2),
            num_tokens_unpadded=2,
            num_reqs=2,
            request_ids=["request-0", "request-1"],
            graph_key=key_two,
        )

        self.assertIsNotNone(state)
        self.assertEqual(state.graph_key, key_two)

    def test_recalc_last_prefix_hit_does_not_arm_or_mutate_tracking(self):
        runner = self._build_runner()
        runner._staged_sfa_live_parity_request_id = "request-before"
        runner._staged_sfa_live_parity_validated_seq_lens = [5000]
        runner._staged_sfa_live_parity_last_seq_len = 5000
        runner.input_batch.num_prompt_tokens[0] = 6400
        runner.input_batch.num_computed_tokens_cpu[0] = 6399

        recalc_last = self._prepare(
            runner,
            6400,
            request_id="full-prefix-hit",
        )

        self.assertIsNone(recalc_last)
        self.assertEqual(
            runner._staged_sfa_live_parity_request_id,
            "request-before",
        )
        self.assertEqual(
            runner._staged_sfa_live_parity_validated_seq_lens,
            [5000],
        )
        self.assertEqual(
            runner._staged_sfa_live_parity_last_seq_len,
            5000,
        )

        runner.input_batch.num_computed_tokens_cpu[0] = 6400
        true_decode = self._prepare(
            runner,
            6401,
            request_id="full-prefix-hit",
        )
        self.assertIsNone(true_decode)
        self.assertEqual(
            runner._staged_sfa_live_parity_request_id,
            "request-before",
        )
        self.assertEqual(
            runner._staged_sfa_live_parity_validated_seq_lens,
            [5000],
        )
        self.assertEqual(
            runner._staged_sfa_live_parity_last_seq_len,
            5000,
        )

    def test_completed_key_is_inert_across_request_churn(self):
        runner = self._build_runner()
        state = self._prepare(runner, 4096, "request-a")
        self.assertIsNotNone(state)
        self._commit(runner, state)
        self.assertIsNone(self._prepare(runner, 4097, "request-a"))
        tracked_batch = runner._staged_sfa_live_parity_last_batches[STAGED_SFA_SINGLETON_GRAPH_KEY]

        switched = self._prepare(runner, 6000, "request-b")
        self.assertIsNone(switched)
        self.assertEqual(
            runner._staged_sfa_live_parity_validated_seq_lens,
            [4096],
        )
        self.assertEqual(
            runner._staged_sfa_live_parity_request_id,
            "request-a",
        )
        self.assertEqual(
            runner._staged_sfa_live_parity_last_seq_len,
            4096,
        )
        self.assertEqual(
            runner._staged_sfa_live_parity_last_batches[STAGED_SFA_SINGLETON_GRAPH_KEY],
            tracked_batch,
        )

        decreased = self._prepare(runner, 5999, "request-b")
        self.assertIsNone(decreased)
        self.assertEqual(
            runner._staged_sfa_live_parity_validated_seq_lens,
            [4096],
        )
        self.assertEqual(
            runner._staged_sfa_live_parity_last_seq_len,
            4096,
        )
        self.assertEqual(
            runner._staged_sfa_live_parity_last_batches[STAGED_SFA_SINGLETON_GRAPH_KEY],
            tracked_batch,
        )

    def test_finalize_rejects_checked_layer_count_mismatch(self):
        runner = self._build_runner()
        state = self._prepare(runner, 4096)
        self.assertIsNotNone(state)
        state.checked_impl_ids.add(101)

        with (
            patch.object(
                model_runner_module,
                "get_tp_group",
                return_value=SimpleNamespace(world_size=1, cpu_group=object()),
            ),
            self.assertRaisesRegex(
                RuntimeError,
                "local_checked=1.*expected=2",
            ),
        ):
            runner._finalize_staged_sfa_live_parity(state)

        self.assertEqual(
            runner._staged_sfa_live_parity_validated_seq_lens,
            [],
        )
        self.assertIsNotNone(self._prepare(runner, 4097))

    def test_finalize_materializes_deferred_comparison_flags_once(self):
        runner = self._build_runner()
        state = self._prepare(runner, 4096)
        self.assertIsNotNone(state)
        self._mark_all_layers_passed(
            state,
            failed_label="layer-0: post.output",
        )
        state.pending_saves.append(("layer-0", [torch.tensor([1.0])]))

        with (
            patch.object(
                model_runner_module,
                "get_tp_group",
                return_value=SimpleNamespace(world_size=1, cpu_group=object()),
            ),
            patch.object(
                model_runner_module,
                "maybe_save_kv_layer_to_connector",
            ) as save_layer,
            self.assertRaisesRegex(
                RuntimeError,
                "layer-0: post.output",
            ),
        ):
            runner._finalize_staged_sfa_live_parity(state)

        self.assertEqual(
            state.failures,
            ["layer-0: post.output"],
        )
        save_layer.assert_not_called()

    def test_finalize_flushes_ordered_saves_only_after_success(self):
        runner = self._build_runner()
        state = self._prepare(runner, 4096)
        self.assertIsNotNone(state)
        self._mark_all_layers_passed(state)
        latent = torch.tensor([1.0])
        index = torch.tensor([2.0])
        state.pending_saves.extend(
            [
                ("layer-0", [latent]),
                ("layer-0.index", [index]),
            ]
        )

        with (
            patch.object(
                model_runner_module,
                "get_tp_group",
                return_value=SimpleNamespace(
                    world_size=1,
                    cpu_group=object(),
                ),
            ),
            patch.object(
                model_runner_module,
                "maybe_save_kv_layer_to_connector",
            ) as save_layer,
            patch.object(model_runner_module.logger, "info"),
        ):
            runner._finalize_staged_sfa_live_parity(state)

        self.assertEqual(
            [call.args[0] for call in save_layer.call_args_list],
            ["layer-0", "layer-0.index"],
        )
        self.assertIs(save_layer.call_args_list[0].args[1][0], latent)
        self.assertIs(save_layer.call_args_list[1].args[1][0], index)
        self.assertEqual(
            runner._staged_sfa_live_parity_validated_seq_lens,
            [4096],
        )

    def test_finalize_rejects_missing_per_layer_comparison_flag(self):
        runner = self._build_runner()
        state = self._prepare(runner, 4096)
        self.assertIsNotNone(state)
        self._mark_all_layers_passed(state)
        state.match_flags.pop()

        with (
            patch.object(
                model_runner_module,
                "get_tp_group",
                return_value=SimpleNamespace(
                    world_size=1,
                    cpu_group=object(),
                ),
            ),
            self.assertRaisesRegex(
                RuntimeError,
                "comparison-count mismatch.*flags=15.*expected=16",
            ),
        ):
            runner._finalize_staged_sfa_live_parity(state)

        self.assertEqual(
            runner._staged_sfa_live_parity_validated_seq_lens,
            [],
        )

    def test_finalize_rejects_duplicate_comparison_labels(self):
        runner = self._build_runner()
        state = self._prepare(runner, 4096)
        self.assertIsNotNone(state)
        self._mark_all_layers_passed(state)
        duplicate_label = state.match_flags[-2][0]
        state.match_flags[-1] = (
            duplicate_label,
            torch.tensor(True),
        )

        with (
            patch.object(
                model_runner_module,
                "get_tp_group",
                return_value=SimpleNamespace(
                    world_size=1,
                    cpu_group=object(),
                ),
            ),
            self.assertRaisesRegex(
                RuntimeError,
                "duplicate parity comparison labels",
            ),
        ):
            runner._finalize_staged_sfa_live_parity(state)

        self.assertEqual(
            runner._staged_sfa_live_parity_validated_seq_lens,
            [],
        )

    def test_finalize_propagates_remote_tp_failure_over_cpu_group(self):
        runner = self._build_runner()
        state = self._prepare(runner, 4096)
        self.assertIsNotNone(state)
        self._mark_all_layers_passed(state)
        cpu_group = object()

        def report_remote_failure(status, **_kwargs):
            status[0] = 1

        with (
            patch.object(
                model_runner_module,
                "get_tp_group",
                return_value=SimpleNamespace(
                    world_size=8,
                    cpu_group=cpu_group,
                ),
            ),
            patch.object(
                model_runner_module.dist,
                "all_reduce",
                side_effect=report_remote_failure,
            ) as all_reduce,
            self.assertRaisesRegex(
                RuntimeError,
                "none on this TP rank",
            ),
        ):
            runner._finalize_staged_sfa_live_parity(state)

        all_reduce.assert_called_once()
        self.assertEqual(
            all_reduce.call_args.kwargs["op"],
            model_runner_module.dist.ReduceOp.MAX,
        )
        self.assertIs(all_reduce.call_args.kwargs["group"], cpu_group)
        self.assertEqual(
            runner._staged_sfa_live_parity_validated_seq_lens,
            [],
        )

    def test_finalize_rejects_tp_checked_count_divergence(self):
        runner = self._build_runner()
        state = self._prepare(runner, 4096)
        self.assertIsNotNone(state)
        state.pending_saves.append(("layer-0", [torch.tensor([1.0])]))
        self._mark_all_layers_passed(state)
        cpu_group = object()

        def report_missing_remote_layers(status, **_kwargs):
            status[1] = 2
            status[2] = 0

        with (
            patch.object(
                model_runner_module,
                "get_tp_group",
                return_value=SimpleNamespace(
                    world_size=8,
                    cpu_group=cpu_group,
                ),
            ),
            patch.object(
                model_runner_module.dist,
                "all_reduce",
                side_effect=report_missing_remote_layers,
            ),
            patch.object(
                model_runner_module,
                "maybe_save_kv_layer_to_connector",
            ) as save_layer,
            self.assertRaisesRegex(
                RuntimeError,
                r"TP checked range=0\.\.2",
            ),
        ):
            runner._finalize_staged_sfa_live_parity(state)

        save_layer.assert_not_called()


if __name__ == "__main__":
    unittest.main()
