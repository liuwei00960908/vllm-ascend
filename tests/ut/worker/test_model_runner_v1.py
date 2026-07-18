import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from vllm.config import CUDAGraphMode
from vllm.forward_context import BatchDescriptor
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec, KVCacheTensor

import vllm_ascend.worker.model_runner_v1 as model_runner_module
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


class TestStagedSFAStartupCaptureValidation(unittest.TestCase):

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

    @staticmethod
    def _layers(*impls):
        return {
            f"model.layers.{index}.self_attn.attn": SimpleNamespace(
                impl=impl,
            )
            for index, impl in enumerate(impls)
        }

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
        pre_entry = impl._staged_sfa_pre_graph.concrete_aclgraph_entries[
            descriptor
        ]
        pre_signatures = list(
            impl._staged_sfa_graph_input_signatures["pre"]
        )
        pre_signatures[-5] = (
            pre_signatures[-4][0],
            *pre_signatures[-5][1:],
        )
        impl._staged_sfa_graph_input_signatures["pre"] = tuple(
            pre_signatures
        )
        pre_entry.input_addresses[-5] = pre_signatures[-5][0]

        post_entry = impl._staged_sfa_post_graph.concrete_aclgraph_entries[
            descriptor
        ]
        post_signatures = list(
            impl._staged_sfa_graph_input_signatures["post"]
        )
        post_signatures[-1] = (
            post_signatures[-2][0],
            *post_signatures[-1][1:],
        )
        impl._staged_sfa_graph_input_signatures["post"] = tuple(
            post_signatures
        )
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
        pre_entry = impl._staged_sfa_pre_graph.concrete_aclgraph_entries[
            descriptor
        ]
        pre_entry.output = pre_outputs
        pre_signatures = list(
            impl._staged_sfa_graph_input_signatures["pre"]
        )
        expected_tail = (
            *pre_outputs,
            impl._staged_sfa_replay_canaries["pre"],
        )
        pre_signatures[-5:] = [
            self._signature(value) for value in expected_tail
        ]
        impl._staged_sfa_graph_input_signatures["pre"] = tuple(
            pre_signatures
        )
        pre_entry.input_addresses[-5:] = [
            value.data_ptr() for value in expected_tail
        ]

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
    def test_rejects_non_int32_vector_canary_signature(
        self,
        _staged_sfa_graph_configured,
    ):
        runner = self._build_runner()
        impl = self._make_captured_impl()
        descriptor = BatchDescriptor(num_tokens=1)
        malformed_canary = torch.ones((), dtype=torch.float32)
        impl._staged_sfa_replay_canaries["pre"] = malformed_canary
        pre_signatures = list(
            impl._staged_sfa_graph_input_signatures["pre"]
        )
        pre_signatures[-1] = self._signature(malformed_canary)
        impl._staged_sfa_graph_input_signatures["pre"] = tuple(
            pre_signatures
        )
        pre_entry = impl._staged_sfa_pre_graph.concrete_aclgraph_entries[
            descriptor
        ]
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
        impls = (self._make_captured_impl(), self._make_captured_impl())
        runner._staged_sfa_impls = tuple(
            (f"layer-{index}", impl)
            for index, impl in enumerate(impls)
        )

        def ordered_dummy_replay(*args, **kwargs):
            self.assertEqual(args, (1,))
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
        self.assertTrue(
            all(
                impl._staged_sfa_replay_proved == {"pre", "post"}
                for impl in impls
            )
        )
        info.assert_called_once_with(
            "[SFA staged graph POC] startup capture and ordered replay-canary "
            "completeness check passed for %d local SFA layers (%d staged "
            "graphs).",
            2,
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
        impl = self._make_captured_impl()
        runner._staged_sfa_impls = (("layer-0", impl),)
        runner._dummy_run = MagicMock(
            side_effect=lambda *_args, **_kwargs: impl._staged_sfa_replay_canaries[
                "pre"
            ].fill_(1)
        )
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
                    failed_suffix = failed_label[len(label_prefix):]
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
        runner.input_batch = SimpleNamespace(num_prompt_tokens=[4096])
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

    def test_first_two_distinct_lengths_commit_and_equal_length_retries(self):
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

        second = self._prepare(runner, 4097)
        self.assertIsNotNone(second)
        self._commit(runner, second)
        self.assertEqual(
            runner._staged_sfa_live_parity_validated_seq_lens,
            [4096, 4097],
        )
        self.assertIsNone(self._prepare(runner, 4098))

    def test_request_change_and_strict_decrease_reset_length_history(self):
        runner = self._build_runner()
        for seq_len in (4096, 4097):
            state = self._prepare(runner, seq_len, "request-a")
            self.assertIsNotNone(state)
            self._commit(runner, state)
        self.assertIsNone(self._prepare(runner, 4098, "request-a"))

        switched = self._prepare(runner, 6000, "request-b")
        self.assertIsNotNone(switched)
        self.assertEqual(
            runner._staged_sfa_live_parity_validated_seq_lens,
            [],
        )
        self._commit(runner, switched)
        self.assertEqual(
            runner._staged_sfa_live_parity_validated_seq_lens,
            [6000],
        )

        decreased = self._prepare(runner, 5999, "request-b")
        self.assertIsNotNone(decreased)
        self.assertEqual(
            runner._staged_sfa_live_parity_validated_seq_lens,
            [],
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

    def test_finalize_materializes_deferred_comparison_flags_once(self):
        runner = self._build_runner()
        state = self._prepare(runner, 4096)
        self.assertIsNotNone(state)
        self._mark_all_layers_passed(
            state,
            failed_label="layer-0: post.output",
        )
        state.pending_saves.append(
            ("layer-0", [torch.tensor([1.0])])
        )

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
        state.pending_saves.append(
            ("layer-0", [torch.tensor([1.0])])
        )
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
