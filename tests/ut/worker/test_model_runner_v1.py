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
        return runner

    @staticmethod
    def _make_proved_impl():
        descriptor = BatchDescriptor(num_tokens=1)
        pre_entry = SimpleNamespace(aclgraph=object())
        post_entry = SimpleNamespace(aclgraph=object())
        return SimpleNamespace(
            enable_staged_sfa_graph=True,
            _staged_sfa_capture_phases={
                "pre:enter",
                "pre:exit",
                "post:enter",
                "post:exit",
            },
            _staged_sfa_replay_proved={"pre", "post"},
            _staged_sfa_pre_graph=SimpleNamespace(
                concrete_aclgraph_entries={descriptor: pre_entry},
            ),
            _staged_sfa_post_graph=SimpleNamespace(
                concrete_aclgraph_entries={descriptor: post_entry},
            ),
        )

    @patch.object(
        model_runner_module,
        "staged_sfa_graph_configured",
        return_value=True,
    )
    def test_accepts_complete_capture_for_every_local_sfa_layer(
        self,
        _staged_sfa_graph_configured,
    ):
        runner = self._build_runner()
        layers = {
            "model.layers.0.self_attn.attn": SimpleNamespace(
                impl=self._make_proved_impl(),
            ),
            "model.layers.1.self_attn.attn": SimpleNamespace(
                impl=self._make_proved_impl(),
            ),
        }
        with (
            patch.object(
                model_runner_module,
                "get_layers_from_vllm_config",
                return_value=layers,
            ),
            patch.object(model_runner_module.logger, "info") as info,
        ):
            runner._validate_staged_sfa_startup_capture()

        info.assert_called_once()
        self.assertEqual(info.call_args.args[1:], (2, 4))
        self.assertEqual(runner._staged_sfa_expected_layer_count, 2)
        self.assertEqual(len(runner._staged_sfa_impls), 2)

    @patch.object(
        model_runner_module,
        "staged_sfa_graph_configured",
        return_value=True,
    )
    def test_rejects_zero_local_staged_sfa_layers(
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
            self.assertRaisesRegex(
                RuntimeError,
                "did not find any local staged SFA implementations",
            ),
        ):
            runner._validate_staged_sfa_startup_capture()

    @patch.object(
        model_runner_module,
        "staged_sfa_graph_configured",
        return_value=True,
    )
    def test_rejects_partial_layer_capture_proof(
        self,
        _staged_sfa_graph_configured,
    ):
        runner = self._build_runner()
        impl = self._make_proved_impl()
        impl._staged_sfa_capture_phases.remove("post:exit")
        impl._staged_sfa_replay_proved.remove("post")
        impl._staged_sfa_post_graph.concrete_aclgraph_entries.clear()
        layers = {
            "model.layers.7.self_attn.attn": SimpleNamespace(impl=impl),
        }
        with (
            patch.object(
                model_runner_module,
                "get_layers_from_vllm_config",
                return_value=layers,
            ),
            self.assertRaisesRegex(
                RuntimeError,
                "model.layers.7.self_attn.attn: entries=\\['post'\\]",
            ),
        ):
            runner._validate_staged_sfa_startup_capture()

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

    def test_validation_delegates_to_shared_configuration_gate(self):
        runner = self._build_runner()
        with (
            patch.object(
                model_runner_module,
                "staged_sfa_graph_configured",
                return_value=False,
            ) as staged_sfa_graph_configured,
            patch.object(
                model_runner_module,
                "get_layers_from_vllm_config",
            ) as get_layers,
        ):
            runner._validate_staged_sfa_startup_capture()

        staged_sfa_graph_configured.assert_called_once_with(
            runner.vllm_config
        )
        get_layers.assert_not_called()

    def test_capture_model_preserves_parent_memory_result(self):
        runner = self._build_runner()
        with (
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
                model_runner_module.GPUModelRunner,
                "capture_model",
                return_value=123,
            ) as parent_capture,
            patch.object(
                runner,
                "_validate_staged_sfa_startup_capture",
            ) as validate_capture,
        ):
            result = runner.capture_model()

        self.assertEqual(result, 123)
        parent_capture.assert_called_once_with(runner)
        validate_capture.assert_called_once_with()


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
