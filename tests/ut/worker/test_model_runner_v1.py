import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec, KVCacheTensor

from vllm_ascend.worker.model_runner_v1 import MTPProfileCollector, NPUModelRunner


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


class TestNPUModelRunnerSpecDecode(unittest.TestCase):

    @patch("vllm_ascend.worker.model_runner_v1.logger.info")
    @patch("vllm_ascend.worker.model_runner_v1.torch.npu.synchronize")
    @patch("vllm_ascend.worker.model_runner_v1.torch.npu.Event")
    def test_mtp_profile_flushes_after_configured_steps(
        self,
        mock_event,
        mock_synchronize,
        mock_logger_info,
    ):
        event = MagicMock()
        event.elapsed_time.return_value = 2.5
        mock_event.return_value = event
        collector = MTPProfileCollector(enabled=True, max_steps=1, rank=0)

        collector.begin_step()
        with collector.section("target_forward"):
            pass
        collector.end_step()

        self.assertTrue(collector.flushed)
        mock_synchronize.assert_called_once()
        mock_logger_info.assert_called_once()
        self.assertIn("[MTP_PROFILE]", mock_logger_info.call_args.args[0])

    @patch("vllm_ascend.worker.model_runner_v1.RejectionSampler.parse_output")
    def test_sync_spec_decode_preserves_filtered_logprobs(self, mock_parse_output):
        runner = NPUModelRunner.__new__(NPUModelRunner)
        runner.use_async_scheduling = False
        runner.num_discarded_requests = 0
        runner.discard_request_indices = SimpleNamespace(
            np=np.empty(0, dtype=np.int64)
        )
        runner.max_model_len = 16
        runner.input_batch = SimpleNamespace(
            generators={},
            req_ids=["request-0"],
            req_id_to_index={"request-0": 0},
            vocab_size=128,
            num_tokens_no_spec=np.array([0]),
            token_ids_cpu=np.zeros((1, 16), dtype=np.int64),
            is_token_ids=np.zeros((1, 16), dtype=bool),
            num_tokens=np.array([0]),
        )
        runner.requests = {"request-0": SimpleNamespace(output_token_ids=[])}
        runner._get_prompt_logprobs_dict = MagicMock(return_value={})

        filtered_logprobs = MagicMock(name="filtered_logprobs")
        mock_parse_output.return_value = ([[1, 2]], filtered_logprobs)
        logprobs_tensors = MagicMock(name="logprobs_tensors")
        sampler_output = SimpleNamespace(
            sampled_token_ids=torch.tensor([[1, 2]]),
            logprobs_tensors=logprobs_tensors,
        )
        scheduler_output = SimpleNamespace(num_scheduled_tokens={"request-0": 2})

        (
            logprobs_lists,
            valid_sampled_token_ids,
            _,
            _,
            _,
            _,
        ) = runner._bookkeeping_sync(
            scheduler_output,
            sampler_output,
            logits=None,
            hidden_states=torch.empty((2, 1)),
            num_scheduled_tokens=2,
            spec_decode_metadata=MagicMock(),
        )

        self.assertEqual(valid_sampled_token_ids, [[1, 2]])
        self.assertIs(logprobs_lists, filtered_logprobs)
        logprobs_tensors.tolists.assert_not_called()


if __name__ == "__main__":
    unittest.main()
