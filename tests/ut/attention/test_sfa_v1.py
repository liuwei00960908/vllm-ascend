import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from vllm.distributed.parallel_state import GroupCoordinator

from tests.ut.attention.utils import patch_distributed_groups
from tests.ut.base import TestBase
from vllm_ascend.attention.attention_v1 import AscendAttentionState

if "torch_npu._inductor" not in sys.modules:
    sys.modules["torch_npu._inductor"] = MagicMock()

import vllm_ascend.attention.sfa_v1 as sfa_v1
from vllm_ascend.attention.sfa_v1 import (
    AscendSFABackend,
    AscendSFAImpl,
    AscendSFAMetadata,
    AscendSFAMetadataBuilder,
    _update_dsa_split_boundary_in_place,
)
from vllm_ascend.utils import enable_dsa_cp


def test_sparse_boundary_updates_preallocated_storage_in_place():
    boundary_cpu = torch.tensor([9, 9, 19, 0], dtype=torch.int32)
    boundary = torch.empty(4, dtype=torch.int32)
    boundary.copy_(boundary_cpu)
    metadata = SimpleNamespace(
        split_boundary=boundary,
        decode_split_boundary_cpu=boundary_cpu.numpy(),
        decode_split_boundary_cpu_tensor=boundary_cpu,
        decode_req_indices_cpu=torch.tensor([0, 0, 1, -1], dtype=torch.int32).numpy(),
        seq_lens_cpu=torch.tensor([513, 770], dtype=torch.int32),
        num_decode_tokens=3,
        decode_split_boundary=None,
    )
    address = metadata.split_boundary.data_ptr()

    with (
        patch.object(
            sfa_v1.torch,
            "tensor",
            side_effect=AssertionError("unexpected torch.tensor"),
        ),
        patch.object(
            sfa_v1.torch,
            "arange",
            side_effect=AssertionError("unexpected torch.arange"),
        ),
        patch.object(
            sfa_v1.torch.nn.functional,
            "pad",
            side_effect=AssertionError("unexpected pad"),
        ),
    ):
        actual = _update_dsa_split_boundary_in_place(
            metadata,
            cached_tokens=[512, 768],
            decode_window_size=256,
        )

    assert actual.data_ptr() == address
    assert metadata.decode_split_boundary.data_ptr() == address
    assert actual.tolist() == [512, 512, 768, 0]


def test_sparse_boundary_short_frontier_preserves_zero_pad_semantics():
    boundary_cpu = torch.tensor([9, 19], dtype=torch.int32)
    boundary = boundary_cpu.clone()
    metadata = SimpleNamespace(
        split_boundary=boundary,
        decode_split_boundary_cpu=boundary_cpu.numpy(),
        decode_split_boundary_cpu_tensor=boundary_cpu,
        decode_req_indices_cpu=torch.tensor([0, 1], dtype=torch.int32).numpy(),
        seq_lens_cpu=torch.tensor([10, 20], dtype=torch.int32),
        num_decode_tokens=2,
        decode_split_boundary=None,
    )

    actual = _update_dsa_split_boundary_in_place(
        metadata,
        cached_tokens=[8],
        decode_window_size=0,
    )

    assert actual.tolist() == [8, 0]


def test_sparse_boundary_prefers_explicit_committed_end():
    from vllm_ascend.attention import utils as attention_utils

    metadata = SimpleNamespace(
        requests=[
            SimpleNamespace(
                req_id="resident",
                is_sparse_decode=True,
                load_spec=SimpleNamespace(
                    can_load=True,
                    lmcache_cached_tokens=3072,
                    dsa_committed_end=0,
                ),
            ),
            SimpleNamespace(
                req_id="offloaded",
                is_sparse_decode=True,
                load_spec=SimpleNamespace(
                    can_load=True,
                    lmcache_cached_tokens=8192,
                    dsa_committed_end=8192,
                ),
            ),
        ]
    )
    connector = SimpleNamespace(_get_connector_metadata=lambda: metadata)
    with (
        patch.object(attention_utils, "has_kv_transfer_group", return_value=True),
        patch.object(attention_utils, "is_v1_kv_transfer_group", return_value=True),
        patch.object(attention_utils, "get_kv_transfer_group", return_value=connector),
    ):
        assert attention_utils.get_lmcache_sparse_cached_tokens(["resident", "offloaded"]) == [0, 8192]


class TestLMCacheSparseWaitSync(TestBase):
    def setUp(self):
        self.original_once_done = sfa_v1._lmcache_sparse_wait_sync_once_done
        sfa_v1._lmcache_sparse_wait_sync_once_done = False

    def tearDown(self):
        sfa_v1._lmcache_sparse_wait_sync_once_done = self.original_once_done

    def test_once_mode_synchronizes_only_first_sparse_wait(self):
        stream = MagicMock()
        with (
            patch.object(sfa_v1, "_LMCACHE_SPARSE_WAIT_SYNC_ONCE", True),
            patch.object(
                sfa_v1.torch.npu,
                "current_stream",
                return_value=stream,
            ) as current_stream,
        ):
            sfa_v1._sync_compute_stream_after_lmcache_sparse_wait()
            sfa_v1._sync_compute_stream_after_lmcache_sparse_wait()

        current_stream.assert_called_once_with()
        stream.synchronize.assert_called_once_with()
        self.assertTrue(sfa_v1._lmcache_sparse_wait_sync_once_done)

    def test_completed_mode_does_not_touch_npu_stream(self):
        sfa_v1._lmcache_sparse_wait_sync_once_done = True
        with patch.object(sfa_v1.torch.npu, "current_stream") as current_stream:
            sfa_v1._sync_compute_stream_after_lmcache_sparse_wait()

        current_stream.assert_not_called()

    def test_disabled_mode_does_not_synchronize(self):
        with (
            patch.object(sfa_v1, "_LMCACHE_SPARSE_WAIT_SYNC_ONCE", False),
            patch.object(sfa_v1.torch.npu, "current_stream") as current_stream,
        ):
            sfa_v1._sync_compute_stream_after_lmcache_sparse_wait()

        current_stream.assert_not_called()
        self.assertFalse(sfa_v1._lmcache_sparse_wait_sync_once_done)

    def test_sync_compute_stream_skips_when_npu_unavailable(self):
        with (
            patch.object(sfa_v1, "_LMCACHE_SPARSE_WAIT_SYNC_ONCE", True),
            patch.object(sfa_v1.torch, "npu", None),
        ):
            sfa_v1._sync_compute_stream_after_lmcache_sparse_wait()

        self.assertFalse(sfa_v1._lmcache_sparse_wait_sync_once_done)


class TestDSASparsePadding(TestBase):
    def test_trailing_graph_padding_is_zeroed_in_place(self):
        topk = torch.arange(4 * 64, dtype=torch.int32).reshape(4, 1, 64)
        original_actual = topk[:2].clone()
        input_ptr = topk.data_ptr()

        result, result_2d = sfa_v1._dsa_mask_padding_sparse_rows(
            topk,
            torch.tensor([0, 1, -1, -1], dtype=torch.int32),
        )

        self.assertEqual(result.data_ptr(), input_ptr)
        self.assertEqual(result_2d.data_ptr(), input_ptr)
        self.assertTrue(torch.equal(result[:2], original_actual))
        self.assertEqual(torch.count_nonzero(result[2:]).item(), 0)


class TestAscendSFABackend(TestBase):
    def test_get_name(self):
        self.assertEqual(AscendSFABackend.get_name(), "ASCEND_SFA")

    def test_get_builder_cls(self):
        self.assertEqual(AscendSFABackend.get_builder_cls(), AscendSFAMetadataBuilder)

    def test_get_kv_cache_shape(self):
        result = AscendSFABackend.get_kv_cache_shape(2, 4, 8, 128)
        self.assertEqual(result, (2, 4, 8, 128))

    def test_get_impl_cls(self):
        result = AscendSFABackend.get_impl_cls()
        self.assertEqual(result, AscendSFAImpl)


class TestAscendSFAMetadata(TestBase):
    def test_ascend_sfa_metadata_default(self):
        num_actual_tokens = 100
        slot_mapping = torch.randn(100, 4, 1024)
        seq_lens = torch.tensor([30, 50])
        cum_query_lens = torch.tensor([0, 30, 80])
        block_table = torch.randint(0, 100, (100, 4))

        rope_dim = 32
        max_seq_len = int(seq_lens.max().item())
        sin = torch.randn(max_seq_len, rope_dim)
        cos = torch.randn(max_seq_len, rope_dim)

        num_input_tokens = 2
        head_dim = None
        attn_mask = None
        attn_state = AscendAttentionState.ChunkedPrefill

        metadata = AscendSFAMetadata(
            num_actual_tokens=num_actual_tokens,
            slot_mapping=slot_mapping,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens,
            cum_query_lens=cum_query_lens,
            block_table=block_table,
            sin=sin,
            cos=cos,
            num_input_tokens=num_input_tokens,
            head_dim=head_dim,
            attn_mask=attn_mask,
            attn_state=attn_state,
        )

        self.assertEqual(metadata.num_actual_tokens, num_actual_tokens)
        self.assertIs(metadata.slot_mapping, slot_mapping)
        self.assertTrue(torch.equal(metadata.seq_lens, seq_lens))
        self.assertTrue(torch.equal(metadata.cum_query_lens, cum_query_lens))
        self.assertIs(metadata.block_table, block_table)
        self.assertIs(metadata.sin, sin)
        self.assertIs(metadata.cos, cos)
        self.assertEqual(metadata.num_input_tokens, num_input_tokens)
        self.assertIs(metadata.head_dim, head_dim)
        self.assertIs(metadata.attn_mask, attn_mask)
        self.assertEqual(metadata.attn_state, attn_state)


class TestAscendSFAMetadataBuilder(TestBase):
    @patch("vllm.distributed.parallel_state._TP", new_callable=lambda: MagicMock(spec=GroupCoordinator))
    def setUp(self, mock_tp):
        mock_tp.world_size = 2
        mock_tp.rank_in_group = MagicMock()
        mock_tp.device_group = MagicMock()

        self.mock_cfg = MagicMock()

        self.mock_cfg.parallel_config = MagicMock()
        self.mock_cfg.parallel_config.tensor_parallel_size = 1
        self.mock_cfg.parallel_config.prefill_context_parallel_size = 1
        self.mock_cfg.parallel_config.decode_context_parallel_size = 1

        self.mock_cfg.compilation_config = MagicMock()
        self.mock_cfg.compilation_config.pass_config = MagicMock()
        self.mock_cfg.compilation_config.pass_config.enable_sp = False

        self.mock_cfg.speculative_config.num_speculative_tokens = 0

        self.patcher = patch("vllm.config.get_current_vllm_config", return_value=self.mock_cfg)
        self.patcher.start()

        # Mock parent class __init__ to avoid complex initialization,
        # but still set the essential attributes that child class needs
        def mock_parent_init(
            self, kv_cache_spec, layer_names, vllm_config, device, metadata_cls, supports_dcp_with_varlen
        ):
            self.metadata_cls = metadata_cls
            self.kv_cache_spec = kv_cache_spec
            self.model_config = vllm_config.model_config
            self.vllm_config = vllm_config
            self.device = device
            self.chunked_prefill_workspace_size = 128 * 1024
            self.chunked_prefill_workspace = torch.empty(
                (self.chunked_prefill_workspace_size, vllm_config.model_config.get_head_size()),
                dtype=vllm_config.model_config.dtype,
                device=device,
            )

        self.parent_init_patcher = patch(
            "vllm.model_executor.layers.attention.mla_attention.MLACommonMetadataBuilder.__init__", mock_parent_init
        )
        self.parent_init_patcher.start()

        if hasattr(enable_dsa_cp, "cache_clear"):
            enable_dsa_cp.cache_clear()

    def tearDown(self):
        self.patcher.stop()
        self.parent_init_patcher.stop()

    @patch("vllm_ascend.attention.sfa_v1.is_v1_kv_transfer_group")
    @patch("vllm_ascend.attention.sfa_v1.has_kv_transfer_group")
    @patch("vllm_ascend.attention.sfa_v1.get_cos_and_sin_mla")
    @patch("vllm_ascend.attention.sfa_v1.enable_dsa_cp")
    def test_dsa_sparse_metadata_reuses_builder_storage(
        self,
        mock_enable_dsa_cp,
        mock_get_cos_and_sin_mla,
        mock_has_kv_transfer_group,
        mock_is_v1_kv_transfer_group,
    ):
        mock_enable_dsa_cp.return_value = False
        mock_has_kv_transfer_group.return_value = True
        mock_is_v1_kv_transfer_group.return_value = True
        mock_get_cos_and_sin_mla.side_effect = lambda positions, _: (
            torch.zeros_like(positions),
            torch.zeros_like(positions),
        )

        kv_cache_spec = MagicMock()
        vllm_config = MagicMock()
        vllm_config.cache_config.block_size = 16
        vllm_config.model_config.max_model_len = 1024
        vllm_config.model_config.get_head_size.return_value = 64
        vllm_config.model_config.dtype = torch.float16
        vllm_config.model_config.hf_text_config.qk_rope_head_dim = 64
        vllm_config.model_config.hf_text_config.topk_tokens = 16
        vllm_config.speculative_config.num_speculative_tokens = 1
        vllm_config.scheduler_config.max_num_seqs = 4
        vllm_config.scheduler_config.max_num_batched_tokens = 8

        with patch.dict(
            os.environ,
            {
                "VLLM_ASCEND_DSA_UNBUNDLE": "1",
                "VLLM_ASCEND_DSA_SHRINK_LATENT": "2",
            },
        ):
            builder = AscendSFAMetadataBuilder(
                kv_cache_spec=kv_cache_spec,
                layer_names=["layer1", "layer2"],
                vllm_config=vllm_config,
                device=torch.device("cpu"),
            )
        builder.attn_mask_builder.get_attention_mask = MagicMock(return_value=None)

        def common_metadata(
            query_start_loc,
            computed,
            prompt_lens,
            request_ids,
        ):
            num_reqs = len(request_ids)
            num_tokens = int(query_start_loc[-1])
            return SimpleNamespace(
                num_reqs=num_reqs,
                num_actual_tokens=num_tokens,
                num_input_tokens=num_tokens,
                block_table_tensor=torch.zeros((num_reqs, 4), dtype=torch.int32),
                slot_mapping=torch.arange(num_tokens, dtype=torch.int64),
                positions=torch.arange(num_tokens, dtype=torch.int64),
                indexer_block_table_tensor=None,
                indexer_slot_mapping=None,
                prompt_lens_cpu=prompt_lens,
                query_start_loc_cpu=torch.tensor(query_start_loc, dtype=torch.int32),
                num_computed_tokens_cpu=torch.tensor(computed, dtype=torch.int32),
                query_start_loc=torch.tensor(query_start_loc, dtype=torch.int32),
                seq_lens=torch.tensor(computed, dtype=torch.int32),
                seq_lens_cpu=torch.tensor(computed, dtype=torch.int32),
                request_ids=request_ids,
                attn_state=AscendAttentionState.DecodeOnly,
            )

        first = builder.build(
            common_prefix_len=0,
            common_attn_metadata=common_metadata(
                [0, 2, 4],
                [10, 20],
                [9, 19],
                ["r0", "r1"],
            ),
        )
        addresses = (
            first.split_boundary.data_ptr(),
            first.decode_req_indices.data_ptr(),
            first.decode_row_offsets.data_ptr(),
            first.decode_selected_tokens.data_ptr(),
            first.decode_selected_counts.data_ptr(),
            first.decode_target_slot_mapping.data_ptr(),
        )
        assert first.decode_req_indices.tolist() == [0, 0, 1, 1]
        assert first.decode_row_offsets.tolist() == [0, 1, 0, 1]
        assert first.num_decode_tokens == 4

        second = builder.build(
            common_prefix_len=0,
            common_attn_metadata=common_metadata(
                [0, 1],
                [5],
                [4],
                ["r2"],
            ),
        )
        second_addresses = (
            second.split_boundary.data_ptr(),
            second.decode_req_indices.data_ptr(),
            second.decode_row_offsets.data_ptr(),
            second.decode_selected_tokens.data_ptr(),
            second.decode_selected_counts.data_ptr(),
            second.decode_target_slot_mapping.data_ptr(),
        )

        assert second_addresses == addresses
        assert second.split_boundary.tolist() == [4]
        assert second.decode_req_indices.tolist() == [0]
        assert second.decode_row_offsets.tolist() == [0]
        assert second.num_decode_tokens == 1

        third = builder.build(
            common_prefix_len=0,
            common_attn_metadata=common_metadata(
                [0, 2, 4],
                [12, 22],
                [11, 21],
                ["r3", "r4"],
            ),
        )
        assert third.decode_req_indices.tolist() == [0, 0, 1, 1]
        assert third.decode_row_offsets.tolist() == [0, 1, 0, 1]
        assert third.split_boundary.tolist() == [11, 11, 21, 21]

        with self.assertRaisesRegex(RuntimeError, "max_num_batched_tokens=8"):
            builder.build(
                common_prefix_len=0,
                common_attn_metadata=common_metadata(
                    [0, 9],
                    [9],
                    [8],
                    ["too-large"],
                ),
            )

        with self.assertRaisesRegex(RuntimeError, "max_num_seqs=4"):
            builder.build(
                common_prefix_len=0,
                common_attn_metadata=common_metadata(
                    [0, 1, 2, 3, 4, 5],
                    [1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0],
                    ["r0", "r1", "r2", "r3", "r4"],
                ),
            )

    @patch_distributed_groups(dcp_size=2, pcp_size=2, needs_mocks=False)
    def test_ascend_sfa_metadata_builder_default(self):
        kv_cache_spec = MagicMock()
        layer_names = ["layer1", "layer2"]
        vllm_config = MagicMock()
        vllm_config.cache_config.block_size = 16
        vllm_config.model_config.max_model_len = 1024
        vllm_config.model_config.get_head_size.return_value = 64
        vllm_config.model_config.dtype = torch.float16
        vllm_config.model_config.hf_text_config.qk_rope_head_dim = 64
        speculative_config = MagicMock()
        speculative_config.num_speculative_tokens = 4
        vllm_config.speculative_config = speculative_config
        device = torch.device("cpu")

        builder = AscendSFAMetadataBuilder(
            kv_cache_spec=kv_cache_spec, layer_names=layer_names, vllm_config=vllm_config, device=device
        )

        assert builder.device == device
        assert builder.vllm_config == vllm_config

    @patch("vllm_ascend.attention.sfa_v1.get_current_vllm_config")
    @patch("vllm_ascend.attention.sfa_v1.get_cos_and_sin_mla")
    @patch("vllm_ascend.attention.sfa_v1.enable_dsa_cp")
    @patch_distributed_groups(dcp_size=2, pcp_size=2, needs_mocks=False)
    def test_ascend_sfa_metadata_builder_build(
        self,
        mock_enable_dsa_cp,
        mock_get_cos_and_sin_mla,
        mock_get_current_vllm_config,
    ):
        mock_enable_dsa_cp.return_value = False

        cfg = MagicMock()
        cfg.model_config = MagicMock()
        cfg.model_config.hf_text_config = MagicMock()

        mock_get_current_vllm_config.return_value = cfg
        kv_cache_spec = MagicMock()
        layer_names = ["layer1", "layer2"]
        vllm_config = MagicMock()
        vllm_config.cache_config.block_size = 16
        vllm_config.model_config.max_model_len = 1024
        vllm_config.model_config.get_head_size.return_value = 64
        vllm_config.model_config.dtype = torch.float16
        vllm_config.model_config.hf_text_config.qk_rope_head_dim = 64
        speculative_config = MagicMock()
        speculative_config.num_speculative_tokens = 4
        vllm_config.speculative_config = speculative_config
        device = torch.device("cpu")

        builder = AscendSFAMetadataBuilder(
            kv_cache_spec=kv_cache_spec, layer_names=layer_names, vllm_config=vllm_config, device=device
        )

        common_attn_metadata = MagicMock()
        common_attn_metadata.num_reqs = 10
        common_attn_metadata.num_actual_tokens = 100
        common_attn_metadata.query_start_loc = torch.tensor([0, 10, 20, 30, 40, 50, 60, 70, 80, 90])
        common_attn_metadata.query_start_loc_cpu = torch.tensor([0, 10, 20, 30, 40, 50, 60, 70, 80, 90])
        common_attn_metadata.slot_mapping = torch.randn(100, 4, 1024)
        common_attn_metadata.seq_lens_cpu = torch.tensor([2] * 10)
        common_attn_metadata.positions = torch.randn(100)
        common_attn_metadata.attn_mask = None
        common_attn_metadata.attn_state = AscendAttentionState.ChunkedPrefill
        common_attn_metadata.block_table_tensor = torch.randn(100, 4)
        common_attn_metadata.cos = None
        common_attn_metadata.sin = None
        common_attn_metadata.num_input_tokens = 100

        mock_get_cos_and_sin_mla.return_value = (torch.randn(100), torch.randn(100))

        metadata = builder.build(
            common_prefix_len=10,
            common_attn_metadata=common_attn_metadata,
        )

        assert isinstance(metadata, AscendSFAMetadata)
        assert metadata.num_actual_tokens == common_attn_metadata.num_actual_tokens
        assert metadata.slot_mapping.shape == (100, 4, 1024)

    @patch("vllm_ascend.attention.sfa_v1.get_current_vllm_config")
    @patch("vllm_ascend.attention.sfa_v1.get_cos_and_sin_mla")
    @patch_distributed_groups(dcp_size=2, pcp_size=2, needs_mocks=False)
    def test_ascend_sfa_metadata_builder_build_for_graph_capture(
        self, mock_get_cos_and_sin_mla, mock_get_current_vllm_config
    ):
        cfg = MagicMock()
        cfg.model_config = MagicMock()
        cfg.model_config.hf_text_config = MagicMock()

        mock_get_current_vllm_config.return_value = cfg

        kv_cache_spec = MagicMock()
        layer_names = ["layer1", "layer2"]
        vllm_config = MagicMock()
        vllm_config.cache_config.block_size = 16
        vllm_config.model_config.max_model_len = 1024
        vllm_config.model_config.get_head_size.return_value = 64
        vllm_config.model_config.dtype = torch.float16
        vllm_config.model_config.hf_text_config.qk_rope_head_dim = 64
        speculative_config = MagicMock()
        speculative_config.num_speculative_tokens = 4
        vllm_config.speculative_config = speculative_config
        device = torch.device("cpu")

        builder = AscendSFAMetadataBuilder(
            kv_cache_spec=kv_cache_spec, layer_names=layer_names, vllm_config=vllm_config, device=device
        )

        common_attn_metadata = MagicMock()
        common_attn_metadata.num_reqs = 10
        common_attn_metadata.num_actual_tokens = 100
        common_attn_metadata.query_start_loc = torch.tensor([0, 10, 20, 30, 40, 50, 60, 70, 80, 90])
        common_attn_metadata.query_start_loc_cpu = torch.tensor([0, 10, 20, 30, 40, 50, 60, 70, 80, 90])
        common_attn_metadata.slot_mapping = torch.randn(100, 4, 1024)
        common_attn_metadata.seq_lens_cpu = torch.tensor([2] * 10)
        common_attn_metadata.positions = torch.randn(100)
        common_attn_metadata.attn_mask = None
        common_attn_metadata.attn_state = AscendAttentionState.ChunkedPrefill
        common_attn_metadata.block_table_tensor = torch.randn(100, 4)
        common_attn_metadata.cos = None
        common_attn_metadata.sin = None
        common_attn_metadata.num_input_tokens = 100

        mock_get_cos_and_sin_mla.return_value = (torch.randn(100), torch.randn(100))

        attn_metadata = builder.build_for_graph_capture(
            common_attn_metadata=common_attn_metadata,
            attn_state=AscendAttentionState.DecodeOnly,
        )

        assert isinstance(attn_metadata, AscendSFAMetadata)
        assert attn_metadata.attn_state == AscendAttentionState.DecodeOnly
