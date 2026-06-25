from typing import TYPE_CHECKING, Any

from vllm.distributed.kv_transfer.kv_connector.v1.base import SupportsHMA
from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import MultiConnector
from vllm.v1.core.kv_cache_manager import KVCacheBlocks

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_dsa_index_connector import (
    MooncakeDSAIndexConnector,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector import (
    MooncakeLayerwiseConnector,
)

if TYPE_CHECKING:
    from vllm.v1.request import Request


def _is_single_tensor_kv(kv_cache: Any) -> bool:
    return isinstance(kv_cache, (tuple, list)) and len(kv_cache) < 2


def _single_group_blocks(blocks: "KVCacheBlocks", group_idx: int) -> "KVCacheBlocks":
    if group_idx >= len(blocks.blocks):
        raise RuntimeError(
            f"Expected KV cache group {group_idx}, but only "
            f"{len(blocks.blocks)} groups exist."
        )
    return KVCacheBlocks((blocks.blocks[group_idx],))


class AscendMultiConnector(MultiConnector, SupportsHMA):
    # DSA unbundle needs the model runner to pass both latent and indexer KV
    # caches so this connector can route them to different children.
    requires_full_dsa_kv_caches = True

    def _has_dsa_index_connector(self) -> bool:
        return any(isinstance(c, MooncakeDSAIndexConnector) for c in self._connectors)

    def _blocks_for_connector(
        self,
        connector: Any,
        blocks: "KVCacheBlocks",
    ) -> "KVCacheBlocks":
        if isinstance(connector, SupportsHMA):
            return blocks
        return _single_group_blocks(blocks, 0)

    def _should_receive_alloc_update(self, connector: Any) -> bool:
        return isinstance(
            connector,
            (MooncakeLayerwiseConnector, MooncakeDSAIndexConnector),
        )

    def _merge_kv_transfer_params(
        self,
        merged: dict[str, Any] | None,
        new_params: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        if new_params is None:
            return merged
        if merged is None:
            return dict(new_params)
        for key, value in new_params.items():
            if key in merged and merged[key] != value:
                raise RuntimeError(
                    "Multiple connectors produced conflicting KV transfer "
                    f"params for key {key!r}: {merged[key]!r} != {value!r}"
                )
            merged[key] = value
        return merged

    def register_kv_caches(self, kv_caches: dict):
        has_index_connector = self._has_dsa_index_connector()
        latent_only = (
            {
                name: kv
                for name, kv in kv_caches.items()
                if not _is_single_tensor_kv(kv)
            }
            if has_index_connector
            else kv_caches
        )

        for connector in self._connectors:
            if isinstance(connector, MooncakeDSAIndexConnector):
                connector.register_kv_caches(kv_caches)
            else:
                connector.register_kv_caches(latent_only)

    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ):
        chosen_connector = self._requests_to_connector.get(request.request_id, -1)
        empty_blocks = blocks.new_empty()
        for i, connector in enumerate(self._connectors):
            should_update = i == chosen_connector or self._should_receive_alloc_update(
                connector
            )
            target_blocks = blocks if should_update else empty_blocks
            target_tokens = num_external_tokens if should_update else 0
            connector.update_state_after_alloc(
                request,
                self._blocks_for_connector(connector, target_blocks),
                target_tokens,
            )

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        async_saves = 0
        kv_transfer_params: dict[str, Any] | None = None

        for connector in self._connectors:
            if isinstance(connector, SupportsHMA):
                async_save, txfer_params = connector.request_finished_all_groups(
                    request, block_ids
                )
            else:
                async_save, txfer_params = connector.request_finished(
                    request, block_ids[0]
                )

            if async_save:
                async_saves += 1
            kv_transfer_params = self._merge_kv_transfer_params(
                kv_transfer_params, txfer_params
            )

        if async_saves > 1:
            self._extra_async_saves[request.request_id] = async_saves - 1

        self._requests_to_connector.pop(request.request_id, None)
        return async_saves > 0, kv_transfer_params
