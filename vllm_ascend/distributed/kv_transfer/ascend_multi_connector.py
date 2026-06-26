import inspect
from typing import TYPE_CHECKING, Any

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    SupportsHMA,
)
from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import MultiConnector
from vllm.utils.func_utils import supports_kw
from vllm.v1.core.kv_cache_manager import KVCacheBlocks

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_dsa_index_connector import (
    MooncakeDSAIndexConnector,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector import (
    MooncakeLayerwiseConnector,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
    from vllm.v1.kv_cache_interface import KVCacheConfig
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


def _callable_accepts_args(
    func: Any,
    num_positional_args: int,
    keyword_names: set[str],
) -> bool:
    try:
        params = inspect.signature(func).parameters.values()
    except (TypeError, ValueError):
        return False

    positional_count = 0
    accepts_varargs = False
    accepts_varkw = False
    accepted_keywords = set()
    for param in params:
        if param.kind == inspect.Parameter.VAR_POSITIONAL:
            accepts_varargs = True
        elif param.kind == inspect.Parameter.VAR_KEYWORD:
            accepts_varkw = True
        elif param.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            positional_count += 1
            accepted_keywords.add(param.name)
        elif param.kind == inspect.Parameter.KEYWORD_ONLY:
            accepted_keywords.add(param.name)

    accepts_positional = accepts_varargs or positional_count >= num_positional_args
    accepts_keywords = accepts_varkw or keyword_names.issubset(accepted_keywords)
    return accepts_positional and accepts_keywords


class AscendMultiConnector(MultiConnector, SupportsHMA):
    # DSA unbundle needs the model runner to pass both latent and indexer KV
    # caches so this connector can route them to different children.
    requires_full_dsa_kv_caches = True

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: "KVConnectorRole",
        kv_cache_config: "KVCacheConfig | None" = None,
    ):
        KVConnectorBase_V1.__init__(
            self,
            vllm_config=vllm_config,
            role=role,
            kv_cache_config=kv_cache_config,
        )

        self._connectors: list[KVConnectorBase_V1] = []
        self._ktc_kv_transfer_config = []
        for connector_cls, temp_config in self._get_connector_classes_and_configs(
            vllm_config
        ):
            if supports_kw(connector_cls, "kv_cache_config"):
                connector = connector_cls(
                    temp_config,
                    role,
                    kv_cache_config=kv_cache_config,
                )
            else:
                connector = connector_cls(temp_config, role)
            self._connectors.append(connector)
            self._ktc_kv_transfer_config.append(temp_config.kv_transfer_config)

        # A mapping from request id to the index of the connector chosen to
        # load the request from (if any).
        self._requests_to_connector: dict[str, int] = {}

        # Tracks additional async saves beyond the first one. This mirrors
        # MultiConnector while allowing legacy child connector constructors.
        self._extra_async_saves: dict[str, int] = {}

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

    def wait_for_layer_load(
        self,
        layer_name: str,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        for connector in self._connectors:
            wait_for_layer_load = connector.wait_for_layer_load
            if _callable_accepts_args(
                wait_for_layer_load,
                1 + len(args),
                set(kwargs),
            ):
                wait_for_layer_load(layer_name, *args, **kwargs)
            else:
                wait_for_layer_load(layer_name)

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
