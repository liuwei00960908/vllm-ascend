import inspect
from typing import TYPE_CHECKING, Any

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    SupportsHMA,
)
from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import MultiConnector
from vllm.logger import init_logger
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

logger = init_logger(__name__)


def _is_single_tensor_kv(kv_cache: Any) -> bool:
    return isinstance(kv_cache, (tuple, list)) and len(kv_cache) < 2


def _block_group_counts(blocks: "KVCacheBlocks") -> tuple[int, ...]:
    return tuple(len(group) for group in blocks.blocks)


def _single_group_blocks(blocks: "KVCacheBlocks", group_idx: int) -> "KVCacheBlocks":
    if group_idx >= len(blocks.blocks):
        raise RuntimeError(
            f"Expected KV cache group {group_idx}, but only "
            f"{len(blocks.blocks)} groups exist."
        )
    return KVCacheBlocks((blocks.blocks[group_idx],))


def _has_remote_prefill_blocks(request: "Request") -> bool:
    params = getattr(request, "kv_transfer_params", None)
    if not isinstance(params, dict):
        return False
    required_keys = (
        "remote_engine_id",
        "remote_host",
        "remote_port",
        "remote_request_id",
    )
    return (
        bool(params.get("do_remote_prefill"))
        and bool(params.get("remote_block_ids"))
        and all(key in params and params[key] is not None for key in required_keys)
    )


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
        self._index_load_async_req_ids: set[str] = set()
        self._wait_for_layer_load_sig_cache: dict[
            tuple[type, int, tuple[str, ...]], bool
        ] = {}

        logger.info(
            "AscendMultiConnector initialized children: %s",
            [connector.__class__.__name__ for connector in self._connectors],
        )

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

        if has_index_connector:
            logger.info(
                "AscendMultiConnector DSA KV split: total_layers=%d "
                "latent_layers=%d indexer_layers=%d children=%s",
                len(kv_caches),
                len(latent_only),
                len(kv_caches) - len(latent_only),
                [connector.__class__.__name__ for connector in self._connectors],
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
        full_arg_children: list[str] = []
        layer_only_children: list[str] = []
        sig_cache = getattr(self, "_wait_for_layer_load_sig_cache", None)
        if sig_cache is None:
            sig_cache = {}
            self._wait_for_layer_load_sig_cache = sig_cache

        for connector in self._connectors:
            wait_for_layer_load = connector.wait_for_layer_load
            cache_key = (connector.__class__, 1 + len(args), tuple(sorted(kwargs)))
            accepts_args = sig_cache.get(cache_key)
            if accepts_args is None:
                accepts_args = _callable_accepts_args(
                    wait_for_layer_load,
                    1 + len(args),
                    set(kwargs),
                )
                sig_cache[cache_key] = accepts_args

            if accepts_args:
                wait_for_layer_load(layer_name, *args, **kwargs)
                full_arg_children.append(connector.__class__.__name__)
            else:
                wait_for_layer_load(layer_name)
                layer_only_children.append(connector.__class__.__name__)

        if args or kwargs:
            logger.info_once(
                "AscendMultiConnector sparse wait_for_layer_load dispatch: "
                "full_arg_children=%s layer_only_children=%s extra_pos_args=%d "
                "extra_kwargs=%s",
                ",".join(full_arg_children),
                ",".join(layer_only_children),
                len(args),
                ",".join(sorted(kwargs)),
                scope="local",
            )

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        to_return = (0, False)
        chosen_connector = -1
        for i, connector in enumerate(self._connectors):
            tokens, load_async = connector.get_num_new_matched_tokens(
                request,
                num_computed_tokens,
            )
            if tokens is None:
                return None, False
            if to_return[0] == 0 and tokens > 0:
                self._requests_to_connector[request.request_id] = i
                chosen_connector = i
                to_return = (tokens, load_async)

        tokens, load_async = to_return
        if (
            tokens > 0
            and not load_async
            and self._has_dsa_index_connector()
            and _has_remote_prefill_blocks(request)
        ):
            chosen_connector_name = (
                self._connectors[chosen_connector].__class__.__name__
                if 0 <= chosen_connector < len(self._connectors)
                else "none"
            )
            index_load_async_req_ids = getattr(
                self, "_index_load_async_req_ids", None
            )
            if index_load_async_req_ids is None:
                index_load_async_req_ids = set()
                self._index_load_async_req_ids = index_load_async_req_ids
            index_load_async_req_ids.add(request.request_id)
            params = request.kv_transfer_params
            logger.info(
                "AscendMultiConnector scheduling async DSA index load: "
                "request_id=%s external_tokens=%d chosen_connector=%s "
                "remote_index_blocks=%d",
                request.request_id,
                tokens,
                chosen_connector_name,
                len(params["remote_block_ids"]),
            )
            return tokens, True

        return to_return

    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ):
        chosen_connector = self._requests_to_connector.get(request.request_id, -1)
        chosen_connector_name = (
            self._connectors[chosen_connector].__class__.__name__
            if 0 <= chosen_connector < len(self._connectors)
            else "none"
        )
        params = getattr(request, "kv_transfer_params", None)
        do_remote_prefill = (
            params.get("do_remote_prefill") if params is not None else None
        )
        do_remote_decode = (
            params.get("do_remote_decode") if params is not None else None
        )
        index_load_async_req_ids = getattr(self, "_index_load_async_req_ids", set())
        skip_chosen_zero_update = (
            num_external_tokens == 0
            and request.request_id in index_load_async_req_ids
            and getattr(request, "num_computed_tokens", 0) > 0
            and chosen_connector >= 0
        )
        if (
            num_external_tokens > 0
            or chosen_connector >= 0
            or do_remote_prefill
            or do_remote_decode
        ):
            logger.info(
                "AscendMultiConnector alloc dispatch: request_id=%s "
                "external_tokens=%d block_groups=%s chosen_connector=%s "
                "do_remote_prefill=%s do_remote_decode=%s",
                request.request_id,
                num_external_tokens,
                _block_group_counts(blocks),
                chosen_connector_name,
                do_remote_prefill,
                do_remote_decode,
            )
        empty_blocks = blocks.new_empty()
        for i, connector in enumerate(self._connectors):
            if skip_chosen_zero_update and i == chosen_connector:
                logger.info(
                    "AscendMultiConnector preserving latent load state "
                    "after async DSA index load: request_id=%s "
                    "skipped_connector=%s",
                    request.request_id,
                    connector.__class__.__name__,
                )
                continue
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
        if skip_chosen_zero_update:
            index_load_async_req_ids.discard(request.request_id)

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        logger.info(
            "AscendMultiConnector finish dispatch: request_id=%s "
            "block_groups=%s children=%s",
            request.request_id,
            tuple(len(group) for group in block_ids),
            [connector.__class__.__name__ for connector in self._connectors],
        )
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
        index_load_async_req_ids = getattr(self, "_index_load_async_req_ids", None)
        if index_load_async_req_ids is not None:
            index_load_async_req_ids.discard(request.request_id)
        return async_saves > 0, kv_transfer_params
