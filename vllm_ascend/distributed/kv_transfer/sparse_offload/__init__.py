# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSA (DeepSeek Sparse Attention) latent KV offload for GLM5.1 on Ascend NPU.

See DESIGN.md in this directory for the full design. In short: the MLA latent KV is
offloaded to LMCache at prefill end and only the indexer-selected top-k tokens are
gathered back per decode step, while the small indexer-key cache stays resident.
"""

from vllm_ascend.distributed.kv_transfer.sparse_offload.offload_backend import (
    InMemoryLatentOffloadBackend,
    LatentOffloadBackend,
)

__all__ = ["LatentOffloadBackend", "InMemoryLatentOffloadBackend"]
