# Production design and delivery roadmap for staged SFA ACL graphs

## Executive decision

The production feature is a **model-step executor**, not a conditional branch
inside `AscendSFAImpl.forward`. `sfa_v1.py` should own two replayable kernels and
their layer-local tensor contract. The model runner must own admission, graph
selection, stable buffers, TP agreement, connector progress, failure handling,
and step finalization.

The required per-layer order is:

```text
Static preflight -> connector frontier/source-domain lease -> final TP admission
                  (preparation is reversible; no cache/cursor side effect)
                                  |
                                  v
Eager 0: wait/materialize the LMCache index group for this layer,
         then obtain a rank-consistent success/failure verdict
                                  |
                                  v
Graph A: Q/K preprocessing, current-token KV/index writes, indexer/top-k,
         inactive-row mask, scratch remap, fixed-capacity selection output
                                  |
                                  v
Eager 1: selective LMCache latent load into target scratch slots,
         Graph-A producer event -> load stream -> rank-consistent verdict
                                  |
                                  v
Graph B: scratch-backed sparse attention, value projection, output projection
         completion event defers layer finish, lease release, and slot reuse
                                  |
                                  v
Layer/step connector commit and deferred saves
```

The eager index wait is before Graph A because Graph A reads the indexer cache.
The selective latent load is after Graph A because it consumes Graph A's top-k
selection. A single full-model ACL graph is not a valid target until LMCache
offers device-driven, graph-safe retrieval.

Production-ready does not mean every vLLM mode must be captured in the first
release. It means the enabled envelope is correct, bounded, observable, and
recoverable, while every other combination receives a proven pre-mutation
native route, request recomputation route, or explicit rejection. Merely
removing an eligibility guard is never support.

### Release envelopes

| Release | Graph envelope | Required non-graph behavior |
| --- | --- | --- |
| R1: safe exact Q1 | Two-group unbundled LMCache, `SHRINK_LATENT=2`, exact configured Q=1 decode sizes, fp16/bf16, target model, TP, one DP replica, one virtual engine | All unsupported steps are classified before model forward as safe native, recompute, or fatal |
| R2: general Q1 | Fixed-capacity padded Q=1 buckets through `max_num_seqs` | Inactive rows are invisible to LMCache and cannot touch live cache state |
| R3: MTP | Fixed `SPEC_FIXED` candidate-width buckets for the target model | Unsupported widths/acceptance layouts use a proven route selected before mutation |
| R4: serving parallelism | DP padding, empty ranks, PP/virtual-engine isolation, multi-engine lifecycle expansion | Rank decisions cannot diverge or reuse stale graph/cache addresses; base cache-epoch invalidation/rejection already ships in R1 |
| R5: optional modes | Individually qualified LoRA, CP/o-proj TP, C8, MLAPO, prefetch, mixed prefill/decode, legacy offload paths | Each mode remains explicitly eager or rejected until its own design and gates pass |

R1 can be released as a bounded production feature. R2-R5 expand coverage; they
must not weaken the R1 safety contract.

## Current implementation: useful foundation, not yet a release boundary

The current worktree already implements a substantial exact-Q1 proof:

- a structural `StagedSFAGraphKey` and per-key layer capture state;
- two `ACLGraphWrapper` instances per SFA layer, with SFA-owned persistent
  buffers providing the required strong output lifetime;
- persistent Graph-A outputs, input signatures, capture records, canaries, and
  ordered full-model startup replay proof;
- exact configured unpadded batch sizes, long-generation sequence changes,
  true multi-request dummy cache rows, and TP-wide startup/parity verdicts;
- Graph-A current-token KV/index writes, indexer selection, scratch remap, and
  Graph-B sparse attention/projections;
- eager selective LMCache retrieval between the graphs with row-aligned
  selected tokens and exact-Q1 request rows; native helpers/connector code have
  groundwork for duplicate MTP IDs and row-specific target slots, but current
  staged eligibility rejects them;
- eager-versus-graph checks for the first live sequence-length tuple per key;
- a conservative estimate for known persistent tensors and graph count.

Those pieces should be retained. The production change is to move authority
out of the layer and formalize the contracts around them.

### Current support and rejection matrix

| Mode | Current behavior | Production requirement |
| --- | --- | --- |
| Exact unpadded Q=1 | Staged for configured sizes | Finish P0 safety work and NPU qualification |
| Long generation | Reuses a shape key with dynamic sequence contents | Prove every block/window/maximum-length boundary |
| Unconfigured or padded Q=1 | Layer-local native fallback | Plan once at runner; add padded buckets in R2 |
| MTP/speculative target | Startup/runtime rejection | Fixed candidate-width keys, row masks, disjoint scratch in R3 |
| Mixed prefill/decode | Rejected | Decode-row compaction plus native prefill design in R5 |
| Compact-scratch LMCache native path | Available for many rows | Classify precisely when it is safe; never assume it is always a fallback |
| Legacy manager/free-paged/adapter | Rejected by staged path | Stay eager until they implement the same transaction/event contract |
| TP | Startup proof and sampled live parity | Per-step TP admission consensus and TP1/2/8 gates |
| DP | Staged configuration rejected | Common structural bucket, rank-local masks, empty-rank handling in R4 |
| PP/multiple virtual engines | No capture namespace/lifecycle contract | Isolate cache epochs and in-flight slots or reject in R1-R3 |
| LoRA, CP/o-proj TP, C8, MLAPO, prefetch | Rejected in layer eligibility | Centralize the rejection before capture and enable individually in R5 |
| Ubatch/cascade | Runner rejects | Preserve rejection until state is invocation-safe |
| Sleep/wake, reload, cache recreation | No invalidation contract | R1 must invalidate/rebuild supported operations or reject them before state changes; R4 extends this to PP/multiple virtual engines |

## Blocking gap ledger

The first ten items are release blockers even for exact Q1. Feature breadth
work begins only after they are closed.

| ID | Current evidence | Risk | Required outcome |
| --- | --- | --- | --- |
| P0.1 atomic dispatch | Runner proposes a key in `_staged_sfa_live_graph_key`, but every layer re-runs `_staged_sfa_graph_ineligible_reason`; Graph A can mutate before a later layer rejects | One model step can mix graph/native execution or fail after partial KV mutation | One immutable all-layer plan, final TP admission, and TP phase verdicts around rank-local connector gaps; no layer-local fallback |
| P0.2 fallback safety | `sfa_v1.py` can warn and enter native after graph rejection; `model_runner_v1.py` already warns that `prompt_len < index_topk` can alias live positions under `SHRINK_LATENT=2` | A path labelled "fallback" can knowingly return wrong output | Classify `SAFE_NATIVE`, `RECOMPUTE`, or `FATAL`; admission must prove the selected route has all required latent data |
| P0.3 pre-mutation validation | The index wait occurs before full replay validation; Graph B's complete signature is checked only after Graph A writes cache | Pointer drift or connector failure can be detected too late | Validate both graph entries, all layers, frontiers, buffers, and rank agreement before the first wait/write |
| P0.4 store-before-free | The stage-2 latent manager frees middle prompt blocks after prompt computation, but the scheduler has no per-request/group LMCache store-commit acknowledgement; sparse frontier can be inferred from prompt length | The only resident copy can be freed although some chunks were never committed | Free only coverage acknowledged by a versioned store commit; derive retrieval frontier from committed coverage, never prompt length |
| P0.5 retrieval readiness | Strict misses and incomplete transfer masks can surface only when a layer generator resumes; exact top-k rows do not exist until Graph A | A post-selection load can fail after index/cache side effects | Before admission, lease the complete selectable committed source domain and validate every possible destination; a short post-A subset load is a protocol failure handled by coordinated abort |
| P0.6 connector transaction | Frontier discovery uses private `_get_connector_metadata`; progress is an implicit `current_layer` cursor | Exceptions, retry, cancellation, or version skew can double-advance or strand the connector | Public versioned begin/layer/commit/abort protocol with step and layer IDs and idempotency |
| P0.7 stream contract | Wrapper replay synchronization is disabled; correctness relies on connector internals and an optional one-time global NPU synchronize | Race between Graph A, load stream, and Graph B; early lease/slot release; hidden first-request latency | Explicit producer/load/Graph-B-completion event contract; zero hot-path device/global synchronization |
| P0.8 stable ownership | Signatures cover explicit tensor arguments, but graph namespace does not cover weights, workspaces, cache epoch, virtual engine, or overlapping invocation | Stale pointers after lifecycle changes or state races | Capture registry and buffer arena scoped by model/VE/cache epoch/key/in-flight slot, with invalidation |
| P0.9 bounded resources | Memory uses known-tensor estimates plus a fixed graph floor; stream count is empirical and PP is not included | KV sizing can overcommit HBM/streams despite passing startup arithmetic | Measured graph/workspace high-water plus a conservative, topology-aware quota before service readiness |
| P0.10 qualification evidence | Startup canaries prove graph tails ran; live parity samples one length tuple per graph key; smoke validates per-key parity and `2 * layers * keys` startup graph cardinality | One live tuple per key does not prove the full semantic boundary matrix | Automate NPU numerical, trace, lifecycle, and failure matrices |
| P1.1 padded rows | Only remap boundary is persistent; builder allocates per-step CPU/NumPy/device metadata | Exact keys cause graph explosion/fallbacks and cannot safely pad | Stable fixed-capacity row arena, safe pad block/slots, masks through both graphs and connector filtering |
| P1.2 rich ACL dispatch | `StagedSFAGraphKey` collapses to legacy `BatchDescriptor(num_tokens)` | Padded Q1 and `SPEC_FIXED` entries can collide at equal token counts | Carry the full structural key through `ACLGraphWrapper` dispatch |
| P1.3 MTP scratch | Native metadata has row-specific groundwork; staged eligibility rejects it | Candidate rows can alias scratch or lose request-row order | Fixed-width profile, unique-request frontier expansion, disjoint scratch/targets, valid-row mask |
| P1.4 scheduler ownership | Input rows can be condensed/swapped after scheduler output is formed; scratch is configured through scattered environment reads | Request IDs, block rows, selected rows, and targets can describe different generations | Build plan after row condensation; use generation/step identity and typed KV scratch configuration |
| P1.5 DP/PP/concurrency | DP/ubatch are rejected and active-key aliases are mutable without locking | Rank deadlock, cross-request state reuse, stale virtual-engine cache addresses | DP-wide bucket agreement, per-VE/cache namespace, and either isolated in-flight slots or enforced no overlap |
| P1.6 compatibility | Layer eligibility, startup config, memory budgeting, and connector checks encode different support subsets | Service can reserve/capture before discovering an unsupported operator combination | One capability fingerprint and reason enum used by every stage |

## Target ownership model

### 1. Capture namespace and structural key

Keep runtime values out of the graph key. Separate capture identity into two
levels:

```text
CaptureNamespace(
    model_instance_id, target_or_draft, device_rank,
    virtual_engine, kv_cache_epoch, layer_id,
    operator_fingerprint, connector_protocol_major,
    in_flight_slot,
)

StagedSFAGraphKey(
    token_capacity, request_capacity,
    query_profile, max_query_len,
    row_layout_profile,
)
```

The operator fingerprint records static facts that are currently implicit:
model/SFA implementation, torch-npu and CANN support tier, dtypes, KV layout,
block size, top-k, head dimensions, q-LoRA path, quantization/operator variant,
TP/CP sharding, projection mode, and relevant connector capabilities. Most of
these do not need to enlarge every dictionary key if the registry is already
scoped to a layer, but they must be recorded and checked for invalidation.

Dynamic contents include request IDs, row order, actual row count, sequence and
prompt lengths, cache frontiers, block IDs, slot mappings, active masks, and
selected tokens. A change in any captured tensor address is not a routine
runtime fallback; the arena must keep it stable or a new cache epoch must
invalidate and recapture the namespace.

### 2. Runner-owned `StagedSFAExecutionPlan`

Build one immutable plan after input-batch update/condensation and before model
forward. It should contain at least:

- step/generation ID and cache epoch;
- chosen execution mode: `STAGED`, `SAFE_NATIVE`, `RECOMPUTE`, or `FATAL`;
- capture namespace, structural key, actual token/request counts, and profile;
- the single authoritative full-request and token-row mapping;
- stable row-buffer arena and active-row slices;
- every participating SFA layer and its prevalidated graph entries;
- connector transaction handle, immutable committed-frontier snapshot, and
  source/destination coverage lease;
- scratch reservation and target-slot ownership;
- TP consensus result and a typed fallback/rejection reason;
- phase, mutation bit, connector-progress ledger, and finalization state.

All SFA implementations consume this plan. They may assert their layer entry,
but they may not independently choose graph versus native. If a model has one
unsupported SFA layer, the complete step is routed before any layer runs.

### 3. Persistent `StagedSFABufferArena`

Allocate per namespace/key/in-flight slot, not ad hoc inside metadata build:

- hidden/input staging only where the enclosing runner cannot guarantee a
  stable address;
- output storage or a documented caller-owned stable output address;
- active/decode masks and actual-count scalars;
- original request index per token row (`-1` for padding);
- row offsets, query/key lengths, prompt lengths, and frontier boundaries;
- latent/index block tables, slot mappings, and safe padded rows;
- scratch base, selected-token output, valid counts/masks, compact request IDs,
  and physical target slots;
- strong Graph-A outputs and graph input tuples;
- reusable events for Graph-A producer and load completion;
- optional debug canaries/parity buffers kept outside the release arena.

The arena must support explicit `allocate`, `bind`, `quiesce`, `invalidate`, and
`release`. A slot cannot be reused until Graph B and connector save/load work
have completed. If only one invocation may be active, enforce that invariant
with a runtime guard rather than relying on today's scheduler behavior.

### 4. Capture registry

The registry owns both wrappers, the graph-pool entry, startup proof, resource
measurements, and state for every namespace/key. Required states are:

```text
UNALLOCATED -> ALLOCATED -> CAPTURING_A -> CAPTURING_B
            -> PROVING_ORDERED_REPLAY -> READY
READY -> QUARANTINED | INVALIDATING -> RELEASED
```

Capture every enabled key before the worker reports ready. Retry is allowed only
after the failed namespace has been fully invalidated and all graph/pool/arena
objects have been rebuilt. Sleep/wake, model/weight reload, KV cache
reinitialization, virtual-engine recreation, operator workspace relocation, or
connector protocol-major change creates a new cache epoch and invalidates old
entries.

## Step and failure state machine

The step transaction must make the mutation boundary explicit:

```text
CREATED
  -> STATIC_PREFLIGHTED
  -> CONNECTOR_PREPARED           # reversible source pin/handle acquisition
  -> SOURCE_DOMAIN_LEASED         # all rows top-k is allowed to select
  -> FINAL_ADMITTED               # TP verdict after connector preparation
  -> LAYER_i_INDEX_WAIT_STARTED   # first connector/cache side effect
  -> LAYER_i_INDEX_READY
  -> LAYER_i_INDEX_PHASE_AGREED
  -> LAYER_i_GRAPH_A_DONE
  -> LAYER_i_LATENT_LOAD_STARTED
  -> LAYER_i_LATENT_READY
  -> LAYER_i_LATENT_PHASE_AGREED
  -> LAYER_i_GRAPH_B_ENQUEUED
  -> LAYER_i_GRAPH_B_DONE_EVENT
  -> LAYER_i_FINISHED
  -> ... next layer ...
  -> MODEL_DONE
  -> SAVES_DONE
  -> COMMITTED

Before INDEX_WAIT_STARTED, a route change may abort the reversible prepared
transaction. At/after INDEX_WAIT_STARTED, no implicit native fallback is
allowed; coordinated abort/recovery only.
```

Required rules:

1. Static preflight validates both Graph A and Graph B entries for every layer,
   the arena, scratch/destination capacity, connector capabilities, and static
   rank compatibility.
2. Connector preparation then snapshots committed frontiers and leases the
   complete source interval from which top-k may select. Final TP admission runs
   after that preparation; disagreement aborts every prepared handle before the
   index wait.
3. The index wait does not begin until final admission succeeds. Its cache write
   and connector progress are the first side-effect boundary.
4. A rank-local index/latent operation must expose a connector-guaranteed
   rank-consistent result or be followed by a TP phase verdict before any rank
   enters Graph A/Graph B collective work.
5. A layer transition is identified by `(step_id, layer_id, group)` and is
   idempotent. Duplicate calls return the previous result or a typed error; they
   never silently increment a cursor.
6. Cancellation/preemption before the side-effect boundary may switch to
   scheduler recovery after aborting preparation. After the boundary it invokes
   connector abort, marks the request/cache epoch as
   needing recovery, and cannot reuse partially written state as if committed.
7. A connector timeout is not automatically a native fallback. It is
   `SAFE_NATIVE` only if admission proved the necessary latent remains resident;
   otherwise schedule recomputation or fail the request.
8. Graph B records a completion event. `finish_layer`, lease release, arena-slot
   reuse, and save release are event-deferred; enqueueing Graph B is not
   completion.
9. Exceptions at every phase run exactly one finalizer. Saves are released only
   after the graph result is accepted. No empty/no-forward/recalc-last path may
   leak or double-finish the connector transaction.
10. A bad key is quarantined for future steps after a replay error. The current
   mutated step follows abort/recovery; future steps may use safe native mode.

## Public LMCache staged-SFA protocol

Private metadata access and monkey-patched capability attributes are not a
production API. LMCache-NPU and LMCache-Ascend need one versioned protocol. The
names below are illustrative; the semantics are normative:

```python
caps = connector.staged_sfa_capabilities(protocol_major=1)
txn = connector.prepare_staged_sparse_step(
    step_id, unique_requests, required_coverage, role
)
snapshot, coverage_lease = txn.get_committed_frontiers(unique_request_order)

txn.wait_index_layer(layer_id, target_cache, consumer_stream)
txn.load_latent_layer(
    layer_id,
    selected_tokens,
    compact_request_ids,
    target_slot_mapping,
    valid_row_mask,
    producer_event,
    consumer_stream,
)
graph_b_done_event = replay_graph_b_and_record_event()
txn.finish_layer(layer_id, graph_b_done_event)
txn.commit_step()
# or txn.abort_step(reason, recovery_policy)
```

The producer side also needs an explicit path; load-step `commit_step()` is not
a persistence acknowledgement:

```python
store = connector.begin_staged_sparse_store(prefill_step_id, requests, groups)
store.enqueue_layers(layer_payloads, producer_events)
local_commit = store.await_store_commit()
global_commit = tp_broadcast_and_intersect_required_coverage(local_commit)
worker.report_store_commit_to_scheduler(global_commit)
# scheduler releases only aligned whole block bundles in global_commit
```

Capabilities must describe, rather than imply:

- protocol major/minor and compatible vLLM metadata version;
- producer/consumer/both roles and layerwise callback ownership;
- two-group index/latent behavior and which group advances progress;
- selective row load, duplicate MTP request rows, direct physical target slots,
  padded-row masks, and maximum row/token capacities;
- maximum in-flight transactions, stream-pool ownership, and whether overlapping
  forward/load/store calls are serialized or forward-scoped;
- frontier meaning, decode-window interaction, partial/miss outcomes, and
  snapshot/coverage-lease lifetime;
- store-commit acknowledgements by request, group, layer range, token interval,
  and cache generation;
- Graph-A producer-event ownership, load-stream/compute-stream hand-off, and
  Graph-B completion-event-deferred finish/release;
- idempotency, retry, abort, timeout, and cancellation semantics;
- tensor lifetime: when selected/ID/target buffers may be reused, with complete
  destination `data_ptr`/shape/stride/storage-offset and source lease identity;
- TP save/load ownership, `save_only_first_rank`, shared CPU cache, and rank
  failure behavior;
- async store/load backpressure and completion semantics.

Frontier lookup always uses the unique full-request order. Preparation leases
the whole committed interval that the qualified top-k operator is allowed to
select, not the still-unknown exact top-k subset. The snapshot is then
expanded through the plan's token-row map. The final selective payload preserves
candidate-row order and may contain duplicate request IDs. For every valid row,
`selected_tokens[i]`, `request_ids[i]`, `target_slot_mapping[i]`, and the valid
mask describe the same row.

After Graph A, loading the exact selected subset must be guaranteed by that
lease. A short transfer mask or source disappearance is a protocol/health
failure after the side-effect boundary and triggers TP-coordinated abort and
request/cache recovery; it is not a native fallback.

The connector returns typed outcomes such as `READY`, `PARTIAL`, `MISS`,
`TIMEOUT`, `CANCELLED`, and `PROTOCOL_ERROR`. The runner decides recomputation or
fatal handling before mutation where possible. Broad exception-to-`False`
capability checks and silent no-ops are forbidden in staged mode.

### Store-before-free and retrieval leases

Stage-2 memory saving is correct only if persistence, scheduler release, and
later retrieval share one committed-coverage contract:

```text
prefill KV produced
  -> LMCache stores required latent/index groups and waits for durable/usable ack
  -> StoreCommit(request, cache_generation, layers, groups, committed_intervals)
  -> worker intersects required layer/group coverage and propagates the TP
     result (including rank-0 broadcast for save_only_first_rank)
  -> scheduler marks only those intervals releasable
  -> aligned whole latent block bundles are freed
  -> decode acquires a lease over the same generation/coverage
  -> staged/native compact-scratch execution consumes that lease
  -> lease releases after Graph B and connector completion
```

Prompt length or chunk rounding is not evidence of persistence. Missing
metadata, cache mapping, indexer registration, a short transfer mask, or an
empty source is a typed incomplete-coverage result, never a silent save/load
success. A failed store keeps the resident blocks, triggers recomputation before
free, or fails the request according to policy. It cannot advance the committed
frontier.

The decode preparation call must validate and lease every source interval that
the captured top-k is allowed to select before final staged admission. The token binds
request generation, committed intervals, source object/pin generation, and
destination cache epoch. Forced unpin, cache health change, connector restart,
or pointer re-registration invalidates the lease and prevents Graph A. A lease
cannot expire while its asynchronous load or Graph B is still using it; the
Graph-B completion event, not enqueue return, releases it.

If LMCache cannot provide an efficient guarantee for the full selectable
domain, the alternative is to split selection into a read-only graph before the
mutation boundary, validate/lease its exact result, and capture cache writes in
a later graph. That is a different three-island design and must be benchmarked;
R1 may not pretend exact selected rows were known before current Graph A.

## Graph A and Graph B contracts

### Graph A

Inputs are fixed-capacity arena tensors plus stable KV-cache tensors. It may:

- preprocess Q/K and apply rotary/norm operations;
- write the current token's latent and index key;
- execute the qualified lightning-indexer/top-k variant;
- mask inactive candidates before remap;
- remap selected prompt rows into disjoint compact scratch positions;
- emit fixed-capacity selected tokens and an explicit valid mask/count.

It must not call Python metadata lookup, allocate tensors, synchronize the
device, access connector internals, or branch on a host runtime value. Padding
must be present during capture. The selected payload's unused tail may not be
represented only by token zero unless the valid mask contract guarantees that
the connector ignores it.

### Graph B

Inputs are Graph-A strong outputs, stable block/query metadata, the latent cache
after the connector event, and caller/arena output storage. It performs sparse
attention and qualified projections. Inactive rows remain masked through the
attention and output is ignored/zeroed by contract. Graph B returns exactly the
owned output tensor and does not allocate a semantically different result.

### Replay validation

Full pointer/shape/stride/storage/dtype/device checks are valuable during
capture and sampled debug operation, but are too expensive and too late as the
primary hot-path safety mechanism. Release mode relies on registry/arena epochs
and validates the entire plan before mutation. Sampled signature/canary checks
quarantine a key on drift. Startup proof must include numerical reference data,
not only terminal canaries.

## Padded Q1 design

Use bounded capacity buckets selected by the resource planner. A graph captured
for capacity `C` serves `1..C` real requests with the same fixed addresses.
Power-of-two buckets are a reasonable default, but the actual list must be
chosen from workload distribution and resource budget, not hard-coded.

For every inactive row:

- `active=false`, original request index is `-1`, and lengths/counts are safe;
- latent/index block tables point at a permanently reserved safe block;
- slot mappings and scratch targets point at reserved safe slots;
- Graph A masks the row before top-k/remap/current-token writes;
- the connector never sees the row;
- Graph B cannot read live request data for it and its output is ignored;
- safe blocks are never saved, freed, or assigned to a request.

The scheduler owns actual counts; graph shapes remain capacity-sized. Transition
tests must cover every real size in a bucket, request reorder/condense, and
`1 -> C -> 1` without address or row-map drift.

## MTP / fixed multi-token design

The existing native metadata and scratch-remap code provide useful groundwork,
but staged MTP requires a separate `SPEC_FIXED` structural profile.

- Key candidate width and maximum query length; mask the actual accepted/query
  rows within that capacity.
- Look up a frontier once per unique request, then expand it to candidate rows
  using the immutable plan map.
- Preserve duplicate compact request IDs in candidate order.
- Reserve row `j` scratch interval disjoint from every other row, at minimum
  `[scratch_base(j), scratch_base(j) + index_topk)`.
- Validate committed frontier and physical capacity against every row-specific
  base before Graph A.
- Generate and validate target slots for all candidates; never reuse a live
  token slot or an inactive row's safe slot.
- Keep target and draft capture registries/plans separate. A target-model MTP
  release does not imply an SFA draft model is supported.
- Define how partial acceptance, recalc-last, and rejected candidates affect KV
  commit and connector progress.

If `frontier < scratch_base + index_topk`, admission must select an alternative
reserved layout, retain a correct resident route, schedule recomputation, or
fail. Silent scratch/live aliasing is forbidden.

## Parallelism, scheduler, and lifecycle route

### Tensor parallelism

- Perform one support/admission consensus per step within each TP group before
  model forward; never discover rank disagreement in a layer.
- After each rank-local connector index/latent operation, require either a
  protocol-guaranteed uniform result or a TP phase verdict before entering the
  following graph island. One-rank failure must make every rank abort before a
  projection/collective can diverge.
- Gather rank-specific diagnostics when consensus fails.
- Preserve identical layer/phase progress across ranks even when only one rank
  saves to shared LMCache.
- Qualify TP1, TP2, and TP8, including rank-local pointer/layout differences.

### Data parallelism

- Choose a compatible structural bucket across DP replicas before any
  collective; active masks, request IDs, and frontiers remain rank-local.
- Support an all-padding empty rank and uneven loads without connector calls for
  inactive rows.
- Do TP consensus inside each replica and define any DP-wide graph-key
  agreement separately.
- Exercise DP2/DP4 alone and combined with TP; no rank may enter a different
  collective schedule because its graph admission differed.

### Pipeline parallelism and virtual engines

- Namespace graphs and arenas by virtual engine and local PP stage/cache epoch.
- Count only local SFA layers for memory, while validating the global pipeline
  support contract.
- Allocate an in-flight slot per overlapping microbatch or enforce serialized
  replay until such slots exist.
- Pass step/sequence identity across stage boundaries so a condensed request row
  cannot reuse stale metadata.
- Reject PP/multiple virtual engines until invalidation, cache ownership, and
  overlap tests pass.

### Scheduler and KV ownership

- Build the row plan after `_update_states`, request removal/swap/condensation,
  and final scheduled-token counts.
- Use the same mapping for block tables, full request IDs, compact IDs,
  selections, target slots, and output rows.
- Promote DSA scratch/window sizing from scattered raw environment reads into a
  typed KV-cache configuration owned by the scheduler/KV manager.
- Reserve scratch as first-class state and keep it alive until load/save events
  complete. Finish/cancel/preempt/recompute release must be idempotent.
- Capacity shortage is a startup/admission failure, not only a log message.

### Lifecycle invalidation

R1 must implement cache-epoch invalidation for any lifecycle operation it
allows. An operation not implemented in R1 (for example sleep/wake on a given
runtime) is rejected before memory or connector state changes. R4 adds
per-PP-stage, multi-virtual-engine, and overlapping-microbatch coverage; it does
not defer the base R1 safety rule.

Create a new cache epoch and invalidate affected namespaces on:

- sleep/wake or NPU memory-pool recreation;
- KV-cache allocation/reinitialization;
- model or adapter weight reload/hot swap;
- virtual-engine/PP topology recreation;
- operator implementation/workspace relocation;
- connector restart or protocol-major change;
- graph-pool reset or compile configuration change.

Quiesce all replays and connector work before release. Do not leave
`_dsa_idx_cache_t`, strong outputs, proof records, or parity buffers bound to the
old epoch.

## Compatibility policy

One central capability evaluation must drive configuration validation, capture
bucket selection, stream/HBM budget, runner admission, and layer assertions.
Use stable reason codes instead of independent strings.

Initial R1 allowlist:

- MLA/SFA model with the qualified q-LoRA fused preprocessing and q norm path;
- PA_BSND fp16/bf16, one KV head, qualified dimensions/block size;
- `PIECEWISE` graph mode;
- unbundled two-group cache with `SHRINK_LATENT=2`;
- versioned layerwise selective LMCache connector in consumer/both role;
- DecodeOnly Q1, configured exact sizes, DP=1, no ubatch/cascade;
- qualified TP topologies and operator/CANN/torch-npu versions.

Validate shrink mode as an enum across vLLM, vLLM Ascend, and LMCache: stage 0
is resident/native, stage 1 is compact-scratch validation without freeing,
stage 2 is store-commit-gated freeing and the only staged-graph release target,
and stage 3 is diagnostic-only. Reject every other integer at startup. Passing
stage-1 parity does not prove stage-2 store/free safety.

Remain explicit native/reject until separately designed:

- LoRA/adapter switching;
- DSA context parallelism and o-proj TP variants;
- sparse-C8 indexer and other quantization layouts;
- MLAPO and weight prefetch;
- free-paged manager, legacy `dsa_offload_manager`, and adapter cache;
- mixed prefill/decode and prefix-caching combinations;
- unsupported target/draft combinations.

Use four operational policies rather than one ambiguous boolean:

- `off`: never build/use staged graphs;
- `verify`: staged execution plus a side-effect-free eager reference, with saves
  deferred until parity succeeds; the graph is not silently discarded, and a
  mismatch aborts/recomputes the request because Graph A touched live cache;
- `auto`: staged only for ready plans; safe native/recompute elsewhere;
- `strict`: validation mode; any requested-envelope miss fails immediately.

Keep the existing environment variable as a compatibility shim, centralize the
one-time sync debug variable in `envs.py`, and migrate production configuration
to typed fields. Release logs must stop calling the path a POC after all R1
gates pass, but must continue reporting its exact support tier.

## Resource model

The budget must be computed per device before KV-cache sizing:

```text
R_total = sum(namespaces, keys, local_SFA_layers, in_flight_slots)(
              graph_A_pool + graph_B_pool
            + graph_A_workspace + graph_B_workspace
            + persistent_inputs + strong_A_outputs + owned_output
            + safe_padding_storage + events)
          + connector_staging_and_scratch
          + optional_verify_or_parity_storage
          + configured_safety_margin
```

Current capture requires real KV tensors, so a high-water measurement from that
capture cannot retroactively size the same one-pass KV allocation. Production
must choose and document one non-circular strategy:

- use a conservative offline qualification bound keyed by the complete
  hardware/software/operator fingerprint, reserve it before KV sizing, then
  verify the live capture stays below it; or
- use a two-pass startup with provisional reduced KV, measure capture, fully
  tear down graph/cache state, compute the final KV budget, and recapture.

Exceeding the bound fails readiness. It cannot silently shrink the already-live
KV cache or keep stale first-pass graph entries.

Required work:

1. Keep exact static bounds for owned tensors, using local PP layer counts and
   target/draft registries separately.
2. Measure capture and replay high-water for graph pool, driver metadata, and
   operator workspaces on every qualified software/hardware tier; publish the
   offline bound or implement the two-pass allocation protocol.
3. Replace empirical stream arithmetic with the actual number of graph entries,
   communication streams, connector streams, and in-flight slots.
4. Feed the conservative result into KV sizing; fail or reduce buckets before
   service readiness if the budget is negative.
5. Record estimated, reserved, measured-capture, and measured-steady-state bytes
   and stream/event counts per key.
6. Release startup/parity-only buffers after a rank-consistent proof unless
   verify sampling is configured.
7. Bound graph count and provide explicit entry invalidation/destruction; no
   unbounded lazy recapture on live shapes.

## Observability and operational controls

Expose low-cardinality metrics with reason enums:

- configuration support tier and incompatible feature reason;
- capture attempts/success/failure/duration by structural key;
- registry state, ready/quarantined keys, cache epoch, and recapture count;
- step admission counts for staged/safe-native/recompute/fatal;
- per-layer Graph-A replay, index wait, latent load, Graph-B replay, and save
  timing;
- connector transaction starts/layer transitions/commits/aborts/timeouts and
  duplicate-transition detection;
- active/padded/selected rows and cache hit/partial/miss outcome;
- reserved/measured graph, arena, scratch, and connector memory;
- sampled parity/top-k/cache-write mismatch counts;
- TP/DP decision disagreement and lifecycle invalidation reason.

Profiler traces must show the expected index wait, one Graph A replay, one
selective load, and one Graph B replay per participating layer, with explicit
events and no hot-path global synchronize. A health endpoint/log summary should
state enabled keys, protocol version, qualification fingerprint, resource
budget, and fallback/recompute policy.

Use a runtime kill switch that prevents new staged plans. It may route future
steps to a proven safe path; it cannot retroactively fall back a mutated step.

## Code impact map

| Area | Required change |
| --- | --- |
| `vllm_ascend/worker/model_runner_v1.py` | Build/finalize the immutable execution plan; enumerate all local SFA layers; do TP consensus; own arenas/registry; handle lifecycle and parity sampling |
| `vllm_ascend/ascend_forward_context.py` | Pass a plan handle and layer cursor instead of loose dummy/key/parity attributes; carry cache epoch/virtual-engine identity |
| `vllm_ascend/attention/sfa_v1.py` | Reduce to graph kernels plus a plan-driven layer executor; remove layer-local mode choice/private connector discovery; add padded/MTP masks; remove POC sync and hot signature dependence |
| `vllm_ascend/attention/utils.py` | Replace private metadata/capability probes with the public connector transaction adapter and typed outcomes |
| `vllm_ascend/distributed/kv_transfer/sparse_offload/scratch_remap.py` | Add explicit valid mask/count, safe padded rows, disjoint MTP ranges, and device-side bounds assertions where available |
| `vllm_ascend/worker/worker.py` | Topology-aware measured memory reservation, local PP layer count, target/draft and lifecycle accounting |
| `vllm_ascend/utils.py` | One compatibility fingerprint/reason system; rich capture keys; resource-driven bucket selection; actual stream accounting |
| `vllm_ascend/envs.py` | Typed operational mode/buckets/debug sampling; centralize all staged-SFA environment reads |
| `vllm_ascend/compilation/acl_graph.py` | Accept full structural dispatch key, expose bounded invalidation/destruction and resource measurements |
| vLLM scheduler/input batch/KV managers | Final row-generation plan, typed scratch/window config, capacity enforcement, store-commit-gated latent release, idempotent lifecycle/recompute ownership |
| LMCache-NPU vLLM adapter | Versioned step/layer transaction, public committed-frontier/readiness lease, idempotent progress, abort/finalize, event and tensor-lifetime contract |
| LMCache-Ascend connector | Advertise the full negotiated capability object without monkeypatches; implement two-group/rank-specific semantics and failure outcomes |
| smoke/unit/NPU test tools | Multi-key graph-count parsing, plan/failure injection, lifecycle/parallel matrices, automated trace assertions |

## Delivery route and dependencies

### W0: freeze the support contract and repair evidence tooling

- Convert the compatibility table into typed capability/reason data shared by
  startup, resource planning, runner, and layer assertions.
- Document `SAFE_NATIVE` versus `RECOMPUTE` versus `FATAL` for every current
  rejection, especially `prompt_len < index_topk` and missing/partial cache.
- Update `tools/staged_sfa_graph_smoke.py` for the current startup message and
  `graph_count == 2 * local_sfa_layers * keys`.
- Repair the reported profiler-smoke contract: when AsyncLLM frontend profiling
  is enabled, launch with profiler config `ignore_frontend=true` so
  `/stop_profile` finalizes only TP-worker traces. Generalize the client's
  intentional TP8-only parser/trace assertions; for the supplied TP2 launch,
  pass or derive `--expected-ranks 2` instead of its current enforced value 8.
- Add metrics/log schema and qualification fingerprint.

Exit: every rejection has a tested action and the smoke gate matches multi-key
capture. This work can start in parallel with W1.

### W1: public LMCache transaction and typed scratch configuration

- Land the capability/snapshot/begin/layer/commit/abort protocol across
  LMCache-NPU and LMCache-Ascend.
- Add per-request/group/generation store-commit acknowledgement; derive sparse
  frontier from committed coverage and gate scheduler latent release on it.
- Aggregate the intersection across every required layer/group and TP owner,
  broadcast `save_only_first_rank` results, and free only aligned whole block
  bundles acknowledged to the scheduler.
- Lease the full selectable committed source domain in step preparation and
  validate every possible destination. Guarantee any later top-k subset; treat
  a short post-A transfer mask as a coordinated protocol failure.
- Make progress idempotent by step/layer/group ID; remove private metadata reads
  and monkey-patched capabilities from staged admission.
- Add explicit Graph-A producer event and load-completion contract.
- Move scratch/window capacity into typed KV configuration and enforce it.

Exit: mocked and NPU fault-injection tests prove store-before-free, selectable-
domain lease guarantees, Graph-B-event-deferred release, and exactly-once
progress for success, timeout, exception, cancel, preempt, and retry. Blocks W3
release.

### W2: capture registry, prebound arena, invalidation, and resource budget

- Add namespace/key/in-flight ownership and full structural ACL dispatch.
- Allocate stable per-key tensors, including Graph-A outputs and prebound
  Graph-B inputs, so both graph entries can be validated before index wait.
- Implement cache epochs, quiesce/invalidate/rebuild, quarantine, and bounded
  graph destruction. Reject unsupported lifecycle operations before state
  changes.
- Implement the offline-bound or two-pass resource strategy and feed the result
  into KV sizing without circular measurement.
- Remove release-mode hot signature scans after arena/epoch proof.

Exit: both graph bindings are known before side effects; exact Q1 survives
`1 -> B -> 1` and every allowed/rejected R1 lifecycle transition with bounded
memory and no stale-address replay. Can proceed in parallel with W1, but event
and connector resource accounting finish with W1.

### W3: runner-owned transaction for existing exact Q1 kernels

- Introduce the execution plan and the static-preflight -> connector-prepare ->
  source-lease -> final-admission state machine.
- Validate every local layer and both graph entries before index wait; do final
  per-step TP admission and phase verdicts after rank-local connector gaps.
- Forbid layer-local fallback after planning and any fallback after index wait.
- Implement one RAII finalizer plus Graph-B completion-event hand-off for
  success/error/cancel/no-forward paths.
- Keep current exact-Q1 math kernels unchanged except for consuming the plan.
- Remove the first-use global synchronization after event proof passes.

Exit: injected local/rank-asymmetric failure at every layer/phase cannot cause
mixed graph/native execution, collective divergence, double cursor movement,
early lease/slot release, or reuse of partial state. Depends on W1 and W2.

### W4: R1 exact-Q1 NPU qualification and rollout

- Run the complete numerical/boundary/cache/TP/lifecycle matrix.
- Automate profiler/operator/event assertions and performance gates.
- Run verify/canary soak, release parity-only storage, and validate kill switch.
- Publish the qualified hardware/software/operator fingerprint.

Exit: all R1 definition-of-done gates pass; feature remains default-off until
the release owner signs the evidence bundle.

### W5: padded Q1 buckets

- Add fixed-capacity persistent row metadata, safe blocks/slots, active masks,
  connector filtering, and Graph-B output masking.
- Choose buckets from resource budget and workload distribution.
- Prove all real sizes and row reorder/condense transitions within each bucket.

Exit: ordinary sizes through `max_num_seqs` use bounded keys and inactive rows
cannot mutate/read/save request data. Depends on W2-W3.

### W6: fixed-width MTP target decode

- Add `SPEC_FIXED` dispatch, candidate masks, unique-to-compact frontier
  expansion, row-specific scratch/targets, and acceptance/recalc semantics.
- Account target and draft resources separately; keep an SFA draft model out of
  scope unless separately qualified.

Exit: widths 1/2/3 and all accepted-token patterns pass logits/token/KV/LMCache
parity under churn and boundary cases. Depends on W5 row-capacity machinery.

### W7: DP, PP/virtual engines, and overlap

- Implement DP bucket agreement and empty-rank execution.
- Namespace/invalidate per PP stage and virtual engine.
- Add in-flight slots or enforce and test serialization.
- Extend memory/stream budget and fault consensus to combined parallelism.

Exit: DP2/DP4, PP2/VE2, and qualified TP combinations sustain heterogeneous
loads without hangs, row leakage, cursor drift, or stale capture. Depends on
W2-W5; MTP+DP qualification additionally depends on W6.

### W8: optional execution modes

Enable one at a time: mixed prefill/decode, LoRA, CP/o-proj TP, sparse-C8,
MLAPO, weight prefetch, free-paged manager, legacy manager, adapter cache, and
additional prefix-cache configurations. Each work item supplies:

- a capability-fingerprint extension;
- graph/data/event/ownership design;
- resource delta;
- safe pre-mutation fallback classification;
- unit, NPU numerical, lifecycle, parallel, and performance evidence.

## Verification program

### Unit and mocked integration

- structural-key collision/isolation across exact, padded, and `SPEC_FIXED`;
- persistent address stability after request remove/swap/condense;
- full/compact request ordering and duplicate MTP IDs;
- safe padding block/slot behavior and valid-tail masking;
- frontier and capacity below `index_topk` and below
  `scratch_base + index_topk`;
- all-layer plan failure and TP mismatch before mutation;
- one-rank-only index/latent failure prevents every rank from entering the next
  graph island;
- connector exactly-once ledger under an exception injected at every state;
- store failure/short save cannot advance frontier or free resident blocks;
- readiness miss/short transfer/lease invalidation is visible before Graph A;
- connector destination pointer rebind/allocator reuse rebuilds native plans;
- source pin timeout cannot invalidate an active coverage lease;
- delayed Graph-B completion prevents early lease, scratch, or arena-slot reuse;
- missing indexer registration and connector import-order variants fail the
  public handshake rather than silently changing capabilities;
- cancel/preempt/recompute/recalc-last/empty-step/late-save finalization;
- cache-epoch invalidation and bounded entry release;
- resource formula for local PP layers, target/draft, every key and slot;
- offline-bound overrun and two-pass teardown/recapture resource paths;
- smoke log/count tests for one/multiple keys, TP1/2/8 trace counts, and the
  worker-only (`ignore_frontend=true`) profiler requirement.

### NPU correctness matrix

Compare staged output with both the resident eager reference and the native
compact-scratch LMCache path where that native route is valid. Check full layer
output, top-k indices, current-token latent/index writes, scratch contents,
logits, and deterministic generated tokens.

Axes include:

- every enabled capacity, every real padded size, and `1 -> B -> 1`;
- heterogeneous sequence/prompt lengths and request reorder/churn;
- block boundaries 127/128/129;
- decode-window boundaries 255/256/257;
- `index_topk - 1`, `index_topk`, and `index_topk + 1`;
- 6143/6144/6145 and configured `max_model_len`;
- all-warm, cold retrieval, partial frontier, true miss/recompute, and timeout;
- serialized mode and, once supported, overlapping load/store with more than
  the connector's current stream-pool width;
- decode-window save and `save_only_first_rank` behavior;
- arrival, finish, cancel, preempt, recompute, recalc-last, and load failure at
  every layer phase;
- TP-rank-asymmetric index/latent timeout and store-commit failure;
- fp16/bf16 and every allowlisted model/operator/CANN/torch-npu fingerprint;
- TP1/2/8, then DP2/4, PP/VE, and combined qualified topologies;
- MTP widths and partial-acceptance patterns when W6 is enabled.

### Capture and trace assertions

- exactly two ready graph entries per local SFA layer and structural key;
- no unbounded recapture after readiness;
- every intended operator is inside the correct replay island;
- index wait precedes Graph A; producer event precedes selective load; load
  completion precedes Graph B;
- no hot-path `.item()`, global stream/device synchronize, or tensor allocation
  in either replay path;
- graph output aliases the documented owned output and all strong outputs live
  for the required slot lifetime.

### Reliability and performance gates

- startup capture high-water and steady-state HBM stay within the advertised
  reservation and leave the configured KV budget intact;
- stream/event counts stay within the runtime quota;
- long-generation and high-churn soak has zero parity, cursor, stale-pointer,
  or cross-request failures;
- throughput and TPOT improve for the workload/buckets that justify the feature;
- TTFT/TPOT p50 and p99 have no unapproved regression from index waits, event
  hand-off, plan building, or first-use synchronization;
- host launch count and eager gap between Graph A/B match the trace budget.

Numeric performance thresholds, model mix, prompt/decode distribution, and soak
duration must be fixed before W4 begins and stored with the evidence. A claim of
"faster in one smoke run" is not a release gate.

## Rollout

1. Land W0-W3 behind `off`/`strict`; run CI and qualification without serving
   user traffic.
2. Use sampled `verify` on non-production qualification traffic. The eager
   reference is side-effect-free, connector saves are deferred, and any parity
   mismatch aborts/recomputes rather than continuing with mutated live cache.
3. Canary `auto` for an allowlisted model/fingerprint and a small key set.
4. Increase key/traffic coverage only when fallback/recompute, abort, memory,
   TPOT, and parity metrics remain within the signed thresholds.
5. Quarantine a bad key automatically; use the kill switch for future steps.
   Mutated in-flight steps still follow transaction recovery.
6. Make R1 generally available only with the evidence bundle and rollback
   procedure; keep R2+ features separately gated.

## Definition of done

An enabled structural key is production-ready only when all statements are
true:

1. One runner plan and final TP admission select execution before any connector
   wait/cache mutation, and phase verdicts prevent rank divergence after each
   rank-local connector operation.
2. Every graph input/output and implicit dependency has stable owned lifetime
   under a versioned namespace and cache epoch.
3. Connector progress and store commitment are public, versioned,
   event-ordered, idempotent, and exactly once on
   success/error/cancel/recompute paths.
4. Unsupported values have a proven `SAFE_NATIVE`, `RECOMPUTE`, or `FATAL`
   action; no unsafe fallback exists after mutation.
5. Startup capture and ordered replay succeed on every participating rank and
   numerical parity covers the signed boundary/topology matrix.
6. Padding/MTP rows, when enabled, have masks and disjoint cache/scratch/target
   ownership through both graphs and LMCache.
7. Graph, workspace, arena, scratch, stream, and connector resources are bounded
   before KV sizing and agree with measured high-water.
8. Supported lifecycle invalidation and explicit pre-change rejection of
   unsupported lifecycle operations, plus graph quarantine, cancellation,
   preemption, recomputation, and rollback behavior, pass fault-injection and
   soak tests.
9. Profiler evidence proves the intended replay/event structure with no
   hot-path global synchronization or silent recapture.
10. The qualified workload shows the agreed sustained performance improvement,
    and metrics/kill switch make regressions operationally visible.

Until R1 satisfies all ten gates, the runtime and documentation must call the
feature experimental and report the exact admission/fallback/recompute reason.
