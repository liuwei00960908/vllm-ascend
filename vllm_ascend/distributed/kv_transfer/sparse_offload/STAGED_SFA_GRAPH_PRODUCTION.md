# Production design and delivery roadmap for staged SFA ACL graphs

## Executive decision

The production target is a **cross-layer piecewise model-step executor** in
which LMCache retrieval is the only staged-SFA FX split. The outer vLLM
piecewise compiler owns the ACL graph islands. `sfa_v1.py` owns graph-safe
pre-retrieval and post-retrieval operator bodies and their explicit tensor
contract, but it does not own nested per-layer graph wrappers in the target
path. The model runner owns admission, island selection, stable buffers, TP
agreement, connector progress, failure handling, and step finalization.

The required ordered schedule is:

```text
Static preflight -> validate current remap/callback assumptions -> final TP admission
                  (no cache/cursor side effect)
                                  |
                                  v
Bootstrap: wait/materialize the LMCache index group for layer 0,
         then obtain a rank-consistent success/failure verdict
                                  |
                                  v
Island 0: Graph A(0): Q/K preprocessing, current-token KV/index writes,
          indexer/top-k, inactive-row mask, scratch remap, selection output
                                  |
                                  v
Split 0: selective latent load(0) and index load(1), using the existing
         connector callbacks, with a rank-consistent verdict
                                  |
                                  v
Island 1: Graph B(0) -> layer tail(0) -> Graph A(1)
          producer/load/consumer events close every cross-stream dependency
                                  |
                                  v
... repeat one retrieve split per participating layer ...
                                  |
                                  v
Island N: Graph B(N-1) -> layer tail(N-1)
          completion ordering prevents early resource and slot reuse
                                  |
                                  v
Existing connector finalization and deferred saves
```

This architecture fits the current compiler model. Ascend already wraps the
whole model in one ACL graph for `FULL`/`FULL_DECODE_ONLY` when there are no
layerwise host callbacks. In PIECEWISE mode, vLLM isolates every configured
split op and merges all ordinary FX nodes between adjacent splits into one
compiled subgraph. Therefore a retrieve-only split naturally yields islands
that cross transformer-layer boundaries.

It is not available through configuration alone. Today
`torch.ops.vllm.mla_forward` is an opaque custom op and `platform.py` forces the
whole op to be a split in PIECEWISE mode. Its Python body obtains attention
metadata/KV cache through `ForwardContext` and invokes both SFA computation and
LMCache callbacks. The target must decompose this staged path into graph-safe
pre/post ops with explicit tensors plus one mutation-aware eager retrieve op.
Non-staged MLA keeps its existing route. A single uninterrupted full-model ACL
graph remains invalid until retrieval itself becomes device-driven and
capture-safe.

Production-ready does not mean every vLLM mode must be captured in the first
release. It means the enabled envelope is correct, bounded, observable, and
recoverable, while every other combination receives a proven pre-mutation
native route, request recomputation route, or explicit rejection. Merely
removing an eligibility guard is never support.

### Release envelopes

| Release | Graph envelope | Required non-graph behavior |
| --- | --- | --- |
| R1: cross-layer exact Q1 | Retrieve-only outer FX splits, nominally `N + 1` ACL islands for `N` local SFA layers, two-group unbundled LMCache, `SHRINK_LATENT=2`, exact configured Q=1 decode sizes, fp16/bf16, target model, TP, one DP replica, one virtual engine | All unsupported steps are classified before model forward as safe native, recompute, or fatal |
| R2: general Q1 | Fixed-capacity padded Q=1 buckets through `max_num_seqs` | Inactive rows are invisible to LMCache and cannot touch live cache state |
| R3: MTP | Fixed `SPEC_FIXED` candidate-width buckets for the target model | Unsupported widths/acceptance layouts use a proven route selected before mutation |
| R4: serving parallelism | DP padding, empty ranks, PP/virtual-engine isolation, multi-engine lifecycle expansion | Rank decisions cannot diverge or reuse stale graph/cache addresses; base cache-epoch invalidation/rejection already ships in R1 |
| R5: optional modes | Individually qualified LoRA, CP/o-proj TP, C8, MLAPO, prefetch, mixed prefill/decode, legacy offload paths | Each mode remains explicitly eager or rejected until its own design and gates pass |

R1 can be released as a bounded production feature. R2-R5 expand coverage; they
must not weaken the R1 safety contract.

## Current implementation checkpoint: cross-layer exact-Q1 milestone

The branch now contains the cross-layer implementation plus exact-size Q1
batching. Singleton execution and TP2 batch sizes 1/2/4 have passed initial
Ascend functional/performance trials; runtime per-key replay and trace evidence
are still required before the batched path is qualified:

- `vllm::sfa_lmcache_retrieve` is the only staged-SFA FX split;
- Graph A and Graph B reuse the already-validated SFA math but are captured by
  the outer PIECEWISE executor, producing nominally `N + 1` islands for `N`
  local SFA layers;
- the runner performs the layer-zero index/bootstrap wait before model forward,
  then each retrieve split loads the current latent group, prepares the next
  layer's remap boundary, and waits for its index group;
- a persistent event recorded inside Graph A is forwarded through LMCache's
  existing optional `payload_event` argument; the adapter waits it on the
  compute stream and LMCache's load stream waits that stream before packing and
  transfer, so selective-load payloads cannot race captured top-k/remap
  production;
- decode saves are submitted at the model boundary because per-layer Python
  save callbacks cannot remain inside the cross-layer islands;
- capture is restricted to configured exact unpadded Q=1 request counts. A
  fixed-capacity contiguous bridge uses the largest configured capture size;
  smaller exact keys populate that bridge and slice back to their authorized
  rows. This is an internal stable-layout ABI, not general padded-Q1 support;
- LMCache row routing preserves the exact batched request order, and startup
  verifies every configured key on every local SFA layer;
- parity verification is once per graph key rather than once per replay;
- a scheduler step containing both prefill and decode currently disables outer
  replay for the whole step and uses native SFA. Only its decode request rows
  may require sparse frontiers; the trial-exposed filtering fix is implemented
  in the worktree and awaits an NPU rerun;
- other unsupported steps run the existing native SFA path with outer replay
  disabled before model forward;
- startup rejects missing SFA layers, producer events, stable save state, or an
  invalid bridge layout.

This is not yet a release boundary. The TP2 trials established correct startup,
deterministic/acceptable output, decode save, dense-prefix-hit TTFT reduction,
once-per-key parity, repeat-run stability, and a large singleton TPOT gain over
the earlier two-wrapper and native baselines. On the eight-layer TP2 test model,
steady throughput for exact batches 1/2/4 is 56/94/144 tok/s with LMCache and
72/124/184 tok/s with `start_load_kv`/`wait_for_layer_load` bypassed. The nearly
identical batch scaling places most sub-linear scaling outside LMCache, while
the stable 22-24% LMCache gap remains an optimization opportunity. Runtime
proof of exact batch-2/4 replay, the precise `N + 1` cross-layer trace, failure
behavior, resource bounds, and the production ownership plan remain open. The
nested-wrapper implementation has been removed from this branch; its earlier
commit remains the comparison and rollback point.

### Trial-closed issues and newly exposed work

`DONE` below means the stated trial issue is closed; it does not imply that the
whole R1 production gate is complete.

| Trial item | Status | Evidence or remaining gate |
| --- | --- | --- |
| Singleton startup, output, and repeated decode | **DONE** | TP2 startup and repeated requests complete without runtime errors; output is deterministic/acceptable |
| Singleton throughput checkpoint | **DONE** | No-profiler TPOT is about 120 ms average, with 108 ms observed peak, versus about 170 ms for the two-wrapper implementation and 326 ms without staged graph capture |
| Dense prefix hit and decode offload/save | **DONE** | Repeated prompt obtains reduced TTFT; decode save/offload operates as expected |
| Parity verification frequency | **DONE** | Verification runs once per graph key and preserves existing output behavior |
| TP worker profiler control and offline trace parsing | **DONE** | Smoke tooling requires `ignore_frontend=true`, accepts the expected TP rank count, and parses Ascend traces outside daemon workers |
| Fake/native custom-op bridge layout mismatch during large startup profile | **DONE** | The bridge now has one contiguous fixed-capacity ABI based on the largest configured exact key; startup succeeds after the fix |
| Exact batched LMCache row routing and per-key startup completeness | **DONE in code/unit/startup** | Exact request order is covered by focused tests and every configured key is checked for every local SFA layer |
| Exact TP2 batch throughput checkpoint | **DONE** | On the eight-layer model, batches 1/2/4 reach 56/94/144 tok/s with LMCache and 72/124/184 tok/s with LMCache loading bypassed; normal and bypass scaling are nearly identical |
| Mixed prefill/decode native sparse-frontier lookup | **IMPLEMENTED; NPU RERUN PENDING** | The worktree queries frontiers only for request indices referenced by decode rows; rerun the concurrent smoke before closing |
| Exact batch-2/4 runtime replay qualification | **OPEN** | Throughput from concurrent clients does not prove that every steady step was pure decode or identify the exact replay key |
| Cross-layer island and event topology | **OPEN** | Capture an NPU trace proving the precise `N + 1` island plan, middle-island fusion, one eager retrieve per layer, and no nested wrappers |

### Current support and rejection matrix

| Mode | Current behavior | Production requirement |
| --- | --- | --- |
| Exact unpadded Q=1 | Cross-layer capture for configured exact request counts; TP2 batches 1/2/4 have a functional/performance checkpoint, but per-key batched replay is not trace-qualified | Prove actual replay and parity for every enabled key, then finish the P0 safety work |
| Long generation | Singleton repeated decode is stable with live metadata tensors | Prove every block/window/maximum-length boundary and every enabled exact key |
| Unconfigured or padded Q=1 | Runner disables replay before model forward; native SFA executes | Add planned padded buckets in R2 |
| MTP/speculative target | Startup/runtime rejection | Fixed candidate-width keys, row masks, disjoint scratch in R3 |
| Mixed prefill/decode | Whole scheduler step uses safe-native SFA; decode-only frontier filtering is implemented and awaits NPU rerun | R1 must qualify this native route; optional native-prefill/staged-decode row partition belongs to R5 |
| Compact-scratch LMCache native path | Available for many rows | Classify precisely when it is safe; never assume it is always a fallback |
| Legacy manager/free-paged/adapter | Rejected by staged path | Stay eager until their existing connector lifecycle and event behavior are independently qualified |
| TP | TP2 singleton startup/performance works and startup checks every local layer/key; collective failure admission is not complete | Restore rank-consistent startup/admission verdicts and TP1/2/8 gates |
| DP | Staged configuration rejected | Common structural bucket, rank-local masks, empty-rank handling in R4 |
| PP/multiple virtual engines | No capture namespace/lifecycle contract | Isolate cache epochs and in-flight slots or reject in R1-R3 |
| LoRA, CP/o-proj TP, C8, MLAPO, prefetch | Rejected in layer eligibility | Centralize the rejection before capture and enable individually in R5 |
| Ubatch/cascade | Runner rejects | Preserve rejection until state is invocation-safe |
| Sleep/wake, reload, cache recreation | No invalidation contract | R1 must invalidate/rebuild supported operations or reject them before state changes; R4 extends this to PP/multiple virtual engines |

## Blocking gap ledger

The P0 items are release blockers even for exact Q1. Feature breadth work
begins only after they are closed. P2 items are post-feature optimization or
optional cleanup; they do not block R1 and may not change the existing
connector API without a demonstrated correctness need.

| ID | Current evidence | Risk | Required outcome |
| --- | --- | --- | --- |
| P0.0 cross-layer compiler contract | Pre/retrieve/post decomposition and mock FX partitioning pass; singleton TP2 serving shows a large TPOT gain over both baselines | Throughput proves the route is useful but not the exact device partition; the precise middle-island topology has not been signed from an NPU trace | Prove an NPU island contains `Graph B(i) -> layer tail(i) -> Graph A(i+1)`, with nominal `N + 1` islands, one eager retrieve per layer, and no nested wrappers |
| P0.1 atomic dispatch | The runner authorizes configured exact-Q1 keys, disables replay before model forward for unsupported steps, and startup verifies every configured key/layer | Admission is not yet one immutable all-layer/TP plan, so dynamic metadata or rank drift after runner admission is not collectively rejected | One immutable all-layer plan, final TP admission, and TP phase verdicts around rank-local connector gaps; no layer-local fallback |
| P0.2 fallback safety | Unsupported and mixed-phase steps enter native execution with outer replay disabled. A concurrent trial exposed that frontier validation incorrectly included a prefill-only request; the decode-row-only fix is implemented but not NPU-qualified | Another path labelled native may still lack required latent data or apply decode-only validation to unrelated rows | First close the mixed-phase NPU rerun, then classify every route as `SAFE_NATIVE`, `RECOMPUTE`, or `FATAL` and prove the selected route has all required latent data |
| P0.3 pre-mutation validation | Startup verifies every configured key on every local SFA layer, including producer event, stable save binding, and fixed-capacity bridge layout | Live cache/frontier/pointer identity is still not validated as one rank-consistent plan before bootstrap | Validate the complete island/split plan, frontiers, buffers, and rank agreement before the first wait/write |
| P0.4 store-before-free | Decode-window release already waits for `completed_decode_window_saves`, but required latent/index groups and TP-owner aggregation are not yet fault-qualified | The only resident copy could be freed after an incomplete or rank-asymmetric save | Prove the existing completion path covers every required group/owner and frees only acknowledged aligned bundles; strengthen its implementation only where a failing test demonstrates a gap |
| P0.5 retrieval readiness | Strict misses and incomplete transfer masks can surface only when a layer generator resumes; exact top-k rows do not exist until Graph A | A post-selection load can fail after index/cache side effects | Fault-test the current `start_load_kv`/`wait_for_layer_load` path before mutation and after Graph A; use coordinated recovery for post-mutation failure, and extend the connector contract only if the existing lifecycle cannot express a correct result |
| P0.6 connector lifecycle failures | The production connector lifecycle and implicit `current_layer` cursor work on the qualified success path; exception, retry, cancellation, preemption, and rank-asymmetric behavior are not yet fully exercised | An unhandled failure could double-advance or strand connector state | Prove exactly-once callback progress and cleanup through the existing connector contract; prefer local guards/finalization, and require a reproduced correctness failure before proposing any shared API change |
| P0.7 stream ordering | Each captured pre records a persistent producer event that the existing LMCache `payload_event` path waits; each retrieve split also waits the next index group. Generic outer PIECEWISE replay synchronization and the one-time sparse wait fence remain | The event chain is not NPU-trace-qualified, while retaining generic fences can cap TPOT and jitter | Prove producer/load/following-island ordering, then remove only synchronization demonstrated redundant by the closed event chain |
| P0.8 stable ownership | Outer PIECEWISE wrappers own the cross-layer graph storage, and startup retains per-layer cache/event bindings, but the namespace does not cover weights, workspaces, cache epoch, virtual engine, or overlapping invocation | Stale pointers after lifecycle changes or state races | Island registry and buffer arena scoped by model/VE/cache epoch/key/in-flight slot, with invalidation |
| P0.9 bounded resources | Outer islands use ordinary piecewise graph-memory profiling; stream count still relies on the existing per-layer heuristic and PP is not included | KV sizing can overcommit HBM/streams if profiling misses lifecycle high-water or the stream heuristic is wrong | Measured graph/workspace high-water plus a conservative, topology-aware quota before service readiness |
| P0.10 qualification evidence | TP2 startup, deterministic output, prefix hit, decode save, repeat stability, singleton TPOT improvement, and batch-1/2/4 throughput checkpoints pass. Focused batch routing/startup tests pass, but client concurrency has not proven pure-decode key-2/4 replay | A synchronized HTTP launch does not control scheduler phase alignment, so batch responses alone can hide smaller-key replay or whole-step native execution | Add runtime per-key admission/replay evidence or a deterministic phase-aligned exerciser, then automate numerical, partition, trace, lifecycle, and failure matrices for every enabled exact key |
| P1.1 padded rows | Only remap boundary is persistent; builder allocates per-step CPU/NumPy/device metadata | Exact keys cause graph explosion/fallbacks and cannot safely pad | Stable fixed-capacity row arena, safe pad block/slots, masks through both logical SFA phases and connector filtering |
| P1.2 rich ACL dispatch | `StagedSFAGraphKey` collapses to legacy `BatchDescriptor(num_tokens)` | Padded Q1 and `SPEC_FIXED` entries can collide at equal token counts | Carry the full structural key through `ACLGraphWrapper` dispatch |
| P1.3 MTP scratch | Native metadata has row-specific groundwork; staged eligibility rejects it | Candidate rows can alias scratch or lose request-row order | Fixed-width profile, unique-request frontier expansion, disjoint scratch/targets, valid-row mask |
| P1.4 scheduler ownership | Input rows can be condensed/swapped after scheduler output is formed; scratch is configured through scattered environment reads | Request IDs, block rows, selected rows, and targets can describe different generations | Build plan after row condensation; use generation/step identity and typed KV scratch configuration |
| P1.5 DP/PP/concurrency | Exact co-scheduled Q1 request batching is implemented for configured sizes; DP/ubatch are rejected and active-key aliases are mutable without locking | Exact batching does not establish DP agreement, overlapping invocation safety, or virtual-engine isolation | Qualify exact batching first; then add DP-wide bucket agreement, per-VE/cache namespace, and either isolated in-flight slots or enforced no overlap |
| P1.6 compatibility | Layer eligibility, startup config, memory budgeting, and connector checks encode different support subsets | Service can reserve/capture before discovering an unsupported operator combination | One capability fingerprint and reason enum used by every stage |
| P1.7 mixed-phase hybrid replay | A step containing prefill and decode deliberately runs wholly native; the trial confirmed that ordinary client concurrency naturally produces this scheduler phase | Throughput can fall during arrivals even though already-decoding rows match a captured key | Optional R5 feature: compact/partition decode rows, run prefill through native MLA, replay a qualified staged decode key, and recombine outputs without changing LMCache's public connector API |
| P1.8 runtime key observability | Startup logs/counters establish captured keys, but the smoke client cannot prove which key a live scheduler step admitted and replayed | Qualification and operations can confuse a captured key with a used key or a safe-native mixed step | Expose low-cardinality per-key admission/replay/fallback counters and include them in smoke assertions |
| P2.2 vLLM-owned remap frontier | The current path obtains the committed frontier from bound LMCache metadata through `_get_connector_metadata` | This is connector-specific coupling and may complicate upstream review, but no runtime defect has been demonstrated for the pinned version pair | Optional cleanup: retain the remap boundary but derive it from the latent range vLLM actually released. Do not add a connector API solely for encapsulation or refactor resistance |
| P2.3 batch throughput efficiency | TP2 eight-layer throughput is 56/94/144 tok/s for batches 1/2/4 and 72/124/184 tok/s with LMCache loading bypassed. Normal/bypass ratios stay near 76-78%, so LMCache does not introduce a new batch-4 scaling collapse | Absolute LMCache cost remains material, while SFA/indexer, LM head, TP collectives, graph fences, or host work may limit aggregate scaling | After core feature support, attribute steady-step time by layer count and subsystem, then optimize only measured bottlenecks without weakening correctness, fallback, or the connector contract |

## Target ownership model

### 1. Capture namespace and structural key

Keep runtime values out of the graph key. Separate capture identity into two
levels:

```text
CaptureNamespace(
    model_instance_id, target_or_draft, device_rank,
    virtual_engine, kv_cache_epoch, island_id_or_layer_span,
    operator_fingerprint, connector_compatibility_id,
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
scoped to a compiled island plan, but they must be recorded and checked for
invalidation.

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
- the ordered outer-island/retrieve-split plan and every participating SFA
  layer's position in it;
- the remap frontier used by the qualified current path and the existing
  connector-callback sequence expected for every participating layer;
- scratch reservation and target-slot ownership;
- TP consensus result and a typed fallback/rejection reason;
- phase, mutation bit, callback-progress ledger, and finalization state.

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
- strong bridge tensors from Graph A to its retrieve split and following
  island, plus graph input tuples per island;
- reusable events for Graph-A producer and load completion;
- optional debug canaries/parity buffers kept outside the release arena.

The arena must support explicit `allocate`, `bind`, `quiesce`, `invalidate`, and
`release`. A slot cannot be reused until Graph B and connector save/load work
have completed. If only one invocation may be active, enforce that invariant
with a runtime guard rather than relying on today's scheduler behavior.

### 4. Capture registry

The registry owns the compiled outer islands, graph-pool entries, startup proof,
resource measurements, and state for every namespace/key. Required states are:

```text
UNALLOCATED -> ALLOCATED -> CAPTURING_ISLANDS
            -> PROVING_PARTITION_AND_ORDERED_REPLAY -> READY
READY -> QUARANTINED | INVALIDATING -> RELEASED
```

Capture every enabled key before the worker reports ready. Retry is allowed only
after the failed namespace has been fully invalidated and all graph/pool/arena
objects have been rebuilt. Sleep/wake, model/weight reload, KV cache
reinitialization, virtual-engine recreation, operator workspace relocation, or
supported connector software/configuration change creates a new cache epoch and invalidates old
entries.

## Step and failure state machine

The runner plan must make the mutation boundary explicit without replacing the
existing vLLM connector lifecycle:

```text
CREATED
  -> STATIC_PREFLIGHTED
  -> FINAL_ADMITTED               # TP verdict before connector/cache mutation
  -> BOOTSTRAP_INDEX_WAIT_STARTED  # first connector/cache side effect
  -> BOOTSTRAP_INDEX_READY_AND_AGREED
  -> ISLAND_0_ENQUEUED              # contains Graph A(0)
  -> SPLIT_i_STARTED                # latent-load(i) + index-load(i+1)
  -> SPLIT_i_READY_AND_TP_AGREED
  -> ISLAND_i_PLUS_1_ENQUEUED       # B(i) + tail(i) + A(i+1)
  -> ISLAND_i_PLUS_1_DONE_EVENT
  -> ... next retrieve split ...
  -> FINAL_ISLAND_ENQUEUED          # contains B(N-1) + tail(N-1)
  -> FINAL_ISLAND_DONE_EVENT
  -> MODEL_DONE
  -> SAVES_DONE
  -> FINISHED

Before BOOTSTRAP_INDEX_WAIT_STARTED, a route change may use a proven safe route.
At/after BOOTSTRAP_INDEX_WAIT_STARTED, no implicit native fallback is allowed; coordinated
recovery only.
```

Required rules:

1. Static preflight validates every outer island and retrieve split in the
   compiled plan, the arena, scratch/destination capacity, connector
   capabilities, and static rank compatibility.
2. Final TP admission runs before the index wait. The current remap frontier and
   connector metadata/callback assumptions are validated for the pinned
   software/configuration fingerprint; disagreement rejects the staged route.
3. The index wait does not begin until final admission succeeds. Its cache write
   and connector progress are the first side-effect boundary.
4. The bootstrap index operation and every rank-local retrieve split must
   expose a connector-guaranteed rank-consistent result or be followed by a TP
   phase verdict before any rank enters the following collective-bearing
   island.
5. The runner calls each required index/latent layer callback exactly once,
   including the combined `latent(i) + index(i+1)` transition, and verifies
   with injected retry/error tests that no path silently increments the
   existing connector cursor twice.
6. Cancellation/preemption before the side-effect boundary may switch to
   scheduler recovery. After the boundary it marks the request/cache epoch as
   needing recovery and cannot reuse partially written state as successful.
7. A connector timeout is not automatically a native fallback. It is
   `SAFE_NATIVE` only if admission proved the necessary latent remains resident;
   otherwise schedule recomputation or fail the request.
8. Completion of the island containing Graph B must precede arena-slot reuse
   and any connector resource release that could invalidate its inputs;
   enqueueing the island is not proof of completion. Use the existing
   stream/event behavior unless testing proves it insufficient.
9. Exceptions at every phase run exactly one finalizer. Saves are released only
   after the graph result is accepted. No empty/no-forward/recalc-last path may
   leak or double-finish the existing connector lifecycle.
10. A bad key is quarantined for future steps after a replay error. The current
   mutated step follows abort/recovery; future steps may use safe native mode.

## Existing connector contract and API-change threshold

R1 preserves the production vLLM connector lifecycle:

```text
bind_connector_metadata
  -> start_load_kv
  -> wait_for_layer_load for each index/latent layer
  -> save_kv_layer
  -> wait_for_save
  -> clear_connector_metadata
```

The staged executor inserts Graph A and Graph B around the existing selective
layer load; it does not by itself justify a new LMCache-vLLM transaction API.
The current LMCache/LMCache-Ascend implementation, including its established
selected-row, target-slot, and event handling, remains the reference behavior.

Changing the shared connector API is not recommended for encapsulation,
cleanliness, or possible future refactors. It requires a reproduced correctness
failure showing that the existing lifecycle cannot express the required
ordering, completion, failure, or ownership semantics. Before proposing an API
change, fault injection must first rule out a local runner guard, finalizer,
event hand-off, or connector-internal fix. Any unavoidable extension should be
general enough for upstream vLLM connector review rather than staged-SFA- or
LMCache-specific plumbing.

The current `_get_connector_metadata` frontier lookup is a narrow, pinned-version
integration dependency, not an R1 blocker by itself. Removing that dependency
is useful optional cleanup: retain the remap boundary, but derive it from the
latent range vLLM actually released. Do not add a frontier getter to the shared
connector API unless a correctness test demonstrates that no vLLM-owned source
can represent the needed value.

The final selective payload must preserve candidate-row order and may contain
duplicate request IDs. For every valid row, `selected_tokens[i]`,
`request_ids[i]`, `target_slot_mapping[i]`, and the valid mask describe the same
row. A short transfer or source disappearance after Graph A is a failure that
requires coordinated recovery; it is not a native fallback.

### Store-before-free and retrieval readiness

Stage-2 memory saving is correct only if the existing completion signal,
scheduler release, and later retrieval describe the same covered range:

```text
prefill KV produced
  -> LMCache stores the required data and wait_for_save reaches completion
  -> worker reports completed_decode_window_saves through the existing output
  -> scheduler accepts the completion and marks that range releasable
  -> aligned whole latent block bundles are freed
  -> staged/native compact-scratch execution retrieves the released range
```

Prompt length or chunk rounding is not evidence of persistence. Missing
metadata, cache mapping, indexer registration, a short transfer mask, or an
empty source may not be treated as silent save/load success. A failed store
keeps the resident blocks, triggers recomputation before free, or fails the
request according to policy. Fault tests must prove that TP ownership and
`save_only_first_rank` cannot report completion prematurely.

R1 first validates source lifetime and transfer completion through the existing
connector callbacks and stream/event behavior. If a forced-unpin, restart,
timeout, or post-Graph-A failure exposes a correctness gap that cannot be fixed
inside the current implementation, that evidence defines the minimum API
extension to propose. A speculative lease/transaction API is not a prerequisite.

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

## Target cross-layer retrieve-split architecture

The target is supported by the current compiler architecture, with one required
decomposition:

- Ascend `FULL`/`FULL_DECODE_ONLY` already sets `splitting_ops=[]` and wraps the
  whole model with `ACLGraphWrapper`. Existing e2e coverage uses this path.
  `model_runner_v1.py` rejects it specifically when a connector advertises
  layerwise callbacks, because replay would bypass their Python execution.
- In PIECEWISE mode, `split_graph` isolates configured split nodes and compiles
  each maximal run of ordinary FX nodes between them. With retrieval as the
  only staged-SFA split, this directly creates cross-layer graph islands.
- Today `AscendMultiHeadLatentAttention.forward` emits only
  `vllm::mla_forward`, and `platform.py` forcibly adds that whole opaque op to
  `splitting_ops`. The SFA pre phase, LMCache callback, and post phase therefore
  cannot be partitioned independently.
- The model-facing MLA call normally supplies positions and hidden states, not
  populated KV-cache/attention-metadata arguments. The current custom-op body
  deliberately resolves those from `ForwardContext`. The staged decomposition
  must not broaden every upstream model call. For R1's single virtual engine,
  bind model-owned KV tensors and stable runner arena metadata to the Ascend MLA
  layer/plan before capture, then expose mutations and cross-split tensor
  dependencies in the custom-op operands. Any remaining prebound implicit
  tensor must be covered by the namespace, address, and update-event contract.

Consequently this is feasible, but not as a configuration-only change. For the
qualified staged route, expose three logical custom ops: graph-safe SFA pre,
eager LMCache retrieve, and graph-safe SFA post. Only the retrieve op is a
split. The pre/post ops use explicit tensor operands and contain no dynamic
Python/connector decisions, so the surrounding outer ACL wrapper captures their
NPU execution. The existing `vllm::mla_forward` route remains unchanged for
configurations that do not enable this target and as the temporary reference
executor. Within an enabled process, prefill and any pre-admitted native step
still execute with graph runtime `NONE`; the decomposed op sequence must route
that eager invocation through the existing native MLA implementation exactly
once and make retrieve/post no-ops as appropriate. Cross-layer decode support
may not regress dense prefix retrieval, prefill, or safe native behavior.

For layer `i`, use the following nominal island schedule:

```text
bootstrap: index-load(0)

Graph(0): Graph A(0)
Split(0): latent-load(0) + index-load(1)
Graph(1): Graph B(0) + layer-tail(0) + Graph A(1)
Split(1): latent-load(1) + index-load(2)
...
Graph(N): Graph B(N-1) + layer-tail(N-1)
```

Combining `latent-load(i)` with `index-load(i+1)` is required to obtain one
connector split per layer. This does not require a new connector API: the split
may invoke the existing layer callbacks sequentially with static current/next
layer identities. The feasibility spike must prove that the current two-group
cursor does not advance on the index-group wait and advances exactly once on
the latent-group wait. If that cannot be proved, the target becomes two splits
per layer and loses the nominal `N + 1` cardinality; it must not hide the extra
boundary.

The retrieve split op must accept selected rows, destination slots, affected
cache tensors, and a dependency token as real operands. Its schema must declare
cache mutation/aliasing so FX cannot remove or reorder it. Static layer identity
and request IDs may be resolved by the eager body from the runner-owned plan;
the tensor dataflow and ordering may not depend only on hidden
`ForwardContext` side effects. The preceding island publishes a producer event,
the connector load stream waits and scatters, and the following island waits on
a load-completion event. Generic piecewise replay synchronization must be
disabled only for these event-closed islands after their stable input ownership
is proved; the generic wrapper default remains unchanged.

This design produces graphs that cross layer boundaries; it does not produce
one uninterrupted full-model graph. The current full-graph support proves that
ordinary device execution can span all transformer layers, but it does not make
LMCache's request-specific Python/CPU callback capturable. The earlier
two-wrapper commit remains the reference and rollback route until the
cross-layer target has its own numerical, lifecycle, trace, and performance
qualification; production registry/resource work is built for outer islands.

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
- mixed prefill/decode scheduler steps. A dense prefix load followed by an
  admitted exact-Q1 decode is already part of the singleton checkpoint; other
  prefix-cache phase/layout combinations remain unqualified;
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
R_total = sum(namespaces, keys, compiled_islands, in_flight_slots)(
              island_graph_pool + island_workspace
            + persistent_inputs + strong_bridge_outputs + owned_output
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
- per-island replay, per-layer logical Graph-A/Graph-B timing, index/latent
  callback timing, and save timing;
- connector callback starts/layer transitions/completions/failures/timeouts and
  duplicate-callback detection;
- active/padded/selected rows and cache hit/partial/miss outcome;
- reserved/measured graph, arena, scratch, and connector memory;
- sampled parity/top-k/cache-write mismatch counts;
- TP/DP decision disagreement and lifecycle invalidation reason.

Profiler traces must show the bootstrap index wait, exactly one retrieve split
per participating layer, the compiled outer-island plan, explicit
producer/load/consumer events, and no hot-path global synchronize. They must
also show that each middle island contains `Graph B(i)`, the transformer tail,
and `Graph A(i+1)`. A health endpoint/log summary should state enabled keys,
connector compatibility fingerprint, qualification fingerprint, resource
budget, and fallback/recompute policy.

Use a runtime kill switch that prevents new staged plans. It may route future
steps to a proven safe path; it cannot retroactively fall back a mutated step.

## Code impact map

| Area | Required change |
| --- | --- |
| `vllm_ascend/worker/model_runner_v1.py` | Build/finalize the immutable execution plan; enumerate all local SFA layers; do TP consensus; own arenas/registry; handle lifecycle and parity sampling |
| `vllm_ascend/ascend_forward_context.py` | Pass a plan handle and layer cursor instead of loose dummy/key/parity attributes; carry cache epoch/virtual-engine identity |
| `vllm_ascend/attention/sfa_v1.py` | Expose graph-safe logical pre/post kernels with explicit tensor contracts; keep nested wrappers and layer-local fallback out of the target path; add padded/MTP masks |
| `vllm_ascend/attention/utils.py` | Keep current LMCache frontier/capability access localized and version-qualified; optionally replace the frontier source with vLLM-owned released-range state after R1 |
| `vllm_ascend/distributed/kv_transfer/sparse_offload/prepare_sparse_indices.py` | Extend the fused sparse-index preparation contract with disjoint MTP ranges and device-side bounds assertions where needed |
| `vllm_ascend/ops/mla.py`, `vllm_ascend/platform.py` | Add the qualified staged decomposition and retrieve-only mutation-aware split; stop splitting at the whole `vllm::mla_forward` boundary only for that route; preserve existing non-staged MLA behavior |
| `vllm_ascend/worker/worker.py` | Topology-aware measured memory reservation, local PP layer count, target/draft and lifecycle accounting |
| `vllm_ascend/utils.py` | One compatibility fingerprint/reason system; rich capture keys; resource-driven bucket selection; actual stream accounting |
| `vllm_ascend/envs.py` | Typed operational mode/buckets/debug sampling; centralize all staged-SFA environment reads |
| `vllm_ascend/compilation/acl_graph.py` | Accept full structural dispatch keys, expose bounded invalidation/destruction and resource measurements, and add an opt-in event-closed replay policy without changing the generic synchronization default |
| vLLM scheduler/input batch/KV managers | Final row-generation plan, typed scratch/window config, capacity enforcement, completion-gated latent release, and lifecycle/recompute ownership; optional vLLM-owned released frontier |
| LMCache-NPU vLLM adapter | Preserve the existing connector API; fix only reproduced store/retrieve/finalization failures and add fault tests around current callbacks |
| LMCache-Ascend connector | Preserve existing two-group/rank-specific and stream/event behavior; expose a new shared API only if a correctness gap cannot be fixed internally |
| smoke/unit/NPU test tools | Multi-key graph-count parsing, plan/failure injection, lifecycle/parallel matrices, automated trace assertions |

## Delivery route and dependencies

### Immediate execution order after the current trial

1. Rerun the concurrent smoke with the decode-row-only frontier fix. Close the
   mixed-phase native-route issue only if both the prefill and decode requests
   complete, the decode row still uses the native sparse path correctly, and no
   prefill-only frontier is requested.
2. Add per-key runtime admission/replay/fallback evidence, then drive a
   phase-aligned pure-decode size-2 step. Prove output parity, LMCache row
   identity, key-2 replay, and the `1 -> 2 -> 1` transition. Client-side request
   synchronization alone is not sufficient evidence.
3. Capture the cross-layer NPU profile for one qualified key and sign the
   `N + 1` island, middle-island fusion, eager retrieve, and event-ordering
   assertions.
4. Continue W1 tightening and fault qualification before adding padded Q1,
   MTP, DP, or hybrid mixed-phase replay. Do not change the shared LMCache-vLLM
   API unless a reproduced correctness failure cannot be repaired within the
   existing lifecycle.
5. Keep performance regression gates active throughout feature work, but defer
   discretionary throughput optimization until the core padded-Q1, MTP, and
   serving-parallelism feature packages are stable. Execute W8 before expanding
   into the lower-priority optional-mode matrix.

### W0: prove the cross-layer partition and capture contract

Status: **implementation and singleton functional/performance checkpoint
complete; trace exit still open**.

- **Done:** add a narrowly gated staged decomposition in `ops/mla.py`: graph-safe pre,
  eager retrieve, and graph-safe post. Make selected rows, destinations,
  affected caches, and a dependency token explicit operands.
- **Implemented, ownership hardening remains:** add an Ascend-owned per-layer
  binding for model-owned KV tensors and stable
  plan/arena metadata; do not change the model-facing MLA or shared connector
  API. Assert binding identity before capture and replay.
- **Done:** register only the retrieve op as a staged-SFA splitting op. Stop using
  `vllm::mla_forward` as the boundary only for this qualified route; preserve
  the existing path for all other MLA modes.
- **Implemented; mixed-phase NPU rerun pending:** in the enabled process,
  preserve prefill and pre-admitted native execution
  under runtime mode `NONE` by calling the existing MLA implementation exactly
  once; add regression tests that retrieve/post do not duplicate its effects.
- **Done:** reuse the current exact-Q1 Graph-A/Graph-B bodies without nested
  `ACLGraphWrapper` instances in the new route.
- **Done:** add a CPU/mock FX partition test proving that two adjacent layers produce
  `pre(0) | retrieve(0) | post(0)+tail(0)+pre(1) | retrieve(1) | post(1)` and
  that retrieval remains eager and mutation-ordered.
- **Functional/performance trial done; trace open:** on NPU, the current
  eight-layer test model and singleton exact-Q1 key have
  passed startup, output, LMCache, and TPOT trials. The remaining W0 proof is to
  assert nominal `local_sfa_layers + 1` outer graph islands, zero nested staged
  entries, one retrieve call per layer, stable addresses, and a middle island
  containing `Graph B(i)`, the layer tail, and `Graph A(i+1)`.
- **Implemented; trace/fault proof open:** invoke `latent-load(i)` and
  `index-load(i+1)` through the existing connector
  callbacks in one split and prove exact cursor progression. Do not change the
  connector API for this spike.

Exit: deterministic token/logit/KV/top-k parity matches the current executor;
the partition and profiler trace match the intended islands on TP2; warm replay
does not execute connector callbacks inside a graph or recapture; prefill,
dense prefix hit/TTFT behavior, decode save, and pre-admitted native fallback
remain unchanged. If any exit condition fails, stop and document the concrete
compiler/operator/connector constraint before doing production ownership work.

### W1: freeze support behavior and qualify the existing connector lifecycle

Status: **implementation complete; NPU fault/soak sign-off pending**. Hybrid
mixed-phase replay is an R5 feature and is not part of W1.

- Convert the compatibility table into typed capability/reason data shared by
  startup, resource planning, runner, and layer assertions.
- Document `SAFE_NATIVE` versus `RECOMPUTE` versus `FATAL` for every current
  rejection, especially `prompt_len < index_topk` and missing/partial cache.
- Keep the existing vLLM/LMCache connector API and callback order unchanged.
- Fault-test store completion across every required layer/group and TP owner,
  including `save_only_first_rank`, and verify that only acknowledged aligned
  block bundles are released.
- Inject sparse source miss/partial transfer, timeout, exception, cancel,
  preemption, retry, and rank-asymmetric failure around bootstrap, each
  combined retrieve split, and each following island.
- **Done:** make partial batched load setup and cancellation close every published
  layerwise generator, release sparse request state/pins, and leave the next step
  reusable. The success path adds no device synchronization or tensor transfer.
- **Done:** publish decode-window completion only after the final store fence;
  atomically discard group/finalizer failures while preserving the retry frontier.
- **Done:** fault-test deferred consumer joins for submission, event-record, and
  event-wait failure. Detailed timeline assertions remain in W8 as agreed after
  the functional and throughput qualification.
- Move scratch/window capacity into typed vLLM-owned configuration where
  practical and record the pinned connector/software/configuration fingerprint.
- Keep the completed profiler-smoke contract (`ignore_frontend=true`,
  configurable TP rank count, offline analysis). Compiled-island timeline and
  graph-count analysis is part of W8 rather than a W1 implementation gate.
- Add runtime evidence for the admitted structural key and the route actually
  executed (`STAGED`, `SAFE_NATIVE`, `RECOMPUTE`, or `FATAL`), so batch smoke
  tests cannot mistake captured keys for replayed keys.

API-change gate: propose a shared connector extension only if a fault test
demonstrates a correctness failure that cannot be fixed through the existing
lifecycle. Removing `_get_connector_metadata` and deriving the remap frontier
from vLLM-owned released-range state is optional P2.2 cleanup, not W1 exit work.

Implementation exit: every rejection has a tested action; mocked faults prove
store-before-free and exactly-once cleanup for setup, group finalization, final
store fencing, event submission/record/wait, cancellation, and retry. Release
sign-off still runs these paths under TP2/TP8, including a rank-asymmetric fault
and a normal request immediately afterward; do not add an always-on TP
consensus unless that run reproduces rank divergence.

### W2: island registry, prebound arena, invalidation, and resource budget

Status: **active; first R1 ownership slice implemented, NPU validation pending**.

- **Done for R1:** dispatch ACL graph entries by the existing full
  `StagedSFAGraphKey`; generic graph paths retain their legacy
  `BatchDescriptor` key. Add namespace/in-flight ownership before PP, virtual
  engines, or overlap are admitted.
- Allocate stable per-key bridge tensors from Graph A through retrieval into the
  following island so every island/split binding can be validated before the
  bootstrap wait.
- **Done for R1:** reject KV-cache recreation after staged capture has started,
  before any cache mutation. Implement cache epochs, quiesce/invalidate/rebuild,
  quarantine, and bounded graph destruction before dynamic lifecycle support.
- Implement the offline-bound or two-pass resource strategy and feed the result
  into KV sizing without circular measurement.
- Add an opt-in event-closed replay policy for these islands and remove
  per-island host synchronization only after the event and ownership proof.
- Remove release-mode hot signature scans after arena/epoch proof.

Exit: all island/split bindings are known before side effects; exact Q1 survives
`1 -> B -> 1` and every allowed/rejected R1 lifecycle transition with bounded
memory, no stale-address replay, and no hot-path stream/device synchronize.

### W3: runner-owned cross-layer step plan

- Introduce the execution plan and the static-preflight -> final-admission ->
  bootstrap -> alternating island/split state machine.
- Validate every island, split, and participating layer before bootstrap; do
  final per-step TP admission and phase verdicts after rank-local connector
  gaps.
- Forbid layer-local fallback after planning and any fallback after bootstrap.
- Implement one runner finalizer around the existing connector lifecycle plus
  final-island completion hand-off for success/error/cancel/no-forward paths.
- Keep current exact-Q1 math unchanged except for consuming the plan and
  explicit bridge tensors.

Exit: injected local/rank-asymmetric failure at every layer/phase cannot cause
mixed graph/native execution, collective divergence, double cursor movement,
early slot/resource release, or reuse of partial state. Depends on W1 and W2.

### W4: R1 cross-layer exact-Q1 NPU qualification and rollout

- Run the complete numerical/boundary/cache/TP/lifecycle matrix against both
  the archived two-wrapper commit and the valid native LMCache route.
- Automate partition/operator/event assertions and performance gates, including
  graph-island cardinality and zero nested staged wrappers.
- For every configured exact size, prove a phase-aligned pure-decode replay,
  request-row identity through LMCache, numerical parity, and key transitions
  including `1 -> B -> 1`; include mixed arrival/decode steps as safe-native
  fallback tests rather than counting them as staged batch coverage.
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

### W8: post-feature throughput and latency optimization

Status: **planned immediately after core feature support; lower priority than
W5-W7 feature breadth, higher priority than optional-mode expansion**.

- Preserve no-profiler batch-1/2/4 baselines with and without LMCache loading;
  report step time plus throughput/TTFT/TPOT p50, p90, and p99 rather than only
  peak aggregate throughput.
- Attribute the batch-dependent step-time slope with short profiles and a
  1/4/8-layer sweep. Separate SFA index selection, sparse attention, projections,
  LM head/logits/sampling, TP collectives, outer-island replay/fences, scheduler
  and model-runner host work, LMCache submission, transfer, and consumer join.
- Treat the current TP2 result as evidence against a new LMCache batch-4
  serialization bottleneck, not as proof that its absolute 22-24% gap is
  irreducible. Optimize connector submission or transfer only when the profile
  identifies it on the critical path.
- Replace generic replay synchronization only after W1-W2 event, ownership, and
  failure proofs demonstrate that a narrower event dependency is correct.
- Qualify every optimization on the full target layer count and TP2/TP8; keep
  the eight-layer model as an iteration tool because its full LM head can
  exaggerate non-layer costs.
- Do not trade output quality, deterministic behavior, failure safety, resource
  bounds, or the existing public LMCache-vLLM contract for throughput.

Exit: the signed workload meets its aggregate-throughput and TTFT/TPOT p50/p99
targets on the full target model; traces account for the remaining batch-scaling
loss, no correctness/feature matrix regresses, and every retained synchronization
has a documented dependency purpose. Depends on W1-W4 and should follow stable
W5-W7 feature behavior.

### W9: optional execution modes

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
- existing connector callback counts and cursor progress under an exception
  injected at every state;
- store failure/short save cannot advance frontier or free resident blocks;
- readiness miss/short transfer is handled according to the pre/post-mutation
  recovery policy;
- connector destination pointer rebind/allocator reuse rebuilds native plans;
- source pin timeout cannot invalidate data still consumed by the current step;
- delayed completion of an island containing Graph B prevents early
  connector-resource, scratch, or arena-slot reuse;
- missing indexer registration and connector import-order variants fail staged
  admission rather than silently changing capabilities;
- cancel/preempt/recompute/recalc-last/empty-step/late-save finalization;
- cache-epoch invalidation and bounded entry release;
- resource formula for local PP layers, target/draft, every key and slot;
- offline-bound overrun and two-pass teardown/recapture resource paths;
- smoke log/count tests for one/multiple keys, TP1/2/8 trace counts, and the
  worker-only (`ignore_frontend=true`) profiler requirement;
- exact-batch row routing and runtime key-selection counters for `1 -> B -> 1`;
- a mixed scheduler step in which prefill-only request metadata has no sparse
  frontier while every decode row does, proving native lookup touches only the
  decode request indices.

### NPU correctness matrix

Compare staged output with both the resident eager reference and the native
compact-scratch LMCache path where that native route is valid. Check full layer
output, top-k indices, current-token latent/index writes, scratch contents,
logits, and deterministic generated tokens.

Axes include:

- every enabled capacity, every real padded size, and `1 -> B -> 1`;
- phase-aligned pure-decode batching for every enabled exact key, with runtime
  evidence that the intended key—not a singleton key or safe-native route—was
  executed;
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

- target executor cardinality equals the compiled island plan: nominally local
  SFA layers plus one per structural key when retrieval is the only split, with
  no nested Graph-A/Graph-B entries;
- runtime admission/replay counters identify the exact structural key and do
  not increment staged replay for a whole-step mixed-phase native route;
- no nested two-wrapper executor is present in the production R1 branch or
  included in target resource/cardinality accounting;
- no unbounded recapture after readiness;
- every intended operator is inside the correct replay island, including
  `Graph B(i) -> layer tail(i) -> Graph A(i+1)` in every middle island;
- bootstrap index wait precedes Graph A(0); producer event precedes each
  selective load; load completion precedes the island containing Graph B;
- no hot-path `.item()`, global stream/device synchronize, or tensor allocation
  in any target replay island;
- island outputs alias documented owned storage and all bridge tensors live for
  the required slot lifetime.

### Reliability and performance gates

- startup capture high-water and steady-state HBM stay within the advertised
  reservation and leave the configured KV budget intact;
- stream/event counts stay within the runtime quota;
- long-generation and high-churn soak has zero parity, cursor, stale-pointer,
  or cross-request failures;
- arrival/decode churn across exact batch sizes has zero unexpected fallback,
  and every expected mixed-phase fallback has a typed reason;
- throughput and TPOT improve for the workload/buckets that justify the feature;
- TTFT/TPOT p50 and p99 have no unapproved regression from index waits, event
  hand-off, plan building, or first-use synchronization;
- host launch count, outer-island count, and eager retrieve gaps match the trace
  budget.

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
   Mutated in-flight steps still follow the qualified recovery policy.
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
3. Existing connector progress, store completion, and stream/event ordering are
   proven exactly once on success/error/cancel/recompute paths. A new shared API
   is required only if this cannot be achieved through the existing contract.
4. Unsupported values have a proven `SAFE_NATIVE`, `RECOMPUTE`, or `FATAL`
   action; no unsafe fallback exists after mutation.
5. Startup capture and ordered replay succeed on every participating rank and
   numerical parity covers the signed boundary/topology matrix.
6. Padding/MTP rows, when enabled, have masks and disjoint cache/scratch/target
   ownership through both logical SFA phases, every retrieve split, and
   LMCache.
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
