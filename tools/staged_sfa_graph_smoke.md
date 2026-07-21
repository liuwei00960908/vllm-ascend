# Cross-layer staged SFA smoke

This checkpoint targets configured exact, unpadded Q=1 decode batch sizes.
LMCache retrieval is the only staged-SFA FX split; the surrounding PIECEWISE
ACL graphs should cross transformer-layer boundaries. Padding, MTP, DP, LoRA,
CP, MLAPO, sparse-C8, and weight prefetch remain outside the supported envelope.

## 1. Run correctness and TPOT first

Start a fresh server without `--profiler-config`. Keep the same model,
LMCache config, TP size, shortened-layer override, and benchmark prompt used
for the two-graph baseline. The essential settings are:

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1 \
HCCL_DETERMINISTIC=strict \
PYTHONHASHSEED=0 \
LMCACHE_DECODE_WINDOW_SAVE_WINDOW_SIZE=256 \
LMCACHE_CONFIG_FILE="${LMCACHE_CONFIG}" \
VLLM_ASCEND_DSA_DISABLE_INDEX_LMCACHE=0 \
VLLM_ASCEND_DSA_UNBUNDLE=1 \
VLLM_ASCEND_DSA_TWO_GROUPS=1 \
VLLM_ASCEND_DSA_SHRINK_LATENT=2 \
VLLM_ASCEND_SFA_STAGED_GRAPH=1 \
vllm serve "${MODEL}" \
  --tensor-parallel-size 2 \
  --max-model-len 20000 \
  --max-num-seqs 32 \
  --max-num-batched-tokens 16384 \
  --no-enable-prefix-caching \
  --compilation-config '{"cudagraph_mode":"PIECEWISE"}' \
  --kv-transfer-config \
    '{"kv_connector":"LMCacheAscendConnectorV1Dynamic","kv_role":"kv_both","kv_connector_module_path":"lmcache_ascend.integration.vllm.lmcache_ascend_connector_v1"}' \
  2>&1 | tee "${LOG}"
```

Add your existing `--hf-overrides`, model path, memory utilization, host, and
port arguments unchanged. Keep
`VLLM_ASCEND_SFA_STAGED_GRAPH_CAPTURE_SIZES` unset for this singleton baseline;
the default remains `1`.

Run the same deterministic request at least twice. The acceptance gates are:

1. tokens exactly match the two-graph baseline and repeated requests;
2. the repeated prompt receives the expected LMCache dense-prefix hit and TTFT
   reduction;
3. decode-window saves still appear at the expected boundaries;
4. no-profiler concurrency-1 TPOT is lower than the two-graph baseline (about
   170 ms in the previous measurement).

The log-only smoke is useful before profiling:

```bash
python3 tools/staged_sfa_graph_smoke.py \
  --base-url http://127.0.0.1:9000 \
  --model "${MODEL}" \
  --server-log "${LOG}" \
  --profile-dir "${PROFILE_DIR}" \
  --expected-ranks 2 \
  --skip-profile
```

It requires the cross-layer startup-capture marker but does not prove output
quality or performance.

## 2. Qualify exact batch 2

First add only one new key to the no-profiler server:

```bash
VLLM_ASCEND_SFA_STAGED_GRAPH_CAPTURE_SIZES=1,2 \
```

Then run two synchronized, equal-length requests:

```bash
python3 tools/staged_sfa_graph_smoke.py \
  --base-url http://127.0.0.1:9000 \
  --model "${MODEL}" \
  --server-log "${LOG}" \
  --profile-dir "${PROFILE_DIR}" \
  --expected-ranks 2 \
  --expected-keys 2 \
  --concurrency 2 \
  --skip-profile
```

Run key switching in the order `1 -> 2 -> 1`, compare both batch responses to
native execution, and confirm no cross-request contamination. A mixed
dense-prefix/sparse-decode step intentionally uses the native path; incomplete
sparse metadata remains fatal.

## 3. Profile only after the no-profiler gate passes

Restart with a unique log/profile directory and add:

```bash
--profiler-config \
  "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"${PROFILE_DIR}\",\"torch_profiler_with_stack\":false,\"ignore_frontend\":true}"
```

Then run:

```bash
python3 tools/staged_sfa_graph_smoke.py \
  --base-url http://127.0.0.1:9000 \
  --model "${MODEL}" \
  --server-log "${LOG}" \
  --profile-dir "${PROFILE_DIR}" \
  --expected-ranks 2
```

`ignore_frontend=true` is required so `/stop_profile` finalizes only TP worker
traces. To profile exact batch 2, add `--expected-keys 2 --concurrency 2`;
the first synchronized request owns the single profiler start/stop interval
while both requests continue streaming.

## 4. MindStudio acceptance

For one rank and one steady decode step, verify:

1. `sfa_cross_layer::bootstrap` occurs once before the first captured island;
2. `sfa_cross_layer::lmcache_retrieve` occurs once per local SFA layer and is
   outside every ACL model replay;
3. the middle replay contains post-compute for layer N, intervening transformer
   work, and pre-compute for layer N+1;
4. the captured producer event orders selection before LMCache payload use, and
   the next-layer index wait orders the following replay;
5. the timeline contains no nested per-layer pre/post ACL graph replays.

The expected structural reduction for N SFA layers is from roughly `3N + 1`
replays in the old outer-plus-two-inner design to `N + 1` outer replays, with N
eager retrieval splits. Trace structure alone is not a performance proof; use
the no-profiler TPOT comparison as the final checkpoint.

## Failure triage

| Symptom | First check |
| --- | --- |
| No cross-layer startup marker | Confirm the staged flag, unbundle/two-groups/shrink-latent settings, PIECEWISE mode, and active LMCache connector. |
| Authorized key becomes ineligible | The layer-level exception identifies metadata or feature drift after runner admission. |
| Warmup/capture incomplete | Confirm every local SFA layer created a persistent producer event and stable cache binding. |
| Output mismatch | Inspect the producer-event/load-stream dependency and whether the next-layer index wait precedes its captured pre-compute. |
| Prefix TTFT does not fall | Confirm the startup command actually includes the LMCache connector and inspect dense store/retrieve logs. |
| TPOT is not below baseline | Count ACL model replays; any nested pre/post replay means the retrieve-only partition was not selected. |
