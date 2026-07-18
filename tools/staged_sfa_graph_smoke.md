# Staged SFA graph TP8 smoke

This smoke is intentionally restricted to the first proof-of-concept target:

- one GLM-5.1 request at a time;
- tensor parallel size 8;
- no MTP/speculative decoding and no LoRA;
- `VLLM_ASCEND_DSA_SHRINK_LATENT=2`;
- `PIECEWISE` graph mode;
- two captured SFA regions with eager LMCache retrieval between them.

Use a fresh log and a unique profiler directory for every run. The automated
checks reject old feature-specific failures and only consider newly written
`trace_view.json` files.

## 1. Start the server

Set the paths for the container, then run this in terminal A:

```bash
MODEL=/workspace/models/GLM-5.1-w4a8
LMCACHE_CONFIG=/workspace/qzy/lmcache_config.yaml
RUN_ID=$(date +%Y%m%d_%H%M%S)
LOG=/workspace/qzy/staged-sfa-${RUN_ID}.log
PROFILE_DIR=/workspace/qzy/staged_sfa_profile_${RUN_ID}
printf 'LOG=%q\nPROFILE_DIR=%q\n' "${LOG}" "${PROFILE_DIR}"
mkdir -p "${PROFILE_DIR}"

ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
HCCL_DETERMINISTIC=strict \
LMCACHE_DECODE_WINDOW_SAVE_WINDOW_SIZE=256 \
LMCACHE_CONFIG_FILE="${LMCACHE_CONFIG}" \
VLLM_ASCEND_DSA_DISABLE_INDEX_LMCACHE=0 \
VLLM_ASCEND_DSA_UNBUNDLE=1 \
VLLM_ASCEND_DSA_TWO_GROUPS=1 \
VLLM_ASCEND_DSA_SHRINK_LATENT=2 \
VLLM_ASCEND_SFA_STAGED_GRAPH=1 \
vllm serve "${MODEL}" \
  --gpu-memory-utilization 0.9 \
  --trust-remote-code \
  --tensor-parallel-size 8 \
  --max-model-len 20000 \
  --max-num-seqs 32 \
  --max-num-batched-tokens 16384 \
  --host 0.0.0.0 \
  --port 9000 \
  --no-enable-prefix-caching \
  --compilation-config '{"cudagraph_mode":"PIECEWISE"}' \
  --profiler-config \
    "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"${PROFILE_DIR}\",\"torch_profiler_with_stack\":false}" \
  --kv-transfer-config \
    '{"kv_connector":"LMCacheAscendConnectorV1Dynamic","kv_role":"kv_both","kv_connector_module_path":"lmcache_ascend.integration.vllm.lmcache_ascend_connector_v1"}' \
  2>&1 | tee "${LOG}"
```

Do not add `--speculative-config`; this POC rejects MTP. Also omit optional
MLAPO, DSA context-parallel, sparse-C8-indexer, weight-prefetch, and
free-paged-offload settings during this isolation run.

## 2. Drive correctness and steady-state profiling

Copy the printed `LOG` and `PROFILE_DIR` values into terminal B, then run from
the same vLLM Ascend checkout used by the server:

```bash
python3 tools/staged_sfa_graph_smoke.py \
  --base-url http://127.0.0.1:9000 \
  --model /workspace/models/GLM-5.1-w4a8 \
  --server-log "${LOG}" \
  --profile-dir "${PROFILE_DIR}"
```

The client sends one 4096-word prompt so its tokenized prompt should exceed
`index_topk`. It profiles streamed chunks 5 through 16; the two numerical
parity steps should therefore have completed before collection. Increase
`--prompt-words` if the server reports that the prompt boundary is smaller than
`index_topk`.

The automated gate requires all of the following:

1. every local staged SFA implementation has both captured entries and fixed,
   strongly owned Graph-A handoff buffers;
2. one size-one full-model replay, run only after all captures finish, writes
   the dedicated pre/post canary for every local SFA layer;
3. always-on captured-input signature validation (pointer, shape, stride,
   storage offset, dtype, and device) is enabled for live replay;
4. eager and graph results pass at two distinct sequence lengths on all TP
   ranks;
5. at least eight new worker/rank traces contain `pre`, `lmcache_retrieve`, and
   `post`, plus an ACL model-replay API; extra coordinator traces without those
   ranges are reported and ignored;
6. the steady-state trace does not contain the eager live-parity ranges.

The canary smoke proves that ordered replay reached the terminal operation in
each staged graph without modifying production activations. It does not prove
that every intended compute kernel was captured or is input-sensitive. These
gates are necessary, but trace-name/API presence alone is not a capture proof.

## 3. Complete the hardware proof in MindStudio Insight

Open one newly generated `trace_view.json` in MindStudio Insight (or the
equivalent parsed `msprof` timeline). On one rank and one steady decode step,
inspect each SFA layer in CPU and NPU timelines:

1. `sfa_staged_graph_poc::pre` contains one `aclmdlRIExecuteAsync` (or the
   equivalent ACL model-execute API for that CANN version), rather than a list
   of eager PyTorch operator launches;
2. `sfa_staged_graph_poc::lmcache_retrieve` is outside both graph ranges;
3. `sfa_staged_graph_poc::post` contains a second ACL graph replay;
4. the first replay drives the indexer/top-k/remap compute island, LMCache work
   occurs between the islands, and the second replay drives sparse
   attention/projection;
5. `pre` and `post` each occur approximately once per local SFA layer per
   decoded token, with no `live_parity_pre` or `live_parity_post` in the
   measured interval.

Seeing only many `aclmdlRIExecuteAsync` calls is insufficient. The required
evidence is their nesting in the two named ranges and the eager LMCache range
between them.

The online PyTorch profiler is preferred for this smoke because
`/start_profile` and `/stop_profile` fan out to all eight workers and retain the
`record_function` names above. Do not run it concurrently with a direct
`msprof` session. If direct `msprof` attachment is required for a CANN-specific
investigation, set `PROFILING_MODE=dynamic` before launching the server and
attach to one container-visible TP worker PID/card at a time; direct attachment
does not replace the named-range proof.

## Failure triage

| Symptom | First check |
| --- | --- |
| No staged-SFA startup messages | Confirm the feature flag, `UNBUNDLE=1`, `TWO_GROUPS=1`, `SHRINK_LATENT=2`, `PIECEWISE`, an MLA model with `index_topk`, and a KV connector. |
| Dummy pass is ineligible | The exception gives the incompatible feature or missing fixed-shape metadata; remove that feature for this POC run. |
| Startup capture/replay-canary completeness fails | Inspect the reported layer and missing `pre`/`post` entry, capture phase, persistent output binding, or replay-canary failure. |
| Captured input-signature check fails | A live positional tensor's storage, shape, stride, offset, dtype, or device differs from capture; inspect the reported pre/post input index. |
| `pre.*` parity fails | Suspect dynamic-length indexer/top-k replay, remap, or captured cache writes. |
| `post.output` parity fails | Suspect LMCache scratch visibility/stream ordering or dynamic-length sparse-attention replay. |
| TP checked count differs | At least one local layer/rank fell back; find the nearby `using the existing forward` reason. |
| Trace has live-parity ranges | Start profiling later with `--profile-after-chunks 6` or more. |
| Trace has named ranges but no ACL replay API | Verify the CANN API name and inspect whether `ACLGraphWrapper` replayed or eager code ran inside the range. |
| Large gaps remain between graph ranges | Attribute the gap inside `lmcache_retrieve`; a single compute stream is normal, but host dispatch inside `pre` or `post` is not. |
