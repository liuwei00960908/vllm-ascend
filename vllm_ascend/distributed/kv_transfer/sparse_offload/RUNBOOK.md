# DSA Latent Offload — NPU Bring-up Runbook (single card, GLM-5.1-w4a8)

Tee每次 `vllm serve` 的输出到日志文件，跑完用文末的 grep 命令筛出需要的信息。
把 `<GLM5.1-w4a8>` / `<model名>` / `<你的单卡参数>` 替换成你的实际值。

> 注意：当前用 in-memory 参考 backend（latent 暂存 NPU 显存），**这版不省显存**，
> Step 2/3 只看 **parity diff + 输出一致性**，别看显存。真正省显存要等 LMCache backend 接入。

---

## 0. 拉最新代码

```bash
cd /workspace/sqh/vllm-ascend && git fetch --all && git reset --hard <你的remote>/sparse
git log --oneline -1
```

---

## Step 0 — 基线（开关全关，存对照输出）

```bash
vllm serve <GLM5.1-w4a8> <你的单卡参数> 2>&1 | tee /tmp/dsa_base.log &
# 起来后发固定请求并保存输出：
curl -s http://localhost:8000/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"<model名>","prompt":"Explain what a transformer is in two sentences.","max_tokens":32,"temperature":0}' \
  | tee /tmp/dsa_base_out.json
# 然后停掉服务
```

带回：`/tmp/dsa_base_out.json`（基线输出，用于 Step 3 对拍）。

---

## Step 2 — 开 offload + 双路对拍（最关键）

```bash
VLLM_ASCEND_ENABLE_DSA_LATENT_OFFLOAD=1 \
VLLM_ASCEND_DSA_OFFLOAD_ASSERT_PARITY=1 \
  vllm serve <GLM5.1-w4a8> <你的单卡参数> 2>&1 | tee /tmp/dsa_parity.log &
# 发同样的请求（触发 decode）：
curl -s http://localhost:8000/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"<model名>","prompt":"Explain what a transformer is in two sentences.","max_tokens":32,"temperature":0}' \
  | tee /tmp/dsa_parity_out.json
# 停服务
```

跑完筛日志：

```bash
# (1) 路径是否启用 + 预留显存 + 层数
grep -E "DSA latent offload enabled|Reserved.*DSA|inactive on this path" /tmp/dsa_parity.log

# (2) 对拍 diff —— 最大的几个（核心指标）+ 总条数
grep "DSA-PARITY" /tmp/dsa_parity.log | awk -F'max_abs_diff=' '{print $2}' | sort -g | tail -5
grep -c "DSA-PARITY" /tmp/dsa_parity.log

# (3) 有没有报错
grep -iE "error|traceback|runtimeerror|assert|exception" /tmp/dsa_parity.log | head -40
```

带回：(1)(2)(3) 的输出，尤其 (2) 的 diff 值；(3) 有栈就带完整栈段落。

---

## Step 3 — 真正走 offload（仅当 Step 2 的 diff ≈ 0 后）

```bash
VLLM_ASCEND_ENABLE_DSA_LATENT_OFFLOAD=1 \
  vllm serve <GLM5.1-w4a8> <你的单卡参数> 2>&1 | tee /tmp/dsa_real.log &
curl -s http://localhost:8000/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"<model名>","prompt":"Explain what a transformer is in two sentences.","max_tokens":32,"temperature":0}' \
  | tee /tmp/dsa_real_out.json
# 停服务
```

筛选 + 对拍：

```bash
grep -E "DSA latent offload enabled|inactive on this path" /tmp/dsa_real.log
grep -iE "error|traceback" /tmp/dsa_real.log | head -40
diff <(jq -r '.choices[0].text' /tmp/dsa_base_out.json) <(jq -r '.choices[0].text' /tmp/dsa_real_out.json) \
  && echo "✅ 输出与基线一致" || echo "❌ 输出有差异"
```

带回：`diff` 的结论（一致/不一致）+ error 筛选结果。

---

## 给开发侧的回执（按优先级）

1. Step 2 (2) 的 **max_abs_diff** 值（判断方案对错的那一锤）
2. Step 2 (1) 路径是否启用、(3) 报错栈（若有）
3. Step 3 输出与基线是否一致

## 判读速查

- 有 `[DSA-PARITY] ... max_abs_diff=` → 路径走到了；diff ≈ 0（如 <1e-2）= 通过，明显偏大 = 需修。
- 只有 `[DSA] latent offload enabled but inactive on this path (...)` → 没走到，括号里是原因。
- 两者都没有 → 检查 `VLLM_ASCEND_ENABLE_DSA_LATENT_OFFLOAD=1`，以及启动日志的
  `DSA latent offload enabled for N MLA layers`。

## 可选开关

- `VLLM_ASCEND_DSA_OFFLOAD_INTROSPECT=1`（+ `VLLM_ASCEND_DSA_INTROSPECT_FILE=/tmp/x.log`）：
  只读 ground-truth dump（Round 1 已用过）。
- `VLLM_ASCEND_DSA_OFFLOAD_BACKEND_DEVICE=cpu`：mock backend 把 latent 存主存（模拟离 NPU）。
- `VLLM_ASCEND_DSA_OFFLOAD_FREE_PAGED=1`：Stage 2（拆 spec 真正省显存，需后续工作，暂勿用）。

> 注：decode 生成的 token 不再有常驻上限——它们的 latent 留在 paged cache（vLLM 管理），
> 被 indexer 选中时从 paged 读回。原 `VLLM_ASCEND_DSA_MAX_RESIDENT_DECODE_TOKENS` 已移除。
