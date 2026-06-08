# DSA Latent Offload — NPU Bring-up Runbook (single card, GLM-5.1-w4a8)

Tee 每次 `vllm serve` 的输出到日志文件，跑完用各步末尾的 grep 命令筛信息。
把 `<GLM5.1-w4a8>` / `<model名>` / `<你的单卡参数>` 换成实际值。

## 进度 / Status

| 阶段 | 内容 | 状态 |
|---|---|---|
| Stage 1 | prefill 存 LMCache(mock) + decode gather 进 scratch + A1 重映射 + kernel 重定向 | ✅ NPU 验证 parity=0 |
| Stage2-B 池 | decode latent 存进**按需增长的独立池**(无长度上限);decode 选中项从池读 | ✅ 已接线，**待复跑 parity** |
| Stage2-B 省显存 | 释放 prefill latent(缩 paged latent spec + 改 exec_kv) | ⬜ 进行中(task 10) |
| LMCache 接入 | 用真 backend 替换 in-memory mock | ⬜ 等同事 |

**当前内存态**：仍是 Stage 1（paged latent 还在，**还没省显存**）。Stage2-B 的池只改了
decode latent 的存放位置，省显存要等 task 10 完成。

**现在要做的事 → 下面的 Step P（复跑 parity 验证池路径）。**

---

## 0. 拉最新代码

```bash
cd /workspace/sqh/vllm-ascend && git fetch --all && git reset --hard <你的remote>/sparse
git log --oneline -1
```

---

## Step P — 【当前要做】复跑 parity，验证 decode 池路径

我们把 decode latent 的来源从 paged 改成了独立增长池。先确认输出仍与原生稀疏一致。

```bash
VLLM_ASCEND_ENABLE_DSA_LATENT_OFFLOAD=1 \
VLLM_ASCEND_DSA_OFFLOAD_ASSERT_PARITY=1 \
  vllm serve <GLM5.1-w4a8> <你的单卡参数> 2>&1 | tee /tmp/dsa_pool.log &
curl -s http://localhost:8000/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"<model名>","prompt":"Explain what a transformer is in two sentences.","max_tokens":32,"temperature":0}'
# 停服务，然后筛日志：
grep -E "DSA latent offload enabled|Reserved.*DSA|inactive on this path" /tmp/dsa_pool.log
grep "DSA-PARITY" /tmp/dsa_pool.log | awk -F'max_abs_diff=' '{print $2}' | sort -g | tail -5
grep -c "DSA-PARITY" /tmp/dsa_pool.log
grep -iE "error|traceback|runtimeerror|exception" /tmp/dsa_pool.log | head -40
```

**带回**：`max_abs_diff` 最大几个（应仍为 0）、是否有报错。
- 全 0 + 无报错 → 池路径正确，进 task 10（省显存）。
- 非 0 / 报错 → 带回栈，开发侧修（多半是池的 rel-index 换算或存/取顺序）。

---

## Step M — 【task 10 完成后】验证省显存

task 10（释放 prefill latent）落地后，跑这步对比基线，看显存/可用 KV 是否变大。

先记下**基线**（offload 全关）启动日志里的两行：
```bash
vllm serve <GLM5.1-w4a8> <你的单卡参数> 2>&1 | tee /tmp/dsa_base.log
grep -E "Available KV cache memory|GPU KV cache size|Maximum concurrency" /tmp/dsa_base.log
# 基线参考：Available KV cache memory: 17.24 GiB / GPU KV cache size: 1,643,264 tokens
```
再开 offload（建议同时用 CPU mock 模拟离 NPU 存储）：
```bash
VLLM_ASCEND_ENABLE_DSA_LATENT_OFFLOAD=1 \
VLLM_ASCEND_DSA_OFFLOAD_BACKEND_DEVICE=cpu \
  vllm serve <GLM5.1-w4a8> <你的单卡参数> 2>&1 | tee /tmp/dsa_mem.log
grep -E "Available KV cache memory|GPU KV cache size|Maximum concurrency|Reserved.*DSA" /tmp/dsa_mem.log
```
**带回**：两次的 `Available KV cache memory` / `GPU KV cache size`。后者应明显**变大**
（每 token 的 paged 成本从 latent+indexer ≈704 降到只剩 indexer ≈128）。

---

## Step V — 输出一致性回归（任何阶段都可用）

```bash
# 基线输出（开关全关）
curl -s http://localhost:8000/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"<model名>","prompt":"Explain what a transformer is in two sentences.","max_tokens":32,"temperature":0}' \
  | tee /tmp/dsa_base_out.json
# offload 输出（不开 PARITY，真正走 scratch/池）
VLLM_ASCEND_ENABLE_DSA_LATENT_OFFLOAD=1 vllm serve ... ; curl ... | tee /tmp/dsa_real_out.json
diff <(jq -r '.choices[0].text' /tmp/dsa_base_out.json) <(jq -r '.choices[0].text' /tmp/dsa_real_out.json) \
  && echo "✅ SAME" || echo "❌ DIFF"
```

---

## 判读速查

- `[DSA-PARITY] ... max_abs_diff=` → offload 路径走到了；≈0 = 通过，明显偏大 = 需修。
- `[DSA] latent offload enabled but inactive on this path (...)` → 没走到，括号里是原因
  （CP / sparse_c8 / mlapo）。
- 两者都没有 → 检查 `VLLM_ASCEND_ENABLE_DSA_LATENT_OFFLOAD=1` 和启动日志
  `DSA latent offload enabled for N MLA layers`。

## 开关速查

- `VLLM_ASCEND_ENABLE_DSA_LATENT_OFFLOAD=1`：总开关。
- `VLLM_ASCEND_DSA_OFFLOAD_ASSERT_PARITY=1`：bring-up 对拍（双路跑、打 diff、用原生结果驱动生成）。
- `VLLM_ASCEND_DSA_OFFLOAD_BACKEND_DEVICE=cpu`：mock backend 把 latent 存主存（模拟离 NPU）。
- `VLLM_ASCEND_DSA_OFFLOAD_INTROSPECT=1`（+ `..._INTROSPECT_FILE=/tmp/x.log`）：只读 ground-truth dump。
- `VLLM_ASCEND_DSA_OFFLOAD_FREE_PAGED=1`：预留给 task 10（缩 paged latent），暂未启用。

> 说明：decode 生成的 token 现在存在独立增长池里，**无长度上限**；不再有
> `VLLM_ASCEND_DSA_MAX_RESIDENT_DECODE_TOKENS`（已移除）。
