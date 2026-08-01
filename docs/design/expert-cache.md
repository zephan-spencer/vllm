# Host-Backed W4A16 Expert Cache

## Target

A vLLM serving feature for routed W4A16 MoE weights:

- full routed-expert weights live in host memory;
- each MoE layer owns a bounded VRAM cache of routed experts;
- decode traffic controls cache admission and replacement;
- prefill traffic reads resident experts and streams the remaining experts;
- tensor-parallel ranks share one logical residency map.

Reference deployment: two AMD Radeon AI PRO R9700 GPUs with TP=2.

Reference models:

- production: DeepSeek-V4-Flash W4A16;
- control: Qwen3.6-35B-A3B W4A16.

## Usage

```bash
# All routed experts resident in VRAM.
vllm serve MODEL

# Routed experts in host memory; every routed expert streams to the GPU.
vllm serve MODEL --expert-cache-capacity 0

# Routed experts in host memory; 64 logical expert slots per MoE layer.
vllm serve MODEL --expert-cache-capacity 64
```

```bash
vllm serve canada-quant/DeepSeek-V4-Flash-W4A16-FP8-MTP \
    --tensor-parallel-size 2 \
    --expert-cache-capacity 64
```

`--expert-cache-capacity` accepts a non-negative integer. Its unit is logical
routed experts per MoE layer.

The server prints host-expert memory, cache memory per GPU, dense-model memory,
and available KV-cache memory during startup.

## Supported checkpoint contract

- Hugging Face Safetensors with `compressed-tensors` W4A16 routed experts.
- Packed expert weights, group scales, and zero points.
- BF16 or FP16 activations.
- Fixed expert count and top-k per MoE layer.
- TP-compatible expert tensor shapes.
- vLLM model support with routed expert IDs.

The cache preserves the checkpoint's W4A16 values and quantization parameters.

## Memory hierarchy

```text
checkpoint storage
        ↓
host-resident routed experts
        ↓
per-layer VRAM expert cache
        ↓
GPU hardware caches and compute
```

Dense weights, attention weights, routers, shared experts, embeddings, and
output heads use normal vLLM placement.

## Cache semantics

Each layer has an independent LRU cache.

Decode routes own cache policy:

- resident experts produce cache hits and update recency;
- missing experts fill cache slots;
- full caches evict the least-recently-used entries.

Prefill routes have read-only cache access:

- resident experts execute from VRAM;
- nonresident experts execute from host-backed streaming storage;
- prefill leaves admission, eviction, and recency unchanged.

An expert selected by decode and prefill follows decode policy. Every matching
row executes from the resulting resident slot.

Capacity overflow preserves correctness. Overflow expert work executes through
the streaming path and contributes to cache-overflow metrics.

## Mixed batches

One vLLM forward contains decode and prefill rows together.

For every MoE layer:

1. The router selects experts for every row.
2. Decode routes establish the layer's resident set.
3. Resident experts process matching decode and prefill rows from VRAM.
4. Nonresident prefill experts process matching rows from host-backed storage.

The resulting output matches all-resident execution within the normal numerical
tolerance of the selected W4A16 kernel.

## Tensor parallelism

A logical cache entry contains one local expert shard on every TP rank.

- Every rank uses the same logical expert-to-slot map.
- Every rank fills its local shard.
- A logical entry becomes available after all rank-local fills complete.
- Eviction invalidates every rank-local shard.
- Cache metrics include rank-local transfer and wait time.

## Observability

The server exports per-model, per-layer, and per-rank values for:

- accesses, hits, misses, fills, and evictions;
- resident entries and capacity overflow;
- host-to-GPU bytes and streamed-expert bytes;
- fill time, transfer wait time, MoE time, and TP collective time;
- decode and prefill rows served from VRAM and host storage.

## Required behavior

- Qwen W4A16 supports all-resident, fully streamed, and cached runs through
  the same W4A16 execution path.
- DeepSeek-V4-Flash W4A16 serves through bounded expert caches on two R9700s.
- Cache capacity remains a runtime setting.
- Cold and warm caches produce numerically equivalent outputs.
- Mixed batches preserve decode-controlled cache residency.
- Request arrival, completion, preemption, and batch reshaping preserve cache
  correctness.
