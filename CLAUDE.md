# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

Fork of llama.cpp (`spiritbuun/buun-llama-cpp`) adding KV-cache compression and speculative
decoding on top of mainline. Everything here is CUDA-first and mirrored to ROCm/HIP.

- **TurboQuant**: custom KV-cache quantization types for aggressive context compression, decoded
  fused inside flash-attention (no intermediate f16 buffer) on Turing+ / RDNA.
  - *Vanilla* (FWHT rotation + centroid LUT, per-element parallel): `turbo8_0` (8.125 bpv),
    `turbo4_0` (4.125), `turbo3_0` (3.5), `turbo2_0` (2.5).
  - *TCQ* (trellis-coded quantization, Viterbi encode, **separate K/V codebooks**): `turbo3_tcq`
    (3.25 bpv), `turbo2_tcq` (2.25), `turbo1_tcq` (1.25 — lowest-bit KV codec in the fork).
  - *Affine tap*: per-(layer, head/channel) mean-subtraction at encode (baked calibration in
    `ggml-turbo-meansub-data.inc`), K decode-free, V added back in-graph. Default-on for qwen35.
- **VBR (Variable Bit-Rate KV cache)**: dynamic, decode-time KV tiering. A price-ordered ladder
  `F16 → T8 → T4 → T3_TCQ → T2_TCQ → T1_TCQ` degrades one (layer, side) slot at a time under memory
  pressure, so context length trades against fidelity automatically. Enable with `-ctk vbr`.
- **DFlash**: cross-attention drafter (DeltaNet-based). Captures target hidden states via eval
  callback and feeds them to a small drafter through cross-attention layers. Arch-agnostic
  registration (keys off `dflash_block_size > 0`): Qwen3.5/3.6 (`dflash_draft.cpp`) and
  Gemma-4 (`gemma4_dflash_draft.cpp`).
- **Gemma-4 MTP**: multi-token-prediction assistant drafter for the Gemma-4 family.
- **DDTree**: tree-based speculative decoding with SSM tree kernels (parent_ids on GPU).
- **CopySpec**: suffix/copy-based speculation (model-free).

## Build

CUDA (RTX 3090 / sm_86 shown; set `CMAKE_CUDA_ARCHITECTURES` to your GPU):

```bash
cmake -B build -DGGML_CUDA=ON \
  -DCMAKE_CUDA_COMPILER=/opt/cuda/bin/nvcc -DCUDAToolkit_ROOT=/opt/cuda \
  -DGGML_CUDA_FA=ON -DGGML_CUDA_FA_ALL_QUANTS=ON \
  -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

ROCm/HIP (RDNA; VBR VMM needs `-DGGML_HIP_NO_VMM=OFF`):

```bash
cmake -B build -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1201 \
  -DGGML_HIP_NO_VMM=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

Key binaries: `build/bin/llama-server`, `build/bin/llama-cli`, `build/bin/llama-bench`,
`build/bin/llama-perplexity`. Each is a separate static-linked target — rebuilding one does not
relink the others.

## Architecture

### Layers
- `ggml/` — tensor library, CUDA/HIP/CPU backends, quantization
- `src/` — llama.cpp core: model loading, context, graph building, sampling
- `src/models/` — per-architecture graph builders (one `.cpp` per model family)
- `common/` — shared utilities, speculative-decoding orchestration
- `tools/server/` — HTTP server (slots, chat-completions API)
- `include/llama.h` — public C API

### TurboQuant / VBR files
- `ggml/src/ggml-turbo-quant.c` — CPU reference codec registration + type traits
- `ggml/src/ggml-cuda/turbo-quant-cuda.cuh` — device dequant/encode kernels, codebooks, FWHT
- `ggml/src/ggml-cuda/fattn-mma-turbo.cuh` + `fattn.cu` — fused turbo/TCQ flash-attention decode
- `ggml/src/ggml-cuda/set-rows.cu` — KV encode (quantize into cache), incl. Viterbi for TCQ
- `ggml/include/ggml-vbr.h` — backend vtable (`ggml_vbr_backend_iface`), runtime-resolved per backend
- `ggml/src/ggml-cuda/vbr-transcode.cu`, `vbr-vmm.cu` — VBR tier transcode + VMM pool (CUDA/HIP)
- `src/llama-kv-cache.cpp` — VBR controller (tier enum, budget/watermark, degrade cursor, pools)
- `src/llama-vbr-degrade-orders.inc` — baked price-ordered degrade tables, keyed by (arch, n_layer)

### DFlash-specific files
- `src/models/dflash_draft.cpp` — Qwen drafter graph builder (cross-attention + DeltaNet recurrent)
- `src/models/gemma4_dflash_draft.cpp` — Gemma-4 drafter (same KV-injection graph, gemma4 grafts)
- `src/models/qwen35.cpp` — target model (hybrid attention + recurrent layers)
- `common/speculative.cpp` — `common_speculative_state_dflash`: ring buffer, draft/verify loop
- `ggml/src/ggml-cuda/cross-ring-interleave.cu` — GPU ring buffer + interleave kernel
- `ggml/src/ggml-cuda/gated-delta-net.cu` — DeltaNet forward (flat + tree variants)
- `ggml/src/ggml-cuda/ssm-conv.cu` — SSM conv1d (flat + tree variants)

### Key patterns
- **proc_address**: custom CUDA/HIP entry points (VBR vtable, GPU ring, tree kernels) are registered
  via `ggml_backend_*_reg_get_proc_address` and resolved at runtime, not linked directly. A backend
  that does not export the VBR proc refuses turbo KV (CPU-bound layers fall back to q8_0).
- **Fused KV decode**: turbo/TCQ K and V are decoded inside the flash-attention tile loader; the
  unfused (`ggml_mul_mat`) attention path does not support turbo types.
- **VBR requires unified KV + flash-attention** (`n_stream == 1`, `!v_trans`); server parallel
  sequences force `--kv-unified`. Multi-GPU uses one VBR pool per device buffer under a single
  device-global degrade cursor.
- **Eval callback / ring buffer** (DFlash): target hidden states captured during the forward pass
  via `llama_set_eval_callback`, stored in a CPU ring buffer mirrored to a GPU ring; cross-attention
  input is built by interleaving layers over the context window.

## Test / Benchmark

```bash
# Perplexity (PPL)
build/bin/llama-perplexity -m model.gguf -f test.txt -c 4096

# Decode speed
build/bin/llama-bench -m model.gguf -p 0 -n 64 -t 1            # tg64

# Turbo KV cache
build/bin/llama-cli -m model.gguf -ctk turbo3_tcq -ctv turbo3_tcq -fa on -p "..."

# Dynamic VBR KV cache
build/bin/llama-cli -m model.gguf -ctk vbr -fa on -p "..."

# Server with a DFlash drafter
build/bin/llama-server -m target.gguf --draft-model drafter.gguf \
  --spec-dflash-default --draft-max 12 -ngl 99 -c 4096 --port 8080
```

Grep `"Final estimate"` (PPL) and `"tg64"` (decode speed). Run with `-v` to see live VBR degrade
lines.

## Multi-GPU Notes
- VBR: one pool per device buffer; degrade price order is device-global, each step resolves its
  tensor's owning pool. Single-GPU reduces bit-for-bit to the single-pool path.
- DFlash tree verify (`parent_ids_gpu`) lives on GPU 0 only; auto-disabled when `n_devices() > 1`
  (recurrent layers can't read cross-device).
- Uses `cudaStreamPerThread` throughout — each host thread gets its own per-device stream.

## Git Conventions
- Branch per experiment; keep experiment branches as archives (negative results are records too).
- Only merged, verified, clean code on the release trunk.
- Commit/PR titles: `<module> : <title>`.
