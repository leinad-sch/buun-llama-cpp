#pragma once

#include "common.cuh"

// Dynamic VBR transcode — Stage 1 (read side).
// Dequant the first `n_cells` rows of a turbo-quantized KV tensor to ORIGINAL-domain f32,
// so the set_rows re-encoder's forward FWHT is correct. MUST be defined in fattn.cu (the decode
// __constant__ codebooks/InnerQ/alpha are static to that translation unit).
//
//   src       : pointer to the source tensor data (turbo type `src_type`)
//   scratch_f16: caller-provided device scratch, >= n_cells*ne0 halfs
//   dst_f32   : caller-provided device output, >= n_cells*ne0 floats (dense [n_cells, ne0])
//   ne0       : per-cell embedding dim (n_embd_gqa, multiple of 128)
//   nb1       : byte stride between cells in src (source tensor nb[1])
//   is_v      : V-side decode (selects V codebook / alpha for TCQ); ignored for non-TCQ
//   device    : CUDA device (for per-device codebook/alpha staging)
void vbr_dequant_turbo_to_f32(const char * src, enum ggml_type src_type, enum ggml_type type_B,
                              half * scratch_f16, float * dst_f32,
                              int64_t n_cells, int64_t ne0, size_t nb1,
                              bool is_v, int device, cudaStream_t stream);

// S5 side-stream fence, consume side (defined in vbr-transcode.cu): if a VBR degrade wave armed
// the per-device fence (ggml_backend_cuda_vbr_fence_arm), make `stream` GPU-wait on it and disarm.
// Called at the top of every CUDA graph_compute so the next decode graph orders after the wave's
// async transcodes without a host block. No-op when unarmed (one branch on a static bool).
void ggml_cuda_vbr_fence_consume(int device, cudaStream_t stream);

// q_rot_buf reallocation epoch (defined in fattn.cu): k_turbo_fwht_forward's Q-rotation scratch
// (q_rot_buf[device] in fattn.cu) is a grow-only cudaMalloc'd buffer, NOT one of the flash-attn
// node's declared src[] tensors — its address is invisible to ggml_cuda_graph_update_required's
// node-property diff. A captured CUDA graph bakes in whatever address q_rot_buf[device] held at
// capture time; if a later call cudaFree+cudaMalloc's the buffer to a new address (a Q-shape grow
// on that device), every replay of the stale-captured graph still launches the FWHT kernel against
// the OLD address, now dangling/reused -> invalid write (seen: reset-triggered re-prefill growing Q
// right as a shared-KV drafter's differently-shaped Q first engages on the same device). Bump this
// once per (re)allocation; ggml-cuda.cu snapshots it per-graph and forces a recapture on mismatch,
// exactly like a node-property change.
unsigned long long ggml_cuda_q_rot_buf_epoch(int device);
