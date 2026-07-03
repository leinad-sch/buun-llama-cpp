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
