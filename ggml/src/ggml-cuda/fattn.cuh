#include "common.cuh"

void ggml_cuda_flash_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

bool ggml_cuda_flash_attn_ext_supported(int device, const ggml_tensor * dst);

size_t ggml_cuda_flash_attn_ext_get_alloc_size(int device, const ggml_tensor * dst);

// Per-device epoch counting address moves of fattn.cu's persistent scratch buffers (the
// Q-rotation buffer and the K/V f16 dequant scratch). These buffers are internal to the
// flash-attention dispatch — never a graph node's src[] — so a captured CUDA graph bakes their
// addresses into its recorded launches, and the node-property diff in
// ggml_cuda_graph_update_required cannot see them move. The graph runtime snapshots this epoch
// per captured graph and forces a recapture when it changes (ggml_cuda_graph in common.cuh,
// graph_update_required in ggml-cuda.cu).
unsigned long long ggml_cuda_fattn_scratch_epoch(int device);
