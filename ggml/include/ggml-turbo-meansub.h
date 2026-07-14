#pragma once

// Baked per-architecture KV affine-tap means (computed mu = sum/cnt, frozen from offline PFH1
// calibration). Lives in the runtime/codec, NOT in the GGUF: one table per base architecture
// transfers across finetunes and weight-quant tiers. The model loader records the active model
// identity once; the CUDA encode path and the graph decode path pull the matching baked slab when
// no TURBO_KMEAN_SUB / TURBO_VMEAN_SUB env override is set.

#include <stdint.h>
#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

// Record the active model identity (call once at load). arch = general.architecture string.
// The match requires arch + n_layer + n_embd to agree, so a different model (e.g. the MoE
// "qwen35moe", or any other geometry) never receives the wrong baked means.
GGML_API void ggml_turbo_meansub_set_model(const char * arch, int n_layer, int n_embd);

// Dense baked mean slab [max_l * max_c] for the active model; kvsel: 0 = K, 1 = V.
// Returns NULL when no baked table matches the active model. out_* may be NULL.
GGML_API const float * ggml_turbo_meansub_active(int kvsel, int * out_max_l, int * out_max_c, int * out_live);

#ifdef __cplusplus
}
#endif
