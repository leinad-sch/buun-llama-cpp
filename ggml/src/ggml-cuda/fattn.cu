#include "common.cuh"
#include "fattn-common.cuh"
#include "fattn-mma-f16.cuh"
#include "fattn-mma-turbo.cuh"
#include "fattn-tile.cuh"
#include "fattn-vec.cuh"
#include "fattn-wmma-f16.cuh"
#include "fattn.cuh"

#include <atomic>
#include <sys/stat.h>
#include <vector>

#include "turbo1_cq_codebook.h"   // macro TURBO1_CQ_CB_INIT (256x8 codebook)
static __constant__ float d_turbo1_cq_cb_fattn[256*8] = TURBO1_CQ_CB_INIT;

// turbo1_nsn per-chunk per-head centering buffer (defined in set-rows.cu) + table dims.
extern float * turbo1_nsn_o_buf(int device, int is_k);
#ifndef PFHEAD_MAX_L
#define PFHEAD_MAX_L 128
#define PFHEAD_MAX_C 2048
#endif

// E6 subtractive dither (decode side). MUST match set-rows.cu encode exactly (same TPDF, same
// block-address seed). Constants set once from TURBO1_DITHER env.
static __constant__ int   d_turbo1_dither_fattn;
static __constant__ float d_turbo1_dither_w_fattn;
static __device__ __forceinline__ float turbo1_tpdf_fattn(unsigned long long seed, int j, float w) {
    unsigned long long s = seed + (unsigned long long)(j + 1) * 0x9E3779B97F4A7C15ULL;
    s = (s ^ (s >> 30)) * 0xBF58476D1CE4E5B9ULL;
    s = (s ^ (s >> 27)) * 0x94D049BB133111EBULL;
    s =  s ^ (s >> 31);
    float u1 = (float)(unsigned int)(s & 0xFFFFFFFFu) * (1.0f / 4294967296.0f);
    float u2 = (float)(unsigned int)(s >> 32)         * (1.0f / 4294967296.0f);
    return (u1 + u2 - 1.0f) * w;
}
static void turbo1_load_dither_fattn() {
    static bool done = false;
    if (done) return;
    done = true;
    const char * dz = getenv("TURBO1_DITHER");
    float dw = dz ? (float)atof(dz) : 0.0f;
    int don = dw > 0.0f ? 1 : 0;
    cudaMemcpyToSymbol(d_turbo1_dither_fattn, &don, sizeof(int));
    cudaMemcpyToSymbol(d_turbo1_dither_w_fattn, &dw, sizeof(float));
}

// InnerQ: update the fattn-side inverse scale array from host (all devices)
void turbo_innerq_update_fattn_scales(const float * scale_inv) {
    int cur_device;
    cudaGetDevice(&cur_device);
    int device_count;
    cudaGetDeviceCount(&device_count);
    for (int id = 0; id < device_count; id++) {
        cudaSetDevice(id);
        cudaMemcpyToSymbol(d_innerq_channel_scale_inv_fattn, scale_inv, 128 * sizeof(float));
    }
    cudaSetDevice(cur_device);
}

void turbo_innerq_init_fattn() {
    float ones[128];
    for (int i = 0; i < 128; i++) ones[i] = 1.0f;
    int cur_device;
    cudaGetDevice(&cur_device);
    int device_count;
    cudaGetDeviceCount(&device_count);
    for (int id = 0; id < device_count; id++) {
        cudaSetDevice(id);
        cudaMemcpyToSymbol(d_innerq_channel_scale_inv_fattn, ones, sizeof(ones));
    }
    cudaSetDevice(cur_device);
}

// Q² calibration: host-side management
static int q_calibrate_state = 0; // 0=off, 1=collecting, 2=done

void turbo_q_calibrate_init() {
    const char * env = getenv("TURBO_Q_CALIBRATE");
    if (!env || atoi(env) != 1) return;

    double zeros[128] = {};
    int zero = 0, one = 1;
    int cur_device;
    cudaGetDevice(&cur_device);
    int device_count;
    cudaGetDeviceCount(&device_count);
    for (int id = 0; id < device_count; id++) {
        cudaSetDevice(id);
        cudaMemcpyToSymbol(d_q_channel_sq_fattn, zeros, sizeof(zeros));
        cudaMemcpyToSymbol(d_q_channel_count_fattn, &zero, sizeof(zero));
        cudaMemcpyToSymbol(d_q_calibrate_fattn, &one, sizeof(one));
    }
    cudaSetDevice(cur_device);
    q_calibrate_state = 1;
    fprintf(stderr, "TURBO_Q_CALIBRATE: collecting per-position Q² statistics\n");
}

void turbo_q_calibrate_finalize() {
    if (q_calibrate_state != 1) return;

    int cur_device;
    cudaGetDevice(&cur_device);
    int device_count;
    cudaGetDeviceCount(&device_count);

    int zero = 0;
    for (int id = 0; id < device_count; id++) {
        cudaSetDevice(id);
        cudaMemcpyToSymbol(d_q_calibrate_fattn, &zero, sizeof(zero));
    }
    cudaSetDevice(cur_device);

    double sq[128];
    int count;
    cudaMemcpyFromSymbol(sq, d_q_channel_sq_fattn, sizeof(sq));
    cudaMemcpyFromSymbol(&count, d_q_channel_count_fattn, sizeof(count));

    if (count == 0) {
        fprintf(stderr, "TURBO_Q_CALIBRATE: no Q vectors seen, skipping\n");
        q_calibrate_state = 2;
        return;
    }

    // Compute E[Q²] per position and save as 128 float32 values
    float weights[128];
    fprintf(stderr, "TURBO_Q_CALIBRATE: %d Q groups accumulated\n", count);
    double total = 0;
    for (int i = 0; i < 128; i++) {
        weights[i] = (float)(sq[i] / count);
        total += weights[i];
    }
    float mean = (float)(total / 128.0);
    float maxw = 0, minw = 1e30f;
    for (int i = 0; i < 128; i++) {
        if (weights[i] > maxw) maxw = weights[i];
        if (weights[i] < minw) minw = weights[i];
    }
    fprintf(stderr, "  E[Q²] mean=%.6f min=%.6f max=%.6f ratio=%.2f\n",
            mean, minw, maxw, maxw / (minw > 1e-10f ? minw : 1e-10f));

    const char * path = "/tmp/q_weights.bin";
    FILE * fp = fopen(path, "wb");
    if (fp) {
        fwrite(weights, sizeof(float), 128, fp);
        fclose(fp);
        fprintf(stderr, "  Saved Q weights to %s (128 floats)\n", path);
    }

    q_calibrate_state = 2;
}

template <int DKQ, int DV, int ncols2>
static void ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const ggml_tensor * Q = dst->src[0];

    if constexpr (ncols2 <= 8) {
        if (turing_mma_available(cc) && Q->ne[1] <= 8/ncols2) {
            ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 8/ncols2, ncols2>(ctx, dst);
            return;
        }
    }

    if constexpr (ncols2 <= 16) {
        if (Q->ne[1] <= 16/ncols2) {
            ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 16/ncols2, ncols2>(ctx, dst);
            return;
        }
    }

    if (Q->ne[1] <= 32/ncols2 || (GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) == GGML_CUDA_CC_TURING) ||
            (GGML_CUDA_CC_IS_AMD(cc) && DKQ > 256)) {
        ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 32/ncols2, ncols2>(ctx, dst);
        return;
    }

    ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 64/ncols2, ncols2>(ctx, dst);
}

template <int DKQ, int DV>
static void ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const ggml_tensor * KQV  = dst;
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    float max_bias = 0.0f;
    memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

    // Edge cases like no mask, ALiBi, unpadded K/V, or misaligned addresses for large data transfers
    //     are put into the template specialization without GQA optimizations.
    bool use_gqa_opt = mask && max_bias == 0.0f && K->ne[1] % FATTN_KQ_STRIDE == 0;
    for (const ggml_tensor * t : {Q, K, V, mask}) {
        if (t == nullptr || ggml_is_quantized(t->type)) {
            continue;
        }
        for (size_t i = 1; i < GGML_MAX_DIMS; ++i) {
            if (t->nb[i] % 16 != 0) {
                use_gqa_opt = false;
                break;
            }
        }
    }

    GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
    const int gqa_ratio = Q->ne[2] / K->ne[2];

    // On Volta the GQA optimizations aren't as impactful vs. minimizing wasted compute:
    if (cc == GGML_CUDA_CC_VOLTA) {
        if (use_gqa_opt && gqa_ratio % 8 == 0) {
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 8>(ctx, dst);
            return;
        }

        if (use_gqa_opt && gqa_ratio % 4 == 0) {
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 4>(ctx, dst);
            return;
        }

        if constexpr (DKQ <= 256) {
            if (use_gqa_opt && gqa_ratio % 2 == 0) {
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 2>(ctx, dst);
                return;
            }

            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 1>(ctx, dst);
            return;
        } else {
            GGML_ABORT("fatal error");
        }
    }

    if (use_gqa_opt && gqa_ratio > 4) {
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 8>(ctx, dst);
        return;
    }

    if (use_gqa_opt && gqa_ratio > 2) {
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 4>(ctx, dst);
        return;
    }

    if constexpr (DKQ <= 256) {
        if (use_gqa_opt && gqa_ratio > 1) {
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 2>(ctx, dst);
            return;
        }

        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 1>(ctx, dst);
    } else {
        GGML_ABORT("fatal error");
    }
}

// Turbo MMA fused dispatch: ncols1 selection (mirrors f16 version but calls turbo case).
template <int DKQ, int DV, int ncols2, ggml_type type_K, ggml_type type_V>
static void ggml_cuda_flash_attn_ext_mma_turbo_switch_ncols1(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const ggml_tensor * Q = dst->src[0];

    if constexpr (ncols2 <= 8) {
        if (turing_mma_available(cc) && Q->ne[1] <= 8/ncols2) {
            ggml_cuda_flash_attn_ext_mma_turbo_case<DKQ, DV, 8/ncols2, ncols2, type_K, type_V>(ctx, dst);
            return;
        }
    }

    if (Q->ne[1] <= 16/ncols2) {
        ggml_cuda_flash_attn_ext_mma_turbo_case<DKQ, DV, 16/ncols2, ncols2, type_K, type_V>(ctx, dst);
        return;
    }

    // Turing (sm_75) is capped at ncols=32 — the kernel has NO_DEVICE_CODE for ncols>32.
    if (ggml_cuda_highest_compiled_arch(cc) == GGML_CUDA_CC_TURING || Q->ne[1] <= 32/ncols2) {
        ggml_cuda_flash_attn_ext_mma_turbo_case<DKQ, DV, 32/ncols2, ncols2, type_K, type_V>(ctx, dst);
        return;
    }

    ggml_cuda_flash_attn_ext_mma_turbo_case<DKQ, DV, 64/ncols2, ncols2, type_K, type_V>(ctx, dst);
}

// Turbo MMA fused dispatch: ncols2 selection based on GQA ratio.
template <int DKQ, int DV, ggml_type type_K, ggml_type type_V>
static void ggml_cuda_flash_attn_ext_mma_turbo_switch_ncols2(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * KQV  = dst;
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * mask = dst->src[3];

    float max_bias = 0.0f;
    memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

    bool use_gqa_opt = mask && max_bias == 0.0f && K->ne[1] % FATTN_KQ_STRIDE == 0;
    for (const ggml_tensor * t : {Q, K, mask}) {
        if (t == nullptr || ggml_is_quantized(t->type)) {
            continue;
        }
        for (size_t i = 1; i < GGML_MAX_DIMS; ++i) {
            if (t->nb[i] % 16 != 0) {
                use_gqa_opt = false;
                break;
            }
        }
    }

    GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
    const int gqa_ratio = Q->ne[2] / K->ne[2];

    if (use_gqa_opt && gqa_ratio > 4) {
        ggml_cuda_flash_attn_ext_mma_turbo_switch_ncols1<DKQ, DV, 8, type_K, type_V>(ctx, dst);
        return;
    }

    if (use_gqa_opt && gqa_ratio > 2) {
        ggml_cuda_flash_attn_ext_mma_turbo_switch_ncols1<DKQ, DV, 4, type_K, type_V>(ctx, dst);
        return;
    }

    if (use_gqa_opt && gqa_ratio > 1) {
        ggml_cuda_flash_attn_ext_mma_turbo_switch_ncols1<DKQ, DV, 2, type_K, type_V>(ctx, dst);
        return;
    }

    ggml_cuda_flash_attn_ext_mma_turbo_switch_ncols1<DKQ, DV, 1, type_K, type_V>(ctx, dst);
}

static void ggml_cuda_flash_attn_ext_mma_f16(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const ggml_tensor * KQV  = dst;
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    switch (Q->ne[0]) {
        case 64:
            GGML_ASSERT(V->ne[0] == 64);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2< 64,  64>(ctx, dst);
            break;
        case 80:
            GGML_ASSERT(V->ne[0] == 80);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2< 80,  80>(ctx, dst);
            break;
        case 96:
            GGML_ASSERT(V->ne[0] == 96);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2< 96,  96>(ctx, dst);
            break;
        case 112:
            GGML_ASSERT(V->ne[0] == 112);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<112, 112>(ctx, dst);
            break;
        case 128:
            GGML_ASSERT(V->ne[0] == 128);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<128, 128>(ctx, dst);
            break;
        case 192: {
            // MiMo-V2.5 / V2.5-Pro / V2-Flash: gqa_ratio is 8 (SWA) or 16 (full attn)
            GGML_ASSERT(V->ne[0] == 128);
            float max_bias = 0.0f;
            memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));
            const bool use_gqa_opt = mask && max_bias == 0.0f;
            GGML_ASSERT(use_gqa_opt);
            GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
            const int gqa_ratio = Q->ne[2] / K->ne[2];
            if (gqa_ratio % 16 == 0) {
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<192, 128, 16>(ctx, dst);
            } else {
                GGML_ASSERT(gqa_ratio % 8 == 0);
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<192, 128,  8>(ctx, dst);
            }
        } break;
        case 256:
            GGML_ASSERT(V->ne[0] == 256);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<256, 256>(ctx, dst);
            break;
        case 320:
            // For Mistral Small 4, go straight to the ncols1 switch (ncols2=32-only build).
            GGML_ASSERT(V->ne[0] == 256);
            {
                float max_bias = 0.0f;
                memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

                const bool use_gqa_opt = mask && max_bias == 0.0f;
                GGML_ASSERT(use_gqa_opt);
                GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
                const int gqa_ratio = Q->ne[2] / K->ne[2];
                GGML_ASSERT(gqa_ratio % 32 == 0);

                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<320, 256, 32>(ctx, dst);
            }
            break;
        case 512:
            GGML_ASSERT(V->ne[0] == 512);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<512, 512>(ctx, dst);
            break;
        case 576: {
            // For Deepseek, go straight to the ncols1 switch to avoid compiling unnecessary kernels.
            GGML_ASSERT(V->ne[0] == 512);
            float max_bias = 0.0f;
            memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

            const bool use_gqa_opt = mask && max_bias == 0.0f;
            GGML_ASSERT(use_gqa_opt);

            GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
            const int gqa_ratio = Q->ne[2] / K->ne[2];
            if (gqa_ratio == 20) { // GLM 4.7 Flash
                if (cc >= GGML_CUDA_CC_DGX_SPARK) {
                    if (Q->ne[1] <= 8) {
                        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
                        break;
                    }
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
                    break;
                }
                if (cc >= GGML_CUDA_CC_BLACKWELL) {
                    if (Q->ne[1] <= 4 && K->ne[1] >= 65536) {
                        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
                        break;
                    }
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
                    break;
                }
                if (cc >= GGML_CUDA_CC_ADA_LOVELACE) {
                    if (Q->ne[1] <= 4) {
                        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
                        break;
                    }
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
                    break;
                }
                if (cc >= GGML_CUDA_CC_TURING) {
                    if (Q->ne[1] <= 4) {
                        if (K->ne[1] <= 16384) {
                            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
                            break;
                        }
                        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 32>(ctx, dst);
                        break;
                    }
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
                    break;
                }
                // Volta:
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
            } else if (gqa_ratio % 16 == 0) {
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
            } else {
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512,  4>(ctx, dst);
            }
        } break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

// Context-adaptive V alpha: logarithmic scaling based on current KV occupancy.
// 3-bit: alpha = 1.1484 - 0.01443 * ln(n_kv), calibrated on Qwen3.5-27B decode-time KLD sweeps
//   at 8K/16K/32K. Optima: 8K→1.020, 16K→1.005, 32K→1.000.
// 2-bit: alpha = 0.8865 + 0.0195 * ln(n_kv), calibrated on fine-grained 0.005-step decode-time
//   KLD sweeps at 2K/7K/16K/32K on Qwen3.5-27B. Optima: 2K→1.030, 7K→1.065, 16K→1.090, 32K→1.075.
// Override with TURBO_TCQ_DECODE_ALPHA_V env var to force a static alpha (disables adaptive).
static float d_tcq_decode_alpha_v_static = 0.0f; // 0 = use adaptive, >0 = static override
static float d_tcq_decode_alpha_k = 1.0f;       // K decode alpha, static (default 1.0)
static bool d_tcq_decode_alpha_loaded = false;

static inline float tcq_compute_alpha_v(ggml_type v_type, int64_t n_kv) {
    if (d_tcq_decode_alpha_v_static > 0.0f) return d_tcq_decode_alpha_v_static;
    if (n_kv < 1) n_kv = 1;
    const float ln_ctx = logf((float)n_kv);
    (void) ln_ctx;
    if (v_type == GGML_TYPE_TURBO3_TCQ) {
        // Flat optimum for the coord-descent codebook (K=iter374/V=iter500). The retrained
        // codebook removed the depth-dependence — a flat alpha beats the old adaptive curve.
        return 1.02f;
    } else if (v_type == GGML_TYPE_TURBO2_TCQ) {
        // Flat optimum for the coord-descent codebook (K=iter195/V=iter208).
        return 1.06f;
    } else if (v_type == GGML_TYPE_TURBO1_TCQ) {
        // 1-bit wants a higher decode alpha than 2/3-bit (monotone in bits: t3=1.02, t2=1.06, t1=1.14).
        // Sweep optimum on the baked K=seed5/iter135, V=seed1/iter490 codebooks @8192/24ch:
        // V=1.14 → median KLD 0.0452 (−11% vs alpha=1.0's 0.0508). K alpha left at 1.0 (shared global);
        // K=1.02 buys a further −0.3% via TURBO_TCQ_DECODE_ALPHA_K=1.02 env if desired.
        return 1.14f;
    }
    return 1.0f;
}

static void load_tcq_decode_alpha(int device) {
    static bool loaded[GGML_CUDA_MAX_DEVICES] = {};
    if (loaded[device]) return;
    loaded[device] = true;
    if (!d_tcq_decode_alpha_loaded) {
        d_tcq_decode_alpha_loaded = true;
        const char * sk = getenv("TURBO_TCQ_DECODE_ALPHA_K");
        if (sk) {
            char * end;
            errno = 0;
            float a = strtof(sk, &end);
            if (end != sk && errno == 0 && a > 0.0f && a < 10.0f) {
                d_tcq_decode_alpha_k = a;
                fprintf(stderr, "TCQ decode: K alpha=%.4f\n", a);
            }
        }
        const char * s = getenv("TURBO_TCQ_DECODE_ALPHA_V");
        if (s) {
            char * end;
            errno = 0;
            float a = strtof(s, &end);
            if (end != s && errno == 0 && a > 0.0f && a < 10.0f) {
                d_tcq_decode_alpha_v_static = a;
            }
        }
    }
    if (d_tcq_decode_alpha_v_static > 0.0f) {
        cudaMemcpyToSymbol(d_tcq_decode_alpha_v_fattn, &d_tcq_decode_alpha_v_static, sizeof(float));
        if (device == 0) fprintf(stderr, "TCQ decode: V alpha=%.4f (static override)\n", d_tcq_decode_alpha_v_static);
    } else {
        if (device == 0) fprintf(stderr, "TCQ decode: context-adaptive V alpha enabled\n");
    }
}

// === Turbo prefill: bulk dequant to fp16 + MMA attention ===
// During prefill (Q->ne[1] > 1), dequantize turbo K/V to fp16 temp buffers
// and use the fast MMA kernel instead of the slower vec kernel.

static __global__ void k_turbo2_dequant_f16(
        const char * __restrict__ src, half * __restrict__ dst,
        const int64_t ne0, const int64_t ne1, const int64_t ne2,
        const size_t nb1, const size_t nb2, const size_t nb3) {
    const int64_t row  = blockIdx.x;
    const int64_t head = blockIdx.y;
    const int64_t strm = blockIdx.z;
    const int j = threadIdx.x;
    if (j >= ne0) return;

    const char * src_row = src + strm * nb3 + head * nb2 + row * nb1;
    const int blk_idx  = j / QK_TURBO2;
    const int j_in_blk = j % QK_TURBO2;
    const block_turbo2_0 * blk = (const block_turbo2_0 *)src_row + blk_idx;

    const float norm = __half2float(blk->norm);
    const uint8_t idx = (blk->qs[j_in_blk / 4] >> ((j_in_blk % 4) * 2)) & 0x3;
    const float val = d_turbo_centroids_2bit_fattn[idx] * norm;

    dst[strm * (ne1 * ne2 * ne0) + row * (ne2 * ne0) + head * ne0 + j] = __float2half(val);
}

static __global__ void k_turbo3_dequant_f16(
        const char * __restrict__ src, half * __restrict__ dst,
        const int64_t ne0, const int64_t ne1, const int64_t ne2,
        const size_t nb1, const size_t nb2, const size_t nb3) {
    const int64_t row  = blockIdx.x;
    const int64_t head = blockIdx.y;
    const int64_t strm = blockIdx.z;
    const int j = threadIdx.x;
    if (j >= ne0) return;

    const char * src_row = src + strm * nb3 + head * nb2 + row * nb1;
    const int blk_idx  = j / QK_TURBO3;
    const int j_in_blk = j % QK_TURBO3;
    const block_turbo3_0 * blk = (const block_turbo3_0 *)src_row + blk_idx;

    const float norm = __half2float(blk->norm);
    const uint8_t low2 = (blk->qs[j_in_blk / 4] >> ((j_in_blk % 4) * 2)) & 0x3;
    const uint8_t hi1  = (blk->signs[j_in_blk / 8] >> (j_in_blk % 8)) & 0x1;
    const float val = d_turbo_centroids_3bit_fattn[low2 | (hi1 << 2)] * norm;

    dst[strm * (ne1 * ne2 * ne0) + row * (ne2 * ne0) + head * ne0 + j] = __float2half(val);
}

static __global__ void k_turbo3_tcq_dequant_f16(
        const char * __restrict__ src, half * __restrict__ dst,
        const int64_t ne0, const int64_t ne1, const int64_t ne2,
        const size_t nb1, const size_t nb2, const size_t nb3,
        const float alpha, const int is_v) {
    const int64_t row  = blockIdx.x;
    const int64_t head = blockIdx.y;
    const int64_t strm = blockIdx.z;
    const int j = threadIdx.x; // element index within row (0..ne0-1)
    if (j >= ne0) return;

    const char * src_row = src + strm * nb3 + head * nb2 + row * nb1;
    const int blk_idx  = j / QK_TURBO3_TCQ;
    const int t = j % QK_TURBO3_TCQ; // element index within 128-element block
    const block_turbo3_tcq * blk = (const block_turbo3_tcq *)src_row + blk_idx;

    const float norm = __half2float(blk->norm) * alpha;

    // Sliding window decode: read 9-bit state from bitstream at bit offset t*3
    const int bit_pos = t * 3;
    const int byte_idx = bit_pos / 8;
    const int bit_off = bit_pos % 8;
    const uint16_t raw = (uint16_t)blk->qs[byte_idx] | ((uint16_t)blk->qs[byte_idx + 1] << 8);
    const int state = (raw >> bit_off) & 0x1FF;
    const float val = (is_v ? d_turbo3_tcq_codebook_v_fattn : d_turbo3_tcq_codebook_fattn)[state] * norm;

    dst[strm * (ne1 * ne2 * ne0) + row * (ne2 * ne0) + head * ne0 + j] = __float2half(val);
}

static __global__ void k_turbo2_tcq_dequant_f16(
        const char * __restrict__ src, half * __restrict__ dst,
        const int64_t ne0, const int64_t ne1, const int64_t ne2,
        const size_t nb1, const size_t nb2, const size_t nb3,
        const float alpha, const int is_v) {
    const int64_t row  = blockIdx.x;
    const int64_t head = blockIdx.y;
    const int64_t strm = blockIdx.z;
    const int j = threadIdx.x;
    if (j >= ne0) return;

    const char * src_row = src + strm * nb3 + head * nb2 + row * nb1;
    const int blk_idx  = j / QK_TURBO2_TCQ;
    const int t = j % QK_TURBO2_TCQ;
    const block_turbo2_tcq * blk = (const block_turbo2_tcq *)src_row + blk_idx;

    const float norm = __half2float(blk->norm) * alpha;

    // Sliding window decode: read 8-bit state from bitstream at bit offset t*2
    const int bit_pos = t * 2;
    const int byte_idx = bit_pos / 8;
    const int bit_off = bit_pos % 8;
    const uint16_t raw = (uint16_t)blk->qs[byte_idx] | ((uint16_t)blk->qs[byte_idx + 1] << 8);
    const int state = (raw >> bit_off) & 0xFF;
    const float val = (is_v ? d_turbo2_tcq_codebook_v_fattn : d_turbo2_tcq_codebook_fattn)[state] * norm;

    dst[strm * (ne1 * ne2 * ne0) + row * (ne2 * ne0) + head * ne0 + j] = __float2half(val);
}

static __global__ void k_turbo4_dequant_f16(
        const char * __restrict__ src, half * __restrict__ dst,
        const int64_t ne0, const int64_t ne1, const int64_t ne2,
        const size_t nb1, const size_t nb2, const size_t nb3) {
    const int64_t row  = blockIdx.x;
    const int64_t head = blockIdx.y;
    const int64_t strm = blockIdx.z;
    const int j = threadIdx.x;
    if (j >= ne0) return;

    const char * src_row = src + strm * nb3 + head * nb2 + row * nb1;
    const int blk_idx  = j / QK_TURBO4;
    const int j_in_blk = j % QK_TURBO4;
    const block_turbo4_0 * blk = (const block_turbo4_0 *)src_row + blk_idx;

    const float norm = __half2float(blk->norm);
    const uint8_t idx = (j_in_blk & 1) ? (blk->qs[j_in_blk / 2] >> 4) : (blk->qs[j_in_blk / 2] & 0xF);
    const float val = d_turbo_centroids_4bit_fattn[idx] * norm;

    dst[strm * (ne1 * ne2 * ne0) + row * (ne2 * ne0) + head * ne0 + j] = __float2half(val);
}

// turbo3 K dequant with inverse FWHT: produces K in original (unrotated) domain
// so Q does NOT need pre-rotation. 128 threads per block, loops over 128-element FWHT groups
// (each group spans 4 turbo3 storage blocks of 32 elements; all 4 blocks share the same norm).
// In-place 128-point inverse FWHT spread across 128 threads, one element per thread.
// h=1..16 use intra-warp __shfl_xor_sync (no smem, no syncthreads).
// h=32 and h=64 cross warp boundaries, so we fall back to smem with syncs.
// Caller supplies a __shared__ float[128] scratch buffer.
static __device__ __forceinline__ float fwht128_butterfly_inplace(float val, float * smem) {
    const int tid = threadIdx.x;

    // Intra-warp passes: shuffle xor with stride h, no smem, no sync.
    #pragma unroll
    for (int h = 1; h <= 16; h *= 2) {
        const float other = __shfl_xor_sync(0xFFFFFFFFULL, val, h);
        val = (tid & h) ? (other - val) : (val + other);
    }

    // h=32 (cross-warp within block, smem needed)
    smem[tid] = val;
    __syncthreads();
    val = (tid & 32) ? (smem[tid - 32] - val) : (val + smem[tid + 32]);
    __syncthreads();

    // h=64 (cross-warp within block, smem needed)
    smem[tid] = val;
    __syncthreads();
    val = (tid & 64) ? (smem[tid - 64] - val) : (val + smem[tid + 64]);
    __syncthreads();

    return val;
}

// Pair-pack two adjacent threads' fp16 outputs into a single 32-bit half2 store.
// Halves the global store count vs. one __float2half per thread.
static __device__ __forceinline__ void fwht128_store_half(
        float val, half * dst_base) {
    const int tid = threadIdx.x;
    const float neighbor = __shfl_xor_sync(0xFFFFFFFFULL, val, 1);
    if ((tid & 1) == 0) {
        const half2 packed = __floats2half2_rn(val, neighbor);
        *((half2 *)(dst_base + tid)) = packed;
    }
}

static __global__ void k_turbo3_dequant_f16_inv_fwht(
        const char * __restrict__ src, half * __restrict__ dst,
        const int64_t ne0, const int64_t ne1, const int64_t ne2,
        const size_t nb1, const size_t nb2, const size_t nb3) {
    const int64_t row  = blockIdx.x;
    const int64_t head = blockIdx.y;
    const int64_t strm = blockIdx.z;
    const int tid = threadIdx.x;

    const char * src_row = src + strm * nb3 + head * nb2 + row * nb1;
    const int64_t dst_base = strm * (ne1 * ne2 * ne0) + row * (ne2 * ne0) + head * ne0;

    __shared__ float smem[128];

    const float * s1 = d_turbo_wht_signs1_fattn;
    const float * s2 = d_turbo_wht_signs2_fattn;
    constexpr float inv_sqrt_128 = 0.08838834764831845f;

    // ne0 in elements, FWHT group = 128 elements = 4 turbo3 blocks of 32
    const int n_groups = (int)(ne0 / 128);
    constexpr int blocks_per_group = 128 / QK_TURBO3; // 4

    for (int g = 0; g < n_groups; g++) {
        // Element index within the FWHT group
        const int j_in_grp = tid;            // 0..127
        const int blk_in_grp = j_in_grp / QK_TURBO3;  // 0..3
        const int j_in_blk  = j_in_grp % QK_TURBO3;   // 0..31

        const block_turbo3_0 * blk = (const block_turbo3_0 *)src_row + g * blocks_per_group + blk_in_grp;
        const float norm = __half2float(blk->norm);   // same for all 4 blocks in this group

        const uint8_t low2 = (blk->qs[j_in_blk / 4] >> ((j_in_blk % 4) * 2)) & 0x3;
        const uint8_t hi1  = (blk->signs[j_in_blk / 8] >> (j_in_blk % 8)) & 0x1;
        const float c = d_turbo_centroids_3bit_fattn[low2 | (hi1 << 2)];

        // Inverse FWHT: 5 intra-warp shfl passes + 2 cross-warp smem passes.
        float val = fwht128_butterfly_inplace(c * s2[tid], smem);

        // Normalize, apply signs1, undo InnerQ scaling, multiply by norm, cast to fp16
        val = val * inv_sqrt_128 * s1[tid] * d_innerq_channel_scale_inv_fattn[tid] * norm;
        fwht128_store_half(val, dst + dst_base + g * 128);
        __syncthreads();
    }
}

// turbo4 K dequant with inverse FWHT: produces K in original (unrotated) domain
// so Q does NOT need pre-rotation. 128 threads per block, loops over 128-element turbo4 blocks.
static __global__ void k_turbo4_dequant_f16_inv_fwht(
        const char * __restrict__ src, half * __restrict__ dst,
        const int64_t ne0, const int64_t ne1, const int64_t ne2,
        const size_t nb1, const size_t nb2, const size_t nb3) {
    const int64_t row  = blockIdx.x;
    const int64_t head = blockIdx.y;
    const int64_t strm = blockIdx.z;
    const int tid = threadIdx.x;

    const char * src_row = src + strm * nb3 + head * nb2 + row * nb1;
    const int64_t dst_base = strm * (ne1 * ne2 * ne0) + row * (ne2 * ne0) + head * ne0;

    __shared__ float smem[128];

    const float * s1 = d_turbo_wht_signs1_fattn;
    const float * s2 = d_turbo_wht_signs2_fattn;
    constexpr float inv_sqrt_128 = 0.08838834764831845f;

    const int n_blocks = (int)(ne0 / QK_TURBO4);

    for (int blk_idx = 0; blk_idx < n_blocks; blk_idx++) {
        const block_turbo4_0 * blk = (const block_turbo4_0 *)src_row + blk_idx;
        const float norm = __half2float(blk->norm);

        const uint8_t idx = (tid & 1) ? (blk->qs[tid / 2] >> 4) : (blk->qs[tid / 2] & 0xF);

        float val = fwht128_butterfly_inplace(d_turbo_centroids_4bit_fattn[idx] * s2[tid], smem);

        val = val * inv_sqrt_128 * s1[tid] * d_innerq_channel_scale_inv_fattn[tid] * norm;
        fwht128_store_half(val, dst + dst_base + blk_idx * 128);
        __syncthreads();
    }
}

// turbo8 V dequant: simple centroid×norm (rotated domain, V un-rotation done at graph level).
static __global__ void k_turbo8_dequant_f16(
        const char * __restrict__ src, half * __restrict__ dst,
        const int64_t ne0, const int64_t ne1, const int64_t ne2,
        const size_t nb1, const size_t nb2, const size_t nb3) {
    const int64_t row  = blockIdx.x;
    const int64_t head = blockIdx.y;
    const int64_t strm = blockIdx.z;
    const int j = threadIdx.x;
    if (j >= ne0) return;

    const char * src_row = src + strm * nb3 + head * nb2 + row * nb1;
    const int blk_idx  = j / QK_TURBO8;
    const int j_in_blk = j % QK_TURBO8;
    const block_turbo8_0 * blk = (const block_turbo8_0 *)src_row + blk_idx;

    const float norm = __half2float(blk->norm);
    const float val = d_turbo_centroids_8bit_fattn[blk->qs[j_in_blk]] * norm;

    dst[strm * (ne1 * ne2 * ne0) + row * (ne2 * ne0) + head * ne0 + j] = __float2half(val);
}

// turbo8 K dequant with inverse FWHT: produces K in original (unrotated) domain
// so Q does NOT need pre-rotation. 128 threads per block, loops over 128-element turbo8 blocks.
static __global__ void k_turbo8_dequant_f16_inv_fwht(
        const char * __restrict__ src, half * __restrict__ dst,
        const int64_t ne0, const int64_t ne1, const int64_t ne2,
        const size_t nb1, const size_t nb2, const size_t nb3) {
    const int64_t row  = blockIdx.x;
    const int64_t head = blockIdx.y;
    const int64_t strm = blockIdx.z;
    const int tid = threadIdx.x;

    const char * src_row = src + strm * nb3 + head * nb2 + row * nb1;
    const int64_t dst_base = strm * (ne1 * ne2 * ne0) + row * (ne2 * ne0) + head * ne0;

    __shared__ float smem[128];

    const float * s1 = d_turbo_wht_signs1_fattn;
    const float * s2 = d_turbo_wht_signs2_fattn;
    constexpr float inv_sqrt_128 = 0.08838834764831845f;

    const int n_blocks = (int)(ne0 / QK_TURBO8);

    for (int blk_idx = 0; blk_idx < n_blocks; blk_idx++) {
        const block_turbo8_0 * blk = (const block_turbo8_0 *)src_row + blk_idx;
        const float norm = __half2float(blk->norm);

        float val = fwht128_butterfly_inplace(d_turbo_centroids_8bit_fattn[blk->qs[tid]] * s2[tid], smem);

        val = val * inv_sqrt_128 * s1[tid] * d_innerq_channel_scale_inv_fattn[tid] * norm;
        fwht128_store_half(val, dst + dst_base + blk_idx * 128);
        __syncthreads();
    }
}

// turbo1 V dequant: sign * scale in the rotated domain. The graph applies the
// inverse FWHT on the attention output (V is stored rotated), so no rotation here.
static __global__ void k_turbo1_dequant_f16(
        const char * __restrict__ src, half * __restrict__ dst,
        const int64_t ne0, const int64_t ne1, const int64_t ne2,
        const size_t nb1, const size_t nb2, const size_t nb3) {
    const int64_t row  = blockIdx.x;
    const int64_t head = blockIdx.y;
    const int64_t strm = blockIdx.z;
    const int j = threadIdx.x;
    if (j >= ne0) return;

    const char * src_row = src + strm * nb3 + head * nb2 + row * nb1;
    const int blk_idx  = j / QK_TURBO1;
    const int j_in_blk = j % QK_TURBO1;
    const block_turbo1 * blk = (const block_turbo1 *)src_row + blk_idx;

    const float d = __half2float(blk->d);
    const uint8_t bit = (blk->qs[j_in_blk >> 3] >> (j_in_blk & 7)) & 1u;
    const float val = bit ? -d : d;

    dst[strm * (ne1 * ne2 * ne0) + row * (ne2 * ne0) + head * ne0 + j] = __float2half(val);
}

// turbo1 K dequant with inverse FWHT: produces K in original (unrotated) domain
// so Q does NOT need pre-rotation. 128 threads per block, loops over 128-element groups.
static __global__ void k_turbo1_dequant_f16_inv_fwht(
        const char * __restrict__ src, half * __restrict__ dst,
        const int64_t ne0, const int64_t ne1, const int64_t ne2,
        const size_t nb1, const size_t nb2, const size_t nb3) {
    const int64_t row  = blockIdx.x;
    const int64_t head = blockIdx.y;
    const int64_t strm = blockIdx.z;
    const int tid = threadIdx.x;

    const char * src_row = src + strm * nb3 + head * nb2 + row * nb1;
    const int64_t dst_base = strm * (ne1 * ne2 * ne0) + row * (ne2 * ne0) + head * ne0;

    __shared__ float smem[128];

    const float * s1 = d_turbo_wht_signs1_fattn;
    const float * s2 = d_turbo_wht_signs2_fattn;
    constexpr float inv_sqrt_128 = 0.08838834764831845f;

    const int n_blocks = (int)(ne0 / QK_TURBO1);

    for (int blk_idx = 0; blk_idx < n_blocks; blk_idx++) {
        const block_turbo1 * blk = (const block_turbo1 *)src_row + blk_idx;
        const float d = __half2float(blk->d);
        const uint8_t bit = (blk->qs[tid >> 3] >> (tid & 7)) & 1u;
        float recon = bit ? -d : d;
        if (d_turbo1_dither_fattn) {
            // Subtractive: recon = d*(q - u), q=+-1, u = same TPDF as encode (block-address seed).
            const float qv = bit ? -1.0f : 1.0f;
            const float u  = turbo1_tpdf_fattn((unsigned long long)(unsigned long)blk, tid, d_turbo1_dither_w_fattn);
            recon = d * (qv - u);
        }

        float val = fwht128_butterfly_inplace(recon * s2[tid], smem);

        val = val * inv_sqrt_128 * s1[tid] * d_innerq_channel_scale_inv_fattn[tid];
        fwht128_store_half(val, dst + dst_base + blk_idx * 128);
        __syncthreads();
    }
}

// turbo1_nsn dequant: per-row inverse FWHT to original domain + NSNQuant un-normalize/un-center.
// v = s1 * (s2 * sqrt(128) * invFWHT(sign*sigma) + o[layer][head*ne0 + blk*128 + tid]).
// Used for BOTH K and V (both decode to original domain — no graph un-rotation). o_layer = the
// per-(layer) slice of the per-chunk per-head mean buffer (selected K vs V by the caller).
static __global__ void k_turbo1_nsn_dequant_f16(
        const char * __restrict__ src, half * __restrict__ dst,
        const int64_t ne0, const int64_t ne1, const int64_t ne2,
        const size_t nb1, const size_t nb2, const size_t nb3,
        const float * __restrict__ o_layer) {
    const int64_t row  = blockIdx.x;
    const int64_t head = blockIdx.y;
    const int64_t strm = blockIdx.z;
    const int tid = threadIdx.x;

    const char * src_row = src + strm * nb3 + head * nb2 + row * nb1;
    const int64_t dst_base = strm * (ne1 * ne2 * ne0) + row * (ne2 * ne0) + head * ne0;

    __shared__ float smem[128];
    const float * s1a = d_turbo_wht_signs1_fattn;
    const float * s2a = d_turbo_wht_signs2_fattn;
    constexpr float inv_sqrt_128 = 0.08838834764831845f; // sigma
    constexpr float sqrt_128 = 11.31370849898476f;
    const int n_blocks = (int)(ne0 / QK_TURBO1_NSN);

    for (int blk_idx = 0; blk_idx < n_blocks; blk_idx++) {
        const block_turbo1_nsn * blk = (const block_turbo1_nsn *)src_row + blk_idx;
        const float bs1 = __half2float(blk->s1);
        const float bs2 = __half2float(blk->s2);
        const uint8_t bit = (blk->qs[tid >> 3] >> (tid & 7)) & 1u;
        const float recon = bit ? -inv_sqrt_128 : inv_sqrt_128; // sign * sigma

        float val = fwht128_butterfly_inplace(recon * s2a[tid], smem);
        // x_recon[tid] = invFWHT(sign*sigma)
        const float x_recon = val * inv_sqrt_128 * s1a[tid] * d_innerq_channel_scale_inv_fattn[tid];
        const float o = o_layer ? o_layer[head * ne0 + blk_idx * 128 + tid] : 0.0f;
        const float v_out = bs1 * (bs2 * sqrt_128 * x_recon + o);
        fwht128_store_half(v_out, dst + dst_base + blk_idx * 128);
        __syncthreads();
    }
}

// turbo1_cq dequant: per-row inverse FWHT of codebook reconstruction → original domain (K and V).
// recon_rotated[tid] = d * codebook[qs[tid/8]][tid%8], then inverse FWHT.
static __global__ void k_turbo1_cq_dequant_f16(
        const char * __restrict__ src, half * __restrict__ dst,
        const int64_t ne0, const int64_t ne1, const int64_t ne2,
        const size_t nb1, const size_t nb2, const size_t nb3) {
    const int64_t row  = blockIdx.x;
    const int64_t head = blockIdx.y;
    const int64_t strm = blockIdx.z;
    const int tid = threadIdx.x;

    const char * src_row = src + strm * nb3 + head * nb2 + row * nb1;
    const int64_t dst_base = strm * (ne1 * ne2 * ne0) + row * (ne2 * ne0) + head * ne0;

    __shared__ float smem[128];
    const float * s1 = d_turbo_wht_signs1_fattn;
    const float * s2 = d_turbo_wht_signs2_fattn;
    constexpr float inv_sqrt_128 = 0.08838834764831845f;
    const int n_blocks = (int)(ne0 / QK_TURBO1_CQ);

    for (int blk_idx = 0; blk_idx < n_blocks; blk_idx++) {
        const block_turbo1_cq * blk = (const block_turbo1_cq *)src_row + blk_idx;
        const float d = __half2float(blk->d);
        const int g = tid >> 3, k = tid & 7;
        const float recon = d * d_turbo1_cq_cb_fattn[(int)blk->qs[g] * 8 + k];

        float val = fwht128_butterfly_inplace(recon * s2[tid], smem);
        val = val * inv_sqrt_128 * s1[tid] * d_innerq_channel_scale_inv_fattn[tid];
        fwht128_store_half(val, dst + dst_base + blk_idx * 128);
        __syncthreads();
    }
}

// turbo1_tcq decode codebooks (256-state, separate K/V). Cold-start = turbo2_tcq decode anchor;
// TURBO1_TCQ_CB_K / _V (?? TURBO1_TCQ_CB) override. MUST match the encode-side symbols.
// Baked-in turbo1_tcq decode codebooks — MUST be bit-identical to the encode-side arrays in
// turbo-quant-cuda.cuh (K = seed5/iter135, V = seed1/iter490; joint median KLD 0.0509 @8192/24ch).
static __constant__ float d_turbo1_tcq_cb_fattn[256]   = {  // K decode
    -0.05690895f, -0.05690895f, +0.06699874f, +0.15792155f, +0.03688013f, +0.04268654f, -0.14011213f, -0.02965679f,
    -0.02438163f, -0.02438163f, -0.19953799f, -0.09293428f, -0.03144405f, -0.02849112f, +0.08199784f, +0.08199784f,
    +0.09005091f, +0.09005091f, -0.05940611f, -0.05580222f, -0.07773691f, +0.06265988f, -0.01465418f, -0.00772985f,
    -0.07816899f, -0.02297225f, +0.05464951f, +0.05464951f, +0.04231037f, +0.04641661f, -0.08938144f, -0.08938144f,
    +0.00365320f, +0.10741787f, -0.16739547f, -0.04616742f, -0.14219233f, -0.00901757f, -0.08096506f, -0.01814358f,
    +0.01111611f, +0.01161552f, -0.03937600f, -0.01077888f, +0.05573412f, +0.12505241f, -0.06129042f, -0.06129042f,
    +0.01023831f, +0.01023831f, -0.12061471f, -0.12061471f, -0.04995122f, +0.02263966f, +0.03203409f, +0.11107059f,
    -0.07132407f, +0.12659174f, +0.01221013f, +0.04215888f, -0.10731849f, -0.10731849f, +0.05396174f, +0.05396174f,
    -0.08873346f, -0.02921234f, -0.08342038f, +0.01016524f, -0.04502235f, -0.02802503f, +0.08100778f, +0.09756171f,
    +0.07306496f, +0.07941060f, +0.06482599f, +0.17851813f, -0.17608380f, -0.13157773f, -0.02980503f, -0.02493151f,
    -0.08945329f, -0.04105385f, +0.06271980f, +0.06271980f, +0.04196313f, +0.04196313f, -0.07387091f, -0.07387091f,
    -0.17707518f, -0.00678580f, -0.10799771f, -0.01689552f, +0.13846508f, +0.13846508f, +0.00282240f, +0.00900819f,
    -0.06046259f, +0.00667074f, +0.08675562f, +0.08675562f, +0.03267998f, +0.08196472f, -0.05092679f, -0.05092679f,
    -0.14121383f, -0.01879669f, -0.05341130f, -0.00773192f, -0.08829777f, +0.02199137f, +0.02482944f, +0.10630896f,
    -0.04461808f, -0.04461808f, +0.07664362f, +0.07906501f, +0.02414565f, +0.14515512f, -0.12866446f, -0.05883218f,
    +0.07126531f, +0.07126531f, -0.09748757f, -0.04700467f, -0.00578264f, -0.00578264f, -0.05956273f, +0.09283913f,
    +0.07512030f, +0.07512030f, -0.05542693f, -0.02299234f, -0.12123095f, -0.04918547f, +0.00890487f, +0.06046228f,
    +0.07268346f, +0.08064783f, -0.08709023f, -0.00295277f, +0.09379439f, +0.09379439f, -0.05568655f, -0.05568655f,
    -0.07594635f, -0.07594635f, +0.06194503f, +0.08429574f, +0.00319491f, +0.18098263f, +0.01526845f, +0.02230061f,
    +0.01046284f, +0.07030793f, -0.03586691f, +0.12040213f, -0.11210962f, -0.09723122f, +0.02993215f, +0.02993215f,
    -0.16422588f, -0.04235943f, +0.01761191f, +0.03886831f, -0.01567245f, +0.12241358f, +0.03636941f, +0.03636941f,
    -0.10172297f, -0.02071226f, +0.02888141f, +0.09684418f, -0.09620761f, -0.02462198f, +0.03707082f, +0.07376128f,
    -0.06408252f, +0.08888291f, +0.01361814f, +0.04392439f, +0.02771172f, +0.02771172f, -0.12401393f, -0.04075170f,
    +0.01655904f, +0.01655904f, -0.11275508f, -0.11275508f, +0.03323539f, +0.03323539f, -0.10384015f, +0.19677565f,
    +0.01188133f, +0.01514807f, +0.02290127f, +0.13536246f, +0.07205503f, +0.15930237f, -0.07375064f, -0.05046035f,
    -0.08193663f, -0.08193663f, -0.03241898f, +0.04459962f, -0.01891778f, -0.01165876f, +0.09547786f, +0.14283878f,
    +0.02682622f, +0.12911709f, -0.05600601f, -0.05600601f, -0.16854414f, -0.08491614f, +0.04553003f, +0.04553003f,
    +0.00329509f, +0.08321664f, -0.00810051f, +0.10836335f, -0.00792713f, -0.00792713f, -0.13737200f, -0.09546821f,
    +0.06686576f, +0.17066576f, -0.05565165f, -0.04070367f, -0.09663567f, -0.04676020f, +0.09122299f, +0.09122299f,
    +0.02007535f, +0.11729758f, +0.00424169f, +0.00424169f, +0.01952964f, +0.13675623f, -0.11927815f, -0.01915905f,
    +0.08001547f, +0.08001547f, -0.05414105f, -0.05414105f, -0.15429053f, -0.06953919f, +0.02160413f, +0.05599044f,
    -0.06471479f, -0.06471479f, +0.05585973f, +0.13141583f, -0.02186239f, +0.02123550f, -0.02659754f, +0.04245459f,
};
static __constant__ float d_turbo1_tcq_cb_v_fattn[256] = {  // V decode
    -0.05822275f, -0.04057012f, +0.07030253f, +0.07030253f, +0.00910952f, +0.13381734f, -0.15394895f, -0.03654402f,
    -0.06419166f, -0.06419166f, -0.00322056f, -0.00322056f, -0.04590228f, -0.02467079f, +0.07511869f, +0.07511869f,
    +0.05966081f, +0.06482045f, -0.05617626f, -0.05617626f, -0.10456542f, -0.04816543f, -0.03039524f, +0.03820165f,
    -0.02399842f, -0.02399842f, +0.03528102f, +0.11460238f, +0.01120800f, +0.01120800f, -0.12139006f, -0.12139006f,
    +0.06143143f, +0.15165478f, -0.00829714f, +0.03238043f, -0.07274882f, -0.07274882f, +0.01648950f, +0.04747831f,
    -0.13129655f, -0.08037784f, -0.01519387f, -0.01374798f, +0.09782079f, +0.09782079f, -0.04960374f, -0.04960374f,
    -0.02048401f, -0.02048401f, -0.14179772f, -0.10473511f, +0.05546302f, +0.07371018f, +0.01881939f, +0.14756195f,
    -0.04088041f, +0.00697273f, +0.02537729f, +0.03056769f, -0.13503331f, -0.09729782f, +0.05183329f, +0.18428555f,
    -0.17827763f, -0.15089703f, -0.03956327f, -0.01698648f, -0.03595189f, -0.03595189f, +0.07393072f, +0.07393072f,
    +0.08052437f, +0.17165796f, +0.00083603f, +0.03242609f, -0.16325922f, -0.12540159f, -0.01560353f, +0.00438857f,
    -0.06914193f, -0.05732659f, +0.04068601f, +0.05820430f, +0.00498649f, +0.09965154f, -0.11689261f, -0.11689261f,
    +0.04313961f, +0.11466423f, -0.07856513f, -0.00676461f, +0.14289343f, +0.14289343f, -0.01477776f, +0.00207787f,
    -0.08715416f, -0.08715416f, +0.05999934f, +0.05999934f, +0.08748800f, +0.08748800f, -0.03674424f, -0.03598228f,
    +0.04614007f, +0.04614007f, -0.10481524f, -0.10481524f, -0.03229756f, -0.03229756f, +0.02437326f, +0.10609587f,
    -0.04459647f, -0.04459647f, +0.08320531f, +0.08320531f, +0.09055579f, +0.09055579f, -0.09849767f, -0.03800276f,
    +0.02997772f, +0.02997772f, -0.10893033f, -0.10893033f, -0.05067614f, +0.01131399f, -0.02704101f, +0.09590967f,
    +0.08146118f, +0.08146118f, -0.07268019f, -0.05547163f, -0.15487537f, -0.03218244f, +0.03206071f, +0.07322103f,
    -0.01254456f, -0.01254456f, +0.10675898f, +0.11011963f, +0.07422783f, +0.07422783f, -0.05873942f, -0.05318457f,
    -0.08198534f, -0.08198534f, +0.03441365f, +0.06745050f, +0.00344696f, +0.02957109f, +0.10733211f, +0.17381467f,
    -0.12887627f, +0.06219940f, -0.02732148f, -0.00106800f, -0.11212935f, +0.11931811f, +0.01623658f, +0.01623658f,
    -0.05334378f, +0.00559292f, -0.17833017f, -0.06168034f, +0.04205889f, +0.05851458f, -0.06669462f, -0.06669462f,
    +0.00689281f, +0.04394012f, +0.08635867f, +0.11145297f, -0.09817721f, -0.05329924f, +0.05398905f, +0.07623540f,
    +0.11233878f, +0.17313740f, +0.00950995f, +0.01250056f, -0.05599560f, -0.03859993f, -0.07366411f, +0.00118889f,
    +0.02521585f, +0.02521585f, -0.11893971f, -0.10650937f, +0.05810528f, +0.05810528f, -0.16678603f, -0.04473093f,
    -0.01108021f, -0.00942814f, +0.07052016f, +0.14542146f, +0.07294562f, +0.07294562f, -0.06669298f, -0.06669298f,
    -0.03622010f, +0.02799591f, -0.11604622f, -0.06511734f, -0.00890048f, -0.00399124f, +0.13813104f, +0.13813104f,
    +0.03711793f, +0.08076523f, -0.05968392f, -0.05968392f, -0.16921099f, -0.07368306f, +0.01830781f, +0.03795826f,
    -0.03008453f, -0.03008453f, +0.04872094f, +0.16440891f, -0.00369157f, -0.00369157f, -0.15617938f, -0.09730178f,
    +0.04716997f, +0.04716997f, -0.07361570f, -0.06438473f, -0.04917597f, -0.03088157f, +0.08131150f, +0.08131150f,
    -0.07323092f, +0.16313137f, +0.02744950f, +0.02744950f, +0.03818675f, +0.08015263f, -0.10156032f, -0.01302207f,
    +0.05073140f, +0.05073140f, -0.05120790f, -0.05120790f, -0.05612922f, -0.05612922f, +0.06819829f, +0.08482070f,
    -0.10348321f, +0.11792021f, +0.00979673f, +0.03041706f, +0.00188910f, +0.00188910f, -0.06026103f, +0.05430736f,
};
static void turbo1_tcq_load_cb_fattn() {
    auto load_file = [](const char * p, float * out) -> bool {
        FILE * f = fopen(p, "rb"); if (!f) return false;
        bool ok = fread(out, sizeof(float), 256, f) == 256; fclose(f); return ok;
    };
    auto file_mtime = [](const char * p) -> long { struct stat st; return (p && stat(p, &st) == 0) ? (long)st.st_mtime : 0; };
    int dev = 0; cudaGetDevice(&dev);
    static int hot = -1; if (hot < 0) hot = getenv("TURBO1_TCQ_HOTSWAP") ? 1 : 0;
    static bool init[GGML_CUDA_MAX_DEVICES] = {};
    static long mk[GGML_CUDA_MAX_DEVICES] = {}, mv[GGML_CUDA_MAX_DEVICES] = {};
    const char * cb = getenv("TURBO1_TCQ_CB");
    const char * kp = getenv("TURBO1_TCQ_CB_K"); if (!kp) kp = cb;
    const char * vp = getenv("TURBO1_TCQ_CB_V"); if (!vp) vp = cb;
    const bool first = !init[dev];
    const long nk = (hot || first) ? file_mtime(kp) : mk[dev];
    const long nv = (hot || first) ? file_mtime(vp) : mv[dev];
    const bool do_k = first || (hot && nk != mk[dev]);
    const bool do_v = first || (hot && nv != mv[dev]);
    if (!do_k && !do_v) return;
    float buf[256];
    // cold-start default = baked-in best codebook (seed5/iter135 K, seed1/iter490 V); only env overrides it.
    if (do_k && kp && load_file(kp, buf)) { cudaMemcpyToSymbol(d_turbo1_tcq_cb_fattn,   buf, 256*sizeof(float)); mk[dev] = nk; }
    if (do_v && vp && load_file(vp, buf)) { cudaMemcpyToSymbol(d_turbo1_tcq_cb_v_fattn, buf, 256*sizeof(float)); mv[dev] = nv; }
    init[dev] = true;
    if (first)
        fprintf(stderr, "TCQ1 decode: K/V codebooks (K=%s V=%s) hotswap=%d\n", kp?kp:"baked-in", vp?vp:"baked-in", hot);
}

// turbo1_tcq dequant: trellis-state decode of rotated coords, then per-row inverse FWHT → original
// domain (K and V). state_tid = read_8_bits(qs, tid*1); recon = norm * cb[state]. is_v picks K/V book.
static __global__ void k_turbo1_tcq_dequant_f16(
        const char * __restrict__ src, half * __restrict__ dst,
        const int64_t ne0, const int64_t ne1, const int64_t ne2,
        const size_t nb1, const size_t nb2, const size_t nb3,
        const float alpha, const int is_v) {
    const int64_t row  = blockIdx.x;
    const int64_t head = blockIdx.y;
    const int64_t strm = blockIdx.z;
    const int tid = threadIdx.x;

    const char * src_row = src + strm * nb3 + head * nb2 + row * nb1;
    const int64_t dst_base = strm * (ne1 * ne2 * ne0) + row * (ne2 * ne0) + head * ne0;

    __shared__ float smem[128];
    const float * s1 = d_turbo_wht_signs1_fattn;
    const float * s2 = d_turbo_wht_signs2_fattn;
    const float * cb = is_v ? d_turbo1_tcq_cb_v_fattn : d_turbo1_tcq_cb_fattn;
    constexpr float inv_sqrt_128 = 0.08838834764831845f;
    const int n_blocks = (int)(ne0 / QK_TURBO1_TCQ);

    for (int blk_idx = 0; blk_idx < n_blocks; blk_idx++) {
        const block_turbo1_tcq * blk = (const block_turbo1_tcq *)src_row + blk_idx;
        const float norm = __half2float(blk->norm);
        const int bit_pos = tid;                 // k=1: bit offset = tid
        const int byte_idx = bit_pos >> 3;
        const int bit_off  = bit_pos & 7;
        const uint16_t raw = (uint16_t)blk->qs[byte_idx] | ((uint16_t)blk->qs[byte_idx + 1] << 8);
        const int state = (raw >> bit_off) & 0xFF;
        const float recon = norm * alpha * cb[state];

        float val = fwht128_butterfly_inplace(recon * s2[tid], smem);
        val = val * inv_sqrt_128 * s1[tid] * d_innerq_channel_scale_inv_fattn[tid];
        fwht128_store_half(val, dst + dst_base + blk_idx * 128);
        __syncthreads();
    }
}

// turbo2 K dequant with inverse FWHT: produces K in original (unrotated) domain.
// 128 threads per block, loops over 128-element FWHT groups (each group spans
// 4 turbo2 storage blocks of 32 elements; all 4 blocks share the same norm).
static __global__ void k_turbo2_dequant_f16_inv_fwht(
        const char * __restrict__ src, half * __restrict__ dst,
        const int64_t ne0, const int64_t ne1, const int64_t ne2,
        const size_t nb1, const size_t nb2, const size_t nb3) {
    const int64_t row  = blockIdx.x;
    const int64_t head = blockIdx.y;
    const int64_t strm = blockIdx.z;
    const int tid = threadIdx.x;

    const char * src_row = src + strm * nb3 + head * nb2 + row * nb1;
    const int64_t dst_base = strm * (ne1 * ne2 * ne0) + row * (ne2 * ne0) + head * ne0;

    __shared__ float smem[128];

    const float * s1 = d_turbo_wht_signs1_fattn;
    const float * s2 = d_turbo_wht_signs2_fattn;
    constexpr float inv_sqrt_128 = 0.08838834764831845f;

    const int n_groups = (int)(ne0 / 128);
    constexpr int blocks_per_group = 128 / QK_TURBO2; // 4

    for (int g = 0; g < n_groups; g++) {
        const int j_in_grp = tid;            // 0..127
        const int blk_in_grp = j_in_grp / QK_TURBO2;  // 0..3
        const int j_in_blk  = j_in_grp % QK_TURBO2;   // 0..31

        const block_turbo2_0 * blk = (const block_turbo2_0 *)src_row + g * blocks_per_group + blk_in_grp;
        const float norm = __half2float(blk->norm);

        const uint8_t idx = (blk->qs[j_in_blk / 4] >> ((j_in_blk % 4) * 2)) & 0x3;
        const float c = d_turbo_centroids_2bit_fattn[idx];

        float val = fwht128_butterfly_inplace(c * s2[tid], smem);

        val = val * inv_sqrt_128 * s1[tid] * d_innerq_channel_scale_inv_fattn[tid] * norm;
        fwht128_store_half(val, dst + dst_base + g * 128);
        __syncthreads();
    }
}

// turbo3_tcq K dequant with inverse FWHT: produces K in original (unrotated) domain.
// 128 threads per block, loops over 128-element TCQ blocks (1 block per FWHT group).
static __global__ void k_turbo3_tcq_dequant_f16_inv_fwht(
        const char * __restrict__ src, half * __restrict__ dst,
        const int64_t ne0, const int64_t ne1, const int64_t ne2,
        const size_t nb1, const size_t nb2, const size_t nb3,
        const float alpha) {
    const int64_t row  = blockIdx.x;
    const int64_t head = blockIdx.y;
    const int64_t strm = blockIdx.z;
    const int tid = threadIdx.x;

    const char * src_row = src + strm * nb3 + head * nb2 + row * nb1;
    const int64_t dst_base = strm * (ne1 * ne2 * ne0) + row * (ne2 * ne0) + head * ne0;

    __shared__ float smem[128];

    const float * s1 = d_turbo_wht_signs1_fattn;
    const float * s2 = d_turbo_wht_signs2_fattn;
    constexpr float inv_sqrt_128 = 0.08838834764831845f;

    const int n_blocks = (int)(ne0 / QK_TURBO3_TCQ);

    for (int blk_idx = 0; blk_idx < n_blocks; blk_idx++) {
        const block_turbo3_tcq * blk = (const block_turbo3_tcq *)src_row + blk_idx;
        const float norm = __half2float(blk->norm) * alpha;

        // Sliding window decode: read 9-bit state from bitstream at bit offset tid*3
        const int bit_pos = tid * 3;
        const int byte_idx = bit_pos / 8;
        const int bit_off = bit_pos % 8;
        const uint16_t raw = (uint16_t)blk->qs[byte_idx] | ((uint16_t)blk->qs[byte_idx + 1] << 8);
        const int state = (raw >> bit_off) & 0x1FF;
        const float c = d_turbo3_tcq_codebook_fattn[state];

        float val = fwht128_butterfly_inplace(c * s2[tid], smem);

        val = val * inv_sqrt_128 * s1[tid] * d_innerq_channel_scale_inv_fattn[tid] * norm;
        fwht128_store_half(val, dst + dst_base + blk_idx * 128);
        __syncthreads();
    }
}

// turbo2_tcq K dequant with inverse FWHT: produces K in original (unrotated) domain.
static __global__ void k_turbo2_tcq_dequant_f16_inv_fwht(
        const char * __restrict__ src, half * __restrict__ dst,
        const int64_t ne0, const int64_t ne1, const int64_t ne2,
        const size_t nb1, const size_t nb2, const size_t nb3,
        const float alpha) {
    const int64_t row  = blockIdx.x;
    const int64_t head = blockIdx.y;
    const int64_t strm = blockIdx.z;
    const int tid = threadIdx.x;

    const char * src_row = src + strm * nb3 + head * nb2 + row * nb1;
    const int64_t dst_base = strm * (ne1 * ne2 * ne0) + row * (ne2 * ne0) + head * ne0;

    __shared__ float smem[128];

    const float * s1 = d_turbo_wht_signs1_fattn;
    const float * s2 = d_turbo_wht_signs2_fattn;
    constexpr float inv_sqrt_128 = 0.08838834764831845f;

    const int n_blocks = (int)(ne0 / QK_TURBO2_TCQ);

    for (int blk_idx = 0; blk_idx < n_blocks; blk_idx++) {
        const block_turbo2_tcq * blk = (const block_turbo2_tcq *)src_row + blk_idx;
        const float norm = __half2float(blk->norm) * alpha;

        // Sliding window decode: read 8-bit state from bitstream at bit offset tid*2
        const int bit_pos = tid * 2;
        const int byte_idx = bit_pos / 8;
        const int bit_off = bit_pos % 8;
        const uint16_t raw = (uint16_t)blk->qs[byte_idx] | ((uint16_t)blk->qs[byte_idx + 1] << 8);
        const int state = (raw >> bit_off) & 0xFF;
        const float c = d_turbo2_tcq_codebook_fattn[state];

        float val = fwht128_butterfly_inplace(c * s2[tid], smem);

        val = val * inv_sqrt_128 * s1[tid] * d_innerq_channel_scale_inv_fattn[tid] * norm;
        fwht128_store_half(val, dst + dst_base + blk_idx * 128);
        __syncthreads();
    }
}

// q8_0 K dequant to f16 in TKHE layout, matching the turbo K dequant kernels.
// Used at D=512 when K=q8_0 paired with V=turbo: produces (F16, F16) for the FA dispatch
// and bypasses the (Q8_0, TURBO*) D=512 native VEC templates which have buggy SASS on
// sm_120 PTX-JIT for some K/V combos. Q8_0 is in original (unrotated) domain → output too.
// 1 thread per element, 1 block per (token, head, batch).
static __global__ void k_q8_0_dequant_f16_tkhe(
        const char * __restrict__ src, half * __restrict__ dst,
        const int64_t ne0, const int64_t ne1, const int64_t ne2,
        const size_t nb1, const size_t nb2, const size_t nb3) {
    const int64_t row  = blockIdx.x;
    const int64_t head = blockIdx.y;
    const int64_t strm = blockIdx.z;
    const int j = threadIdx.x;
    if (j >= ne0) return;

    const char * src_row = src + strm * nb3 + row * nb1 + head * nb2;
    const int blk_idx = j / QK8_0;
    const int j_in_blk = j % QK8_0;
    const block_q8_0 * blk = (const block_q8_0 *)src_row + blk_idx;
    const float d = __half2float(blk->d);
    const float val = d * (float)blk->qs[j_in_blk];

    dst[strm * (ne1 * ne2 * ne0) + row * (ne2 * ne0) + head * ne0 + j] = __float2half(val);
}

// bf16 K/V cast to f16 in the same TKHE layout as k_q8_0_dequant_f16_tkhe. Used when a bf16
// side is paired with a turbo side: the f16-only MMA/VEC kernels need an f16 input, and bf16 is
// already in the original (unrotated) domain so no rotation is applied. 1 thread per element.
static __global__ void k_bf16_to_f16_tkhe(
        const char * __restrict__ src, half * __restrict__ dst,
        const int64_t ne0, const int64_t ne1, const int64_t ne2,
        const size_t nb1, const size_t nb2, const size_t nb3) {
    const int64_t row  = blockIdx.x;
    const int64_t head = blockIdx.y;
    const int64_t strm = blockIdx.z;
    const int j = threadIdx.x;
    if (j >= ne0) return;

    const nv_bfloat16 * src_row = (const nv_bfloat16 *)(src + strm * nb3 + row * nb1 + head * nb2);
    const float val = __bfloat162float(src_row[j]);

    dst[strm * (ne1 * ne2 * ne0) + row * (ne2 * ne0) + head * ne0 + j] = __float2half(val);
}

static bool vbr_stage2a_promoted_k_type_supported(ggml_type t) {
    return t == GGML_TYPE_F16 ||
           t == GGML_TYPE_BF16 ||
           t == GGML_TYPE_Q8_0 ||
           t == GGML_TYPE_TURBO2_TCQ ||
           t == GGML_TYPE_TURBO1_TCQ ||
           t == GGML_TYPE_TURBO3_TCQ ||
           t == GGML_TYPE_TURBO4_0 ||
           t == GGML_TYPE_TURBO8_0;
}

static bool vbr_stage2a_v_type_uses_rotated_domain(ggml_type t) {
    return t == GGML_TYPE_TURBO2_0 ||
           t == GGML_TYPE_TURBO3_0 ||
           t == GGML_TYPE_TURBO4_0 ||
           t == GGML_TYPE_TURBO8_0 ||
           t == GGML_TYPE_TURBO3_TCQ ||
           t == GGML_TYPE_TURBO2_TCQ;
}

// Persistent Q rotation buffer per device (shared between prefill and decode paths)
static float * q_rot_buf[GGML_CUDA_MAX_DEVICES] = {};
static size_t  q_rot_buf_size[GGML_CUDA_MAX_DEVICES] = {};

// Persistent K/V fp16 dequant buffers per device (shared between prefill and decode paths)
static half * kv_dequant_k_buf[GGML_CUDA_MAX_DEVICES] = {};
static size_t  kv_dequant_k_buf_size[GGML_CUDA_MAX_DEVICES] = {};
static half * kv_dequant_k_promoted_buf[GGML_CUDA_MAX_DEVICES] = {};
static size_t  kv_dequant_k_promoted_buf_size[GGML_CUDA_MAX_DEVICES] = {};
static int32_t * vbr_stage2a_k_row_bands_buf[GGML_CUDA_MAX_DEVICES] = {};
static size_t    vbr_stage2a_k_row_bands_buf_size[GGML_CUDA_MAX_DEVICES] = {};
static half * kv_dequant_v_buf[GGML_CUDA_MAX_DEVICES] = {};
static size_t  kv_dequant_v_buf_size[GGML_CUDA_MAX_DEVICES] = {};
static half * kv_dequant_v_promoted_buf[GGML_CUDA_MAX_DEVICES] = {};
static size_t  kv_dequant_v_promoted_buf_size[GGML_CUDA_MAX_DEVICES] = {};
static int32_t * vbr_stage2a_v_row_bands_buf[GGML_CUDA_MAX_DEVICES] = {};
static size_t    vbr_stage2a_v_row_bands_buf_size[GGML_CUDA_MAX_DEVICES] = {};

// === FWHT rotation kernels for pre-rotate-queries approach ===
// Forward rotation on Q before attention (both prefill and decode paths).
// One block per 128-element group, 128 threads per block.
static __global__ void k_turbo_fwht_forward(
        const float * __restrict__ src, float * __restrict__ dst,
        const int64_t n_elements) {
    const int64_t offset = blockIdx.x * 128;
    if (offset >= n_elements) return;

    const float * s1 = d_turbo_wht_signs1_fattn;
    const float * s2 = d_turbo_wht_signs2_fattn;

    __shared__ float buf[128];

    // InnerQ: apply inverse channel scale to Q before rotation
    float val = src[offset + threadIdx.x] * d_innerq_channel_scale_inv_fattn[threadIdx.x] * s1[threadIdx.x];

    val = fwht128_butterfly_inplace(val, buf);

    constexpr float inv_sqrt_128 = 0.08838834764831845f;
    val = val * inv_sqrt_128 * s2[threadIdx.x];
    dst[offset + threadIdx.x] = val;

    // Q² calibration: accumulate per-position squared values
    if (d_q_calibrate_fattn) {
        atomicAdd_double(&d_q_channel_sq_fattn[threadIdx.x], (double)(val * val));
        if (threadIdx.x == 0) atomicAdd(&d_q_channel_count_fattn, 1);
    }
}

static __device__ int64_t vbr_stage2a_src_row_for_logical_row(
        const int64_t row,
        const int32_t * __restrict__ bands,
        const int64_t n_bands,
        const int64_t band_stride) {
    for (int64_t ib = 0; ib < n_bands; ++ib) {
        const int64_t bo = ib * band_stride;
        const int32_t row0 = bands[bo + 0];
        const int32_t row1 = bands[bo + 1];
        if (row0 < 0 || row1 < 0) {
            continue;
        }
        if (row >= row0 && row < row1) {
            if (band_stride >= 4 && bands[bo + 2] >= 0) {
                return (int64_t) bands[bo + 2] + (row - row0);
            }
            return row;
        }
    }
    return -1;
}

static __global__ void k_vbr_stage2a_copy_promoted_rows_f16(
        half * __restrict__ dst,
        const half * __restrict__ src,
        const int32_t * __restrict__ bands,
        const int64_t n_bands,
        const int64_t band_stride,
        const int64_t ne0, const int64_t dst_ne1, const int64_t src_ne1, const int64_t ne2) {
    const int64_t row  = blockIdx.x;
    const int64_t head = blockIdx.y;
    const int64_t strm = blockIdx.z;
    const int j = threadIdx.x;
    if (j >= ne0 || row >= dst_ne1) {
        return;
    }

    const int64_t src_row = vbr_stage2a_src_row_for_logical_row(row, bands, n_bands, band_stride);
    if (src_row < 0 || src_row >= src_ne1) {
        return;
    }

    const int64_t dst_off = strm * (dst_ne1 * ne2 * ne0) + row     * (ne2 * ne0) + head * ne0 + j;
    const int64_t src_off = strm * (src_ne1 * ne2 * ne0) + src_row * (ne2 * ne0) + head * ne0 + j;
    dst[dst_off] = src[src_off];
}

static __global__ void k_vbr_stage2a_copy_promoted_rows_bf16(
        half * __restrict__ dst,
        const char * __restrict__ src,
        const int32_t * __restrict__ bands,
        const int64_t n_bands,
        const int64_t band_stride,
        const int64_t ne0, const int64_t dst_ne1, const int64_t src_ne1, const int64_t ne2,
        const size_t src_nb1, const size_t src_nb2, const size_t src_nb3) {
    const int64_t row  = blockIdx.x;
    const int64_t head = blockIdx.y;
    const int64_t strm = blockIdx.z;
    const int j = threadIdx.x;
    if (j >= ne0 || row >= dst_ne1) {
        return;
    }

    const int64_t src_row = vbr_stage2a_src_row_for_logical_row(row, bands, n_bands, band_stride);
    if (src_row < 0 || src_row >= src_ne1) {
        return;
    }

    const int64_t dst_off = strm * (dst_ne1 * ne2 * ne0) + row * (ne2 * ne0) + head * ne0 + j;
    const nv_bfloat16 * src_ptr = (const nv_bfloat16 *) (src + strm * src_nb3 + src_row * src_nb1 + head * src_nb2);
    dst[dst_off] = __float2half(__bfloat162float(src_ptr[j]));
}

static __global__ void k_vbr_stage2a_copy_promoted_rows_native_f16(
        half * __restrict__ dst,
        const char * __restrict__ src,
        const int32_t * __restrict__ bands,
        const int64_t n_bands,
        const int64_t band_stride,
        const int64_t ne0, const int64_t dst_ne1, const int64_t src_ne1, const int64_t ne2,
        const size_t src_nb1, const size_t src_nb2, const size_t src_nb3) {
    const int64_t row  = blockIdx.x;
    const int64_t head = blockIdx.y;
    const int64_t strm = blockIdx.z;
    const int j = threadIdx.x;
    if (j >= ne0 || row >= dst_ne1) {
        return;
    }

    const int64_t src_row = vbr_stage2a_src_row_for_logical_row(row, bands, n_bands, band_stride);
    if (src_row < 0 || src_row >= src_ne1) {
        return;
    }

    const int64_t dst_off = strm * (dst_ne1 * ne2 * ne0) + row * (ne2 * ne0) + head * ne0 + j;
    const half * src_ptr = (const half *) (src + strm * src_nb3 + src_row * src_nb1 + head * src_nb2);
    dst[dst_off] = src_ptr[j];
}

static __global__ void k_vbr_stage2a_cast_native_f16_tkhe(
        const char * __restrict__ src, half * __restrict__ dst,
        const int64_t ne0, const int64_t ne1, const int64_t ne2,
        const size_t nb1, const size_t nb2, const size_t nb3) {
    const int64_t row  = blockIdx.x;
    const int64_t head = blockIdx.y;
    const int64_t strm = blockIdx.z;
    const int j = threadIdx.x;
    if (j >= ne0) {
        return;
    }

    const half * src_ptr = (const half *) (src + strm * nb3 + row * nb1 + head * nb2);
    const int64_t dst_off = strm * (ne1 * ne2 * ne0) + row * (ne2 * ne0) + head * ne0 + j;
    dst[dst_off] = src_ptr[j];
}

static __global__ void k_vbr_stage2a_v_rotated_to_original_f16(
        half * __restrict__ data,
        const int64_t ne0, const int64_t ne1, const int64_t ne2, const int64_t ne3) {
    const int64_t group = blockIdx.x;
    const int64_t row   = blockIdx.y;
    const int64_t hz    = blockIdx.z;
    const int64_t head  = hz % ne2;
    const int64_t strm  = hz / ne2;
    const int tid = threadIdx.x;
    if (tid >= 128 || group * 128 + tid >= ne0 || row >= ne1 || strm >= ne3) {
        return;
    }

    const int64_t off = strm * (ne1 * ne2 * ne0) + row * (ne2 * ne0) + head * ne0 + group * 128 + tid;
    __shared__ float smem[128];
    const float val0 = __half2float(data[off]) * d_turbo_wht_signs2_fattn[tid];
    float val = fwht128_butterfly_inplace(val0, smem);
    constexpr float inv_sqrt_128 = 0.08838834764831845f;
    val = val * inv_sqrt_128 * d_turbo_wht_signs1_fattn[tid] * d_innerq_channel_scale_inv_fattn[tid];
    data[off] = __float2half(val);
}

static __global__ void k_vbr_stage2a_v_original_to_rotated_f16(
        half * __restrict__ data,
        const int64_t ne0, const int64_t ne1, const int64_t ne2, const int64_t ne3) {
    const int64_t group = blockIdx.x;
    const int64_t row   = blockIdx.y;
    const int64_t hz    = blockIdx.z;
    const int64_t head  = hz % ne2;
    const int64_t strm  = hz / ne2;
    const int tid = threadIdx.x;
    if (tid >= 128 || group * 128 + tid >= ne0 || row >= ne1 || strm >= ne3) {
        return;
    }

    const int64_t off = strm * (ne1 * ne2 * ne0) + row * (ne2 * ne0) + head * ne0 + group * 128 + tid;
    __shared__ float smem[128];
    const float val0 = __half2float(data[off]) * d_turbo_wht_signs1_fattn[tid];
    float val = fwht128_butterfly_inplace(val0, smem);
    constexpr float inv_sqrt_128 = 0.08838834764831845f;
    val = val * inv_sqrt_128 * d_turbo_wht_signs2_fattn[tid];
    data[off] = __float2half(val);
}

static void ggml_cuda_turbo_prefill_attend(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    load_tcq_decode_alpha(ctx.device);
    cudaStream_t stream = ctx.stream();
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    const bool turbo_k = K->type == GGML_TYPE_TURBO2_0 || K->type == GGML_TYPE_TURBO3_0 || K->type == GGML_TYPE_TURBO4_0 || K->type == GGML_TYPE_TURBO8_0 || K->type == GGML_TYPE_TURBO3_TCQ || K->type == GGML_TYPE_TURBO2_TCQ || K->type == GGML_TYPE_TURBO1 || K->type == GGML_TYPE_TURBO1_NSN || K->type == GGML_TYPE_TURBO1_CQ || K->type == GGML_TYPE_TURBO1_TCQ;
    const bool turbo_v = V->type == GGML_TYPE_TURBO2_0 || V->type == GGML_TYPE_TURBO3_0 || V->type == GGML_TYPE_TURBO4_0 || V->type == GGML_TYPE_TURBO8_0 || V->type == GGML_TYPE_TURBO3_TCQ || V->type == GGML_TYPE_TURBO2_TCQ || V->type == GGML_TYPE_TURBO1 || V->type == GGML_TYPE_TURBO1_NSN || V->type == GGML_TYPE_TURBO1_CQ || V->type == GGML_TYPE_TURBO1_TCQ;
    // Mixed (asymmetric) K/V: a q8_0 side must be dequanted to f16 here too, otherwise the raw
    // q8_0 bytes are handed to the f16-only MMA kernel below and read as f16 → garbage. f16 sides
    // need no conversion (already f16). This is the prefill mirror of the decode dequant path.
    const bool q8_k = K->type == GGML_TYPE_Q8_0;
    const bool q8_v = V->type == GGML_TYPE_Q8_0;
    // bf16 side paired with turbo: cast to f16 too (same reason as q8_0 above).
    const bool bf16_k = K->type == GGML_TYPE_BF16;
    const bool bf16_v = V->type == GGML_TYPE_BF16;

    int device;
    CUDA_CHECK(cudaGetDevice(&device));

    half * k_fp16 = nullptr;
    half * v_fp16 = nullptr;

    // Allocate and dequant K to fp16 (turbo2/3/4, q8_0, or bf16)
    if (turbo_k || q8_k || bf16_k) {
        // Size for full cache (kv_size from root) so we never realloc mid-session.
        const ggml_tensor * k_root = K;
        while (k_root->view_src) k_root = k_root->view_src;
        const size_t k_size = (size_t) ggml_nelements(k_root) * sizeof(half);
        if (k_size > kv_dequant_k_buf_size[device]) {
            if (kv_dequant_k_buf[device]) CUDA_CHECK(cudaFree(kv_dequant_k_buf[device]));
            CUDA_CHECK(cudaMalloc(&kv_dequant_k_buf[device], k_size));
            kv_dequant_k_buf_size[device] = k_size;
        }
        k_fp16 = kv_dequant_k_buf[device];
        dim3 grid_k(K->ne[1], K->ne[2], K->ne[3]);
        if (K->type == GGML_TYPE_TURBO2_0) {
            k_turbo2_dequant_f16<<<grid_k, K->ne[0], 0, stream>>>(
                (const char *)K->data, k_fp16, K->ne[0], K->ne[1], K->ne[2], K->nb[1], K->nb[2], K->nb[3]);
        } else if (K->type == GGML_TYPE_TURBO3_0) {
            k_turbo3_dequant_f16<<<grid_k, K->ne[0], 0, stream>>>(
                (const char *)K->data, k_fp16, K->ne[0], K->ne[1], K->ne[2], K->nb[1], K->nb[2], K->nb[3]);
        } else if (K->type == GGML_TYPE_TURBO3_TCQ) {
            {
                static bool tcq_fattn_k_cb_loaded[GGML_CUDA_MAX_DEVICES] = {};
                static int tcq_hot_k = -1; if (tcq_hot_k < 0) tcq_hot_k = getenv("TURBO_TCQ_HOTSWAP") ? 1 : 0;
                if (!tcq_fattn_k_cb_loaded[device] || tcq_hot_k) {
                    tcq_fattn_k_cb_loaded[device] = true;
                    turbo_tcq_load_kv_decode();
                }
            }
            // TCQ K quality is sensitive to where the FWHT roundtrip is rounded. Decode to the
            // original domain here, matching the decode-side dequant path, instead of rotating Q
            // and consuming rotated-domain K in the F16 MMA kernel.
            k_turbo3_tcq_dequant_f16_inv_fwht<<<grid_k, 128, 0, stream>>>(
                (const char *)K->data, k_fp16, K->ne[0], K->ne[1], K->ne[2], K->nb[1], K->nb[2], K->nb[3], d_tcq_decode_alpha_k);
        } else if (K->type == GGML_TYPE_TURBO2_TCQ) {
            {
                static bool tcq2_fattn_k_cb_loaded[GGML_CUDA_MAX_DEVICES] = {};
                static int tcq2_hot_k = -1; if (tcq2_hot_k < 0) tcq2_hot_k = getenv("TURBO_TCQ_HOTSWAP") ? 1 : 0;
                if (!tcq2_fattn_k_cb_loaded[device] || tcq2_hot_k) {
                    tcq2_fattn_k_cb_loaded[device] = true;
                    turbo2_tcq_load_kv_decode();
                }
            }
            k_turbo2_tcq_dequant_f16_inv_fwht<<<grid_k, 128, 0, stream>>>(
                (const char *)K->data, k_fp16, K->ne[0], K->ne[1], K->ne[2], K->nb[1], K->nb[2], K->nb[3], d_tcq_decode_alpha_k);
        } else if (K->type == GGML_TYPE_TURBO4_0) {
            // turbo4 K: inverse FWHT dequant → produces K in original domain (no Q rotation needed)
            k_turbo4_dequant_f16_inv_fwht<<<grid_k, 128, 0, stream>>>(
                (const char *)K->data, k_fp16, K->ne[0], K->ne[1], K->ne[2], K->nb[1], K->nb[2], K->nb[3]);
        } else if (K->type == GGML_TYPE_TURBO8_0) {
            // turbo8 K: inverse FWHT dequant → produces K in original domain (no Q rotation needed)
            k_turbo8_dequant_f16_inv_fwht<<<grid_k, 128, 0, stream>>>(
                (const char *)K->data, k_fp16, K->ne[0], K->ne[1], K->ne[2], K->nb[1], K->nb[2], K->nb[3]);
        } else if (K->type == GGML_TYPE_TURBO1) {
            // turbo1 K: inverse FWHT dequant → produces K in original domain (no Q rotation needed)
            k_turbo1_dequant_f16_inv_fwht<<<grid_k, 128, 0, stream>>>(
                (const char *)K->data, k_fp16, K->ne[0], K->ne[1], K->ne[2], K->nb[1], K->nb[2], K->nb[3]);
        } else if (K->type == GGML_TYPE_TURBO1_CQ) {
            k_turbo1_cq_dequant_f16<<<grid_k, 128, 0, stream>>>(
                (const char *)K->data, k_fp16, K->ne[0], K->ne[1], K->ne[2], K->nb[1], K->nb[2], K->nb[3]);
        } else if (K->type == GGML_TYPE_TURBO1_TCQ) {
            k_turbo1_tcq_dequant_f16<<<grid_k, 128, 0, stream>>>(
                (const char *)K->data, k_fp16, K->ne[0], K->ne[1], K->ne[2], K->nb[1], K->nb[2], K->nb[3], d_tcq_decode_alpha_k, 0);
        } else if (K->type == GGML_TYPE_TURBO1_NSN) {
            // turbo1_nsn K: inverse FWHT + NSNQuant un-normalize/un-center → original domain.
            const int kl_nsn = atoi(K->name + 9);
            const float * o_k = (kl_nsn >= 0 && kl_nsn < PFHEAD_MAX_L)
                ? turbo1_nsn_o_buf(device, 1) + (size_t) kl_nsn * PFHEAD_MAX_C : nullptr;
            k_turbo1_nsn_dequant_f16<<<grid_k, 128, 0, stream>>>(
                (const char *)K->data, k_fp16, K->ne[0], K->ne[1], K->ne[2], K->nb[1], K->nb[2], K->nb[3], o_k);
        } else if (K->type == GGML_TYPE_BF16) {
            // bf16 K (mixed K/V): cast to f16 in original (unrotated) domain. Q stays unrotated.
            k_bf16_to_f16_tkhe<<<grid_k, K->ne[0], 0, stream>>>(
                (const char *)K->data, k_fp16, K->ne[0], K->ne[1], K->ne[2], K->nb[1], K->nb[2], K->nb[3]);
        } else {
            // q8_0 K (mixed K/V): plain dequant to f16 in original (unrotated) domain. Q stays
            // unrotated (turbo_k is false → the Q-rotation block below is skipped).
            k_q8_0_dequant_f16_tkhe<<<grid_k, K->ne[0], 0, stream>>>(
                (const char *)K->data, k_fp16, K->ne[0], K->ne[1], K->ne[2], K->nb[1], K->nb[2], K->nb[3]);
        }
    }

    // Allocate and dequant V to fp16 (turbo2/3/4, q8_0, or bf16)
    if (turbo_v || q8_v || bf16_v) {
        // Size for full cache (kv_size from root) so we never realloc mid-session.
        const ggml_tensor * v_root = V;
        while (v_root->view_src) v_root = v_root->view_src;
        const size_t v_size = (size_t) ggml_nelements(v_root) * sizeof(half);
        if (v_size > kv_dequant_v_buf_size[device]) {
            if (kv_dequant_v_buf[device]) CUDA_CHECK(cudaFree(kv_dequant_v_buf[device]));
            CUDA_CHECK(cudaMalloc(&kv_dequant_v_buf[device], v_size));
            kv_dequant_v_buf_size[device] = v_size;
        }
        v_fp16 = kv_dequant_v_buf[device];
        dim3 grid_v(V->ne[1], V->ne[2], V->ne[3]);
        if (V->type == GGML_TYPE_TURBO2_0) {
            k_turbo2_dequant_f16<<<grid_v, V->ne[0], 0, stream>>>(
                (const char *)V->data, v_fp16, V->ne[0], V->ne[1], V->ne[2], V->nb[1], V->nb[2], V->nb[3]);
        } else if (V->type == GGML_TYPE_TURBO3_0) {
            k_turbo3_dequant_f16<<<grid_v, V->ne[0], 0, stream>>>(
                (const char *)V->data, v_fp16, V->ne[0], V->ne[1], V->ne[2], V->nb[1], V->nb[2], V->nb[3]);
        } else if (V->type == GGML_TYPE_TURBO3_TCQ) {
            // Runtime codebook loading for 3-bit V decode (in case K is a different type)
            {
                static bool tcq_fattn_v_cb_loaded[GGML_CUDA_MAX_DEVICES] = {};
                static int tcq_hot_v = -1; if (tcq_hot_v < 0) tcq_hot_v = getenv("TURBO_TCQ_HOTSWAP") ? 1 : 0;
                if (!tcq_fattn_v_cb_loaded[device] || tcq_hot_v) {
                    tcq_fattn_v_cb_loaded[device] = true;
                    turbo_tcq_load_kv_decode();
                }
            }
            k_turbo3_tcq_dequant_f16<<<grid_v, V->ne[0], 0, stream>>>(
                (const char *)V->data, v_fp16, V->ne[0], V->ne[1], V->ne[2], V->nb[1], V->nb[2], V->nb[3], tcq_compute_alpha_v(V->type, V->ne[1]), 1);
        } else if (V->type == GGML_TYPE_TURBO2_TCQ) {
            // Runtime codebook loading for 2-bit V decode (in case K is a different type)
            {
                static bool tcq2_fattn_v_cb_loaded[GGML_CUDA_MAX_DEVICES] = {};
                static int tcq2_hot_v = -1; if (tcq2_hot_v < 0) tcq2_hot_v = getenv("TURBO_TCQ_HOTSWAP") ? 1 : 0;
                if (!tcq2_fattn_v_cb_loaded[device] || tcq2_hot_v) {
                    tcq2_fattn_v_cb_loaded[device] = true;
                    turbo2_tcq_load_kv_decode();
                }
            }
            k_turbo2_tcq_dequant_f16<<<grid_v, V->ne[0], 0, stream>>>(
                (const char *)V->data, v_fp16, V->ne[0], V->ne[1], V->ne[2], V->nb[1], V->nb[2], V->nb[3], tcq_compute_alpha_v(V->type, V->ne[1]), 1);
        } else if (V->type == GGML_TYPE_TURBO4_0) {
            k_turbo4_dequant_f16<<<grid_v, V->ne[0], 0, stream>>>(
                (const char *)V->data, v_fp16, V->ne[0], V->ne[1], V->ne[2], V->nb[1], V->nb[2], V->nb[3]);
        } else if (V->type == GGML_TYPE_TURBO8_0) {
            k_turbo8_dequant_f16<<<grid_v, V->ne[0], 0, stream>>>(
                (const char *)V->data, v_fp16, V->ne[0], V->ne[1], V->ne[2], V->nb[1], V->nb[2], V->nb[3]);
        } else if (V->type == GGML_TYPE_TURBO1_CQ) {
            k_turbo1_cq_dequant_f16<<<grid_v, 128, 0, stream>>>(
                (const char *)V->data, v_fp16, V->ne[0], V->ne[1], V->ne[2], V->nb[1], V->nb[2], V->nb[3]);
        } else if (V->type == GGML_TYPE_TURBO1_TCQ) {
            k_turbo1_tcq_dequant_f16<<<grid_v, 128, 0, stream>>>(
                (const char *)V->data, v_fp16, V->ne[0], V->ne[1], V->ne[2], V->nb[1], V->nb[2], V->nb[3], tcq_compute_alpha_v(V->type, V->ne[1]), 1);
        } else if (V->type == GGML_TYPE_TURBO1) {
            // turbo1 V: per-row inverse FWHT → original domain (no graph un-rotation). Diagnostic:
            // the rotated-V + single graph-FWHT path (turbo8-style) is broken for turbo1 V only.
            k_turbo1_dequant_f16_inv_fwht<<<grid_v, 128, 0, stream>>>(
                (const char *)V->data, v_fp16, V->ne[0], V->ne[1], V->ne[2], V->nb[1], V->nb[2], V->nb[3]);
        } else if (V->type == GGML_TYPE_TURBO1_NSN) {
            const int vl_nsn = atoi(V->name + 9);
            const float * o_v = (vl_nsn >= 0 && vl_nsn < PFHEAD_MAX_L)
                ? turbo1_nsn_o_buf(device, 0) + (size_t) vl_nsn * PFHEAD_MAX_C : nullptr;
            k_turbo1_nsn_dequant_f16<<<grid_v, 128, 0, stream>>>(
                (const char *)V->data, v_fp16, V->ne[0], V->ne[1], V->ne[2], V->nb[1], V->nb[2], V->nb[3], o_v);
        } else if (V->type == GGML_TYPE_BF16) {
            // bf16 V (mixed K/V): cast to f16 in original domain. Not rotated, and the graph-level
            // inverse WHT is gated on V being a turbo type (llama-graph.cpp), so none is applied.
            k_bf16_to_f16_tkhe<<<grid_v, V->ne[0], 0, stream>>>(
                (const char *)V->data, v_fp16, V->ne[0], V->ne[1], V->ne[2], V->nb[1], V->nb[2], V->nb[3]);
        } else {
            // q8_0 V (mixed K/V): plain dequant to f16 in original domain. q8_0 V is not rotated,
            // and the graph-level inverse WHT is gated on V being a turbo type (llama-graph.cpp),
            // so no spurious un-rotation is applied here.
            k_q8_0_dequant_f16_tkhe<<<grid_v, V->ne[0], 0, stream>>>(
                (const char *)V->data, v_fp16, V->ne[0], V->ne[1], V->ne[2], V->nb[1], V->nb[2], V->nb[3]);
        }
    }

    // Create fp16 tensor copies on stack
    ggml_tensor K_f16 = *K;
    ggml_tensor V_f16 = *V;

    if (k_fp16) {
        K_f16.type = GGML_TYPE_F16;
        K_f16.data = k_fp16;
        K_f16.nb[0] = sizeof(half);
        K_f16.nb[1] = K->ne[0] * K->ne[2] * sizeof(half);  // row stride: head_dim * n_head_kv (matches native cache)
        K_f16.nb[2] = K->ne[0] * sizeof(half);             // head stride: head_dim (matches native cache)
        K_f16.nb[3] = K->ne[0] * K->ne[1] * K->ne[2] * sizeof(half);
    }

    if (v_fp16) {
        V_f16.type = GGML_TYPE_F16;
        V_f16.data = v_fp16;
        V_f16.nb[0] = sizeof(half);
        V_f16.nb[1] = V->ne[0] * V->ne[2] * sizeof(half);  // row stride: head_dim * n_head_kv (matches native cache)
        V_f16.nb[2] = V->ne[0] * sizeof(half);             // head stride: head_dim (matches native cache)
        V_f16.nb[3] = V->ne[0] * V->ne[1] * V->ne[2] * sizeof(half);
    }

    // Rotate Q for turbo pre-rotate-queries (only when K is in rotated space).
    // turbo4/turbo8 and TCQ K are dequanted via inverse FWHT -> original domain, so Q stays unrotated.
    const ggml_tensor * Q = dst->src[0];
    float * q_rotated = nullptr;
    if (turbo_k &&
            K->type != GGML_TYPE_TURBO4_0 &&
            K->type != GGML_TYPE_TURBO8_0 &&
            K->type != GGML_TYPE_TURBO3_TCQ &&
            K->type != GGML_TYPE_TURBO2_TCQ &&
            K->type != GGML_TYPE_TURBO1 &&
            K->type != GGML_TYPE_TURBO1_NSN &&
            K->type != GGML_TYPE_TURBO1_CQ &&
            K->type != GGML_TYPE_TURBO1_TCQ &&
            Q->ne[0] % 128 == 0) {
        const size_t q_size = ggml_nelements(Q) * sizeof(float);
        if (q_size > q_rot_buf_size[device]) {
            if (q_rot_buf[device]) CUDA_CHECK(cudaFree(q_rot_buf[device]));
            CUDA_CHECK(cudaMalloc(&q_rot_buf[device], q_size));
            q_rot_buf_size[device] = q_size;
        }
        q_rotated = q_rot_buf[device];
        const int64_t n_q_groups = ggml_nelements(Q) / 128;
        k_turbo_fwht_forward<<<(int)n_q_groups, 128, 0, stream>>>(
            (const float *)Q->data, q_rotated, ggml_nelements(Q));
    }

    // Temporarily swap src pointers to fp16 K/V and rotated Q
    ggml_tensor * orig_q = dst->src[0];
    ggml_tensor * orig_k = dst->src[1];
    ggml_tensor * orig_v = dst->src[2];

    ggml_tensor Q_rot;
    if (q_rotated) {
        Q_rot = *Q;
        Q_rot.data = q_rotated;
        dst->src[0] = &Q_rot;
    }
    dst->src[1] = k_fp16 ? &K_f16 : orig_k;
    dst->src[2] = v_fp16 ? &V_f16 : orig_v;
    std::atomic_signal_fence(std::memory_order_seq_cst);

    // Dispatch to MMA kernel (sees rotated Q, fp16 K/V, uses tensor cores)
    ggml_cuda_flash_attn_ext_mma_f16(ctx, dst);

    // Restore original tensor pointers
    dst->src[0] = orig_q;
    dst->src[1] = orig_k;
    dst->src[2] = orig_v;

    // K/V fp16 buffers are persistent (grow-only), no free needed
}

#define FATTN_VEC_CASE(D, type_K, type_V)                                                                        \
    {                                                                                                            \
        const bool type_K_okay = K->type == (type_K) || (K->type == GGML_TYPE_F32 && (type_K) == GGML_TYPE_F16); \
        const bool type_V_okay = V->type == (type_V) || (V->type == GGML_TYPE_F32 && (type_V) == GGML_TYPE_F16); \
        if (Q->ne[0] == (D) && type_K_okay && type_V_okay) {                                                     \
            ggml_cuda_flash_attn_ext_vec_case<D, type_K, type_V>(ctx, dst);                                      \
            return;                                                                                              \
        }                                                                                                        \
    }                                                                                                            \

#define FATTN_VEC_CASES_ALL_D(type_K, type_V) \
    FATTN_VEC_CASE( 64, type_K, type_V)       \
    FATTN_VEC_CASE(128, type_K, type_V)       \
    FATTN_VEC_CASE(256, type_K, type_V)       \

#define FATTN_VEC_CASES_ALL_D_512(type_K, type_V) \
    FATTN_VEC_CASE( 64, type_K, type_V)       \
    FATTN_VEC_CASE(128, type_K, type_V)       \
    FATTN_VEC_CASE(256, type_K, type_V)       \
    FATTN_VEC_CASE(512, type_K, type_V)       \

static void ggml_cuda_flash_attn_ext_vec(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_tensor * Q = dst->src[0];
    ggml_tensor * K = dst->src[1];
    ggml_tensor * V = dst->src[2];

#ifdef GGML_CUDA_FA_ALL_QUANTS
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_F16,  GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_F16)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_Q4_0)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_Q4_1)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_Q5_0)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_Q5_1)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q8_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_Q8_0)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_BF16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_BF16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_BF16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_BF16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_BF16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_BF16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_BF16)

    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO2_0, GGML_TYPE_TURBO2_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO3_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO4_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO2_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO3_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO4_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q8_0,     GGML_TYPE_TURBO2_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q8_0,     GGML_TYPE_TURBO3_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q8_0,     GGML_TYPE_TURBO4_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO3_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO4_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO2_0, GGML_TYPE_TURBO3_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO2_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO3_TCQ, GGML_TYPE_TURBO3_TCQ)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO2_TCQ, GGML_TYPE_TURBO2_TCQ)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO3_TCQ, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO2_TCQ, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q8_0,       GGML_TYPE_TURBO3_TCQ)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q8_0,       GGML_TYPE_TURBO2_TCQ)
#else
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_F16,  GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q8_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_BF16)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO2_0, GGML_TYPE_TURBO2_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO3_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO4_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO2_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO3_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO4_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q8_0,     GGML_TYPE_TURBO2_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q8_0,     GGML_TYPE_TURBO3_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q8_0,     GGML_TYPE_TURBO4_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO3_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO4_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO2_0, GGML_TYPE_TURBO3_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO2_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO3_TCQ, GGML_TYPE_TURBO3_TCQ)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO2_TCQ, GGML_TYPE_TURBO2_TCQ)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO3_TCQ, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO2_TCQ, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q8_0,       GGML_TYPE_TURBO3_TCQ)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q8_0,       GGML_TYPE_TURBO2_TCQ)
#endif // GGML_CUDA_FA_ALL_QUANTS

    GGML_ABORT("fatal error");
}

// Best FlashAttention kernel for a specific GPU:
enum best_fattn_kernel {
    BEST_FATTN_KERNEL_NONE     =   0,
    BEST_FATTN_KERNEL_TILE     = 200,
    BEST_FATTN_KERNEL_VEC      = 100,
    BEST_FATTN_KERNEL_WMMA_F16 = 300,
    BEST_FATTN_KERNEL_MMA_F16  = 400,
};

static best_fattn_kernel ggml_cuda_get_best_fattn_kernel(const int device, const ggml_tensor * dst) {
#ifndef FLASH_ATTN_AVAILABLE
    GGML_UNUSED(device); GGML_UNUSED(dst);
    return BEST_FATTN_KERNEL_NONE;
#endif// FLASH_ATTN_AVAILABLE

    const ggml_tensor * KQV   = dst;
    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * V     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];

    const int gqa_ratio = Q->ne[2] / K->ne[2];
    GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);

    float max_bias = 0.0f;
    memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

    // The effective batch size for the kernel can be increased by gqa_ratio.
    // The kernel versions without this optimization are also used for ALiBi, if there is no mask, or if the KV cache is not padded,
    bool gqa_opt_applies = gqa_ratio >= 2 && mask && max_bias == 0.0f && K->ne[1] % FATTN_KQ_STRIDE == 0;
    for (const ggml_tensor * t : {Q, K, V, mask}) {
        if (t == nullptr || ggml_is_quantized(t->type)) {
            continue;
        }
        for (size_t i = 1; i < GGML_MAX_DIMS; ++i) {
            if (t->nb[i] % 16 != 0) {
                gqa_opt_applies = false;
                break;
            }
        }
    }

    const int cc = ggml_cuda_info().devices[device].cc;

    switch (K->ne[0]) {
        case  40:
        case  64:
        case  72:
        case  80:
        case  96:
        case 128:
        case 112:
        case 256:
            if (V->ne[0] != K->ne[0]) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case 192:
            if (V->ne[0] != 128 || !gqa_opt_applies) {
                return BEST_FATTN_KERNEL_NONE;
            }
            if (gqa_ratio % 8 != 0) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case 320:
            if (V->ne[0] != 256 || !gqa_opt_applies) {
                return BEST_FATTN_KERNEL_NONE;
            }
            if (gqa_ratio % 32 != 0) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case 512:
            if (V->ne[0] != K->ne[0]) {
                return BEST_FATTN_KERNEL_NONE;
            }
            if (!gqa_opt_applies) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case 576:
            if (V->ne[0] != 512) {
                return BEST_FATTN_KERNEL_NONE;
            }
            if (!gqa_opt_applies) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        default:
            return BEST_FATTN_KERNEL_NONE;
    }

#ifndef GGML_CUDA_FA_ALL_QUANTS
    {
        auto is_turbo_type = [](ggml_type t) {
            return t == GGML_TYPE_TURBO2_0 || t == GGML_TYPE_TURBO3_0 || t == GGML_TYPE_TURBO4_0 || t == GGML_TYPE_TURBO8_0 ||
                   t == GGML_TYPE_TURBO3_TCQ || t == GGML_TYPE_TURBO2_TCQ || t == GGML_TYPE_TURBO1 || t == GGML_TYPE_TURBO1_NSN || t == GGML_TYPE_TURBO1_CQ || t == GGML_TYPE_TURBO1_TCQ;
        };
        // Asymmetric turbo K/V (e.g. q8_0 K + turbo4 V) is handled by the turbo dequant-to-f16
        // FA paths (ggml_cuda_turbo_prefill_attend for prefill, do_decode_dequant for decode).
        // Without this exemption the upstream K!=V guard reports FA unsupported, so llama.cpp
        // falls back to the non-FA attention path, which mishandles turbo4 V (garbage output).
        if (K->type != V->type && !is_turbo_type(K->type) && !is_turbo_type(V->type)) {
            return BEST_FATTN_KERNEL_NONE;
        }
    }
#endif // GGML_CUDA_FA_ALL_QUANTS

    switch (K->type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
            break;
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
#ifndef GGML_CUDA_FA_ALL_QUANTS
            return BEST_FATTN_KERNEL_NONE;
#endif // GGML_CUDA_FA_ALL_QUANTS
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_BF16:
        case GGML_TYPE_TURBO2_0:
        case GGML_TYPE_TURBO3_0:
        case GGML_TYPE_TURBO4_0:
        case GGML_TYPE_TURBO8_0:
        case GGML_TYPE_TURBO3_TCQ:
        case GGML_TYPE_TURBO2_TCQ:
        case GGML_TYPE_TURBO1:
        case GGML_TYPE_TURBO1_NSN:
        case GGML_TYPE_TURBO1_CQ:
        case GGML_TYPE_TURBO1_TCQ:
            break;
        default:
            return BEST_FATTN_KERNEL_NONE;
    }

    if (mask && mask->ne[2] != 1) {
        return BEST_FATTN_KERNEL_NONE;
    }

    // For small batch sizes the vector kernel may be preferable over the kernels optimized for large batch sizes:
    // TurboQuant: only the vec kernel has native turbo dequant support.
    if (K->type == GGML_TYPE_TURBO2_0 || V->type == GGML_TYPE_TURBO2_0 ||
        K->type == GGML_TYPE_TURBO3_0 || V->type == GGML_TYPE_TURBO3_0 ||
        K->type == GGML_TYPE_TURBO4_0 || V->type == GGML_TYPE_TURBO4_0 ||
        K->type == GGML_TYPE_TURBO8_0 || V->type == GGML_TYPE_TURBO8_0 ||
        K->type == GGML_TYPE_TURBO3_TCQ || V->type == GGML_TYPE_TURBO3_TCQ ||
        K->type == GGML_TYPE_TURBO2_TCQ || V->type == GGML_TYPE_TURBO2_TCQ ||
        K->type == GGML_TYPE_TURBO1    || V->type == GGML_TYPE_TURBO1 ||
        K->type == GGML_TYPE_TURBO1_NSN || K->type == GGML_TYPE_TURBO1_CQ || V->type == GGML_TYPE_TURBO1_NSN || V->type == GGML_TYPE_TURBO1_CQ ||
        K->type == GGML_TYPE_TURBO1_TCQ || V->type == GGML_TYPE_TURBO1_TCQ) {
        if (Q->ne[0] <= 512 && Q->ne[0] % 64 == 0)
            return BEST_FATTN_KERNEL_VEC;
        return BEST_FATTN_KERNEL_NONE;
    }

    // D=512: MMA/TILE templates don't support this head_dim, use VEC unconditionally
    if (Q->ne[0] == 512) {
        return BEST_FATTN_KERNEL_VEC;
    }

    // 192 satisfies % 64 == 0 but has no vec instance (DKQ != DV); force it onto the MMA path.
    const bool can_use_vector_kernel = Q->ne[0] <= 256 && Q->ne[0] % 64 == 0 && Q->ne[0] != 192 && K->ne[1] % FATTN_KQ_STRIDE == 0;

    // If Turing tensor cores are available, use them:
    if (turing_mma_available(cc) && Q->ne[0] != 40 && Q->ne[0] != 72) {
        if (can_use_vector_kernel) {
            if (!ggml_is_quantized(K->type) && !ggml_is_quantized(V->type)) {
                if (cc >= GGML_CUDA_CC_ADA_LOVELACE && Q->ne[1] == 1 && Q->ne[3] == 1 && !(gqa_ratio > 4 && K->ne[1] >= 8192)) {
                    return BEST_FATTN_KERNEL_VEC;
                }
            } else {
                if (cc >= GGML_CUDA_CC_ADA_LOVELACE) {
                    if (Q->ne[1] <= 2) {
                        return BEST_FATTN_KERNEL_VEC;
                    }
                } else {
                    if (Q->ne[1] == 1) {
                        return BEST_FATTN_KERNEL_VEC;
                    }
                }
            }
            if (!gqa_opt_applies && Q->ne[1] == 1) {
                return BEST_FATTN_KERNEL_VEC;
            }
        }
        return BEST_FATTN_KERNEL_MMA_F16;
    }

    const int ncols2_max = Q->ne[0] == 320 ? 32 : ((Q->ne[0] == 576 || Q->ne[0] == 192) ? 16 : 8);
    int gqa_ratio_eff = 1;
    while (gqa_ratio % (2*gqa_ratio_eff) == 0 && gqa_ratio_eff < ncols2_max) {
        gqa_ratio_eff *= 2;
    }

    if (volta_mma_available(cc) && Q->ne[0] != 40 && Q->ne[0] != 72) {
        if (can_use_vector_kernel && Q->ne[1] * gqa_ratio_eff <= 2) {
            return BEST_FATTN_KERNEL_VEC;
        }
        if (Q->ne[1] * gqa_ratio_eff <= 16) {
            return BEST_FATTN_KERNEL_TILE; // On Volta tensor cores are only faster for sufficiently large matrices.
        }
        return BEST_FATTN_KERNEL_MMA_F16;
    }

    // Use the WMMA kernel if possible:
    if (ggml_cuda_should_use_wmma_fattn(cc) && K->ne[1] % FATTN_KQ_STRIDE == 0 && Q->ne[0] != 40 && Q->ne[0] != 72 && Q->ne[0] != 192 && Q->ne[0] != 512 && Q->ne[0] != 576) {
        if (can_use_vector_kernel && Q->ne[1] <= 2) {
            return BEST_FATTN_KERNEL_VEC;
        }
        return BEST_FATTN_KERNEL_WMMA_F16;
    }

    // AMD MFMA needs a certain minimum batch size to outscale the tile kernel for large head sizes.
    if ((amd_mfma_available(cc) && Q->ne[0] <= 256) && Q->ne[0] != 40 && Q->ne[0] != 72) {
        if ((Q->ne[0] <= 64 && Q->ne[1] * gqa_ratio_eff > 8)) {
            return BEST_FATTN_KERNEL_MMA_F16;
        }
        if ((Q->ne[0] <= 128 && Q->ne[1] * gqa_ratio_eff > 16)) {
            return BEST_FATTN_KERNEL_MMA_F16;
        }
        if ((Q->ne[0] <= 256 && Q->ne[1] * gqa_ratio_eff > 64)) {
            return BEST_FATTN_KERNEL_MMA_F16;
        }
    }

    // AMD WMMA is always faster than the tile kernel if the full tile width of 16 can be utilized.
    if ((amd_wmma_available(cc) && gqa_opt_applies && Q->ne[0] <= 128) && Q->ne[0] != 40 && Q->ne[0] != 72 && Q->ne[1] * gqa_ratio_eff > 8) {
        return BEST_FATTN_KERNEL_MMA_F16;
    }

    // If there are no tensor cores available, use the generic tile kernel:
    if (can_use_vector_kernel) {
        if (!ggml_is_quantized(K->type) && !ggml_is_quantized(V->type)) {
            if (Q->ne[1] == 1) {
                if (!gqa_opt_applies) {
                    return BEST_FATTN_KERNEL_VEC;
                }
            }
        } else {
            if (Q->ne[1] <= 2) {
                return BEST_FATTN_KERNEL_VEC;
            }
        }
    }
    return BEST_FATTN_KERNEL_TILE;
}

// TURBO_CERT_DUMP_Q=<dir>: dump raw post-RoPE query vectors per (layer, head) as fp32 to
// <dir>/q_l%03d_h%02d_w%d.f32. Codec-independent; fires on every flash-attn node. Layer parsed
// from the "__fattn__-<il>" node name. Q src after permute(0,2,1,3): ne0=head_dim (contiguous),
// ne1=n_tokens, ne2=n_head. For product-aware codebook design: gives Sigma_Q per head offline.
static void cert_dump_q(ggml_backend_cuda_context & ctx, const ggml_tensor * dst) {
    static int qstate = 0; static char qdir[1024] = {0}; static int64_t qcap = 4096;
    static int64_t q_done[160][64] = {};
    if (qstate == 0) {
        const char * e = getenv("TURBO_CERT_DUMP_Q");
        if (!e || !e[0]) { qstate = -1; return; }
        strncpy(qdir, e, sizeof(qdir) - 1);
        if (const char * r = getenv("TURBO_CERT_DUMP_Q_ROWS")) qcap = atoll(r);
        qstate = 1;
        fprintf(stderr, "CERT_DUMP_Q: dir=%s cap=%lld rows per (layer,head)\n", qdir, (long long)qcap);
    }
    if (qstate == -1) return;
    const ggml_tensor * q = dst->src[0];
    if (!q || (q->type != GGML_TYPE_F32 && q->type != GGML_TYPE_F16)) return;
    const char * dash = strrchr(dst->name, '-');
    if (!dash) return;
    const int layer = atoi(dash + 1);
    if (layer < 0 || layer >= 160) return;
    const int64_t hd = q->ne[0], ntok = q->ne[1], nhead = q->ne[2];
    if (hd <= 0 || nhead <= 0 || nhead > 64) return;
    cudaStreamSynchronize(ctx.stream());
    std::vector<float> buf((size_t)hd);
    std::vector<char> raw((size_t)hd * (q->type == GGML_TYPE_F16 ? 2 : 4));
    for (int64_t h = 0; h < nhead; h++) {
        int64_t & done = q_done[layer][h];
        if (done >= qcap) continue;
        char path[1200];
        snprintf(path, sizeof(path), "%s/q_l%03d_h%02d_w%lld.f32", qdir, layer, (int)h, (long long)hd);
        FILE * f = fopen(path, done == 0 ? "wb" : "ab");
        if (!f) { qstate = -1; return; }
        for (int64_t t = 0; t < ntok && done < qcap; t++) {
            const char * src = (const char *)q->data + t * q->nb[1] + h * q->nb[2];
            cudaMemcpy(raw.data(), src, raw.size(), cudaMemcpyDeviceToHost);
            if (q->type == GGML_TYPE_F16) {
                const uint16_t * hh = (const uint16_t *)raw.data();
                for (int64_t j = 0; j < hd; j++) buf[j] = __half2float(*(const __half *)&hh[j]);
            } else {
                memcpy(buf.data(), raw.data(), (size_t)hd * 4);
            }
            fwrite(buf.data(), sizeof(float), (size_t)hd, f);
            done++;
        }
        fclose(f);
    }
}

void ggml_cuda_flash_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_set_device(ctx.device);
    turbo1_tcq_load_cb_fattn();  // E7: turbo1_tcq K/V decode codebooks (TURBO1_TCQ_CB_K/_V, cold-start = turbo2 anchor)

    turbo1_load_dither_fattn();  // E6: one-time TURBO1_DITHER env → fattn decode constants

    cert_dump_q(ctx, dst);

    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    // Turbo prefill: dequant to fp16 and use tensor core MMA for batched attention.
    // turbo4 K uses inverse FWHT during dequant — mixes centroids in float32 shmem before
    // fp16 cast, so precision is fine. turbo2/turbo3 use simple centroid×norm dequant.
    // Set TURBO_PREFILL_VEC=1 to force vec kernel for all turbo types (debug override).
    static const bool turbo_prefill_vec = [] {
        const char * e = getenv("TURBO_PREFILL_VEC");
        if (e) fprintf(stderr, "TURBO_PREFILL_VEC=%s: forcing vec prefill for turbo types\n", e);
        return e != nullptr;
    }();
    const bool turbo_kv = K->type == GGML_TYPE_TURBO2_0 || K->type == GGML_TYPE_TURBO3_0 || K->type == GGML_TYPE_TURBO4_0 || K->type == GGML_TYPE_TURBO8_0 || K->type == GGML_TYPE_TURBO3_TCQ || K->type == GGML_TYPE_TURBO2_TCQ || K->type == GGML_TYPE_TURBO1 || K->type == GGML_TYPE_TURBO1_NSN || K->type == GGML_TYPE_TURBO1_CQ || K->type == GGML_TYPE_TURBO1_TCQ ||
                          V->type == GGML_TYPE_TURBO2_0 || V->type == GGML_TYPE_TURBO3_0 || V->type == GGML_TYPE_TURBO4_0 || V->type == GGML_TYPE_TURBO8_0 || V->type == GGML_TYPE_TURBO3_TCQ || V->type == GGML_TYPE_TURBO2_TCQ || V->type == GGML_TYPE_TURBO1 || V->type == GGML_TYPE_TURBO1_NSN || V->type == GGML_TYPE_TURBO1_CQ || V->type == GGML_TYPE_TURBO1_TCQ;

    // Fused MMA turbo: reads raw turbo bytes directly in the MMA kernel, no intermediate fp16 buffers.
    // Phase 1: turbo4_0 matched K/V at D=128. Set GGML_TURBO_MMA_FUSED=0 to disable.
    static const bool turbo_mma_fused = [] {
        const char * e = getenv("GGML_TURBO_MMA_FUSED");
        if (e && atoi(e) == 0) {
            fprintf(stderr, "GGML_TURBO_MMA_FUSED=0: fused turbo MMA kernel disabled\n");
            return false;
        }
        return true;
    }();
    // turbo1 has no fused MMA-turbo instance (dequant-to-f16 codec only); exclude it so it
    // does not enter the fused path and fall through with no kernel launched.
    const bool turbo_matched = K->type == V->type && turbo_kv && K->type != GGML_TYPE_TURBO1 && K->type != GGML_TYPE_TURBO1_NSN && K->type != GGML_TYPE_TURBO1_CQ && K->type != GGML_TYPE_TURBO1_TCQ;
    const bool turbo1_tcq_matched = K->type == GGML_TYPE_TURBO1_TCQ && V->type == GGML_TYPE_TURBO1_TCQ && Q->ne[0] == 128;
    // Asymmetric fused "q6 sweet spot" (6.124 bpw): turbo8 K + turbo4 V. Only this pair is
    // instantiated, and only at D=256 (dense Qwen geometry). Both stay WHT-rotated like the
    // matched path (Q pre-rotated below, V un-rotated at graph level).
    const bool turbo_fused_asym = K->type == GGML_TYPE_TURBO8_0 && V->type == GGML_TYPE_TURBO4_0 && Q->ne[0] == 256;
    if (turbo_mma_fused && (turbo_matched || turbo_fused_asym || turbo1_tcq_matched) && Q->ne[1] <= 4 &&
        (Q->ne[0] == 128 || Q->ne[0] == 256) &&
        turing_mma_available(ggml_cuda_info().devices[ggml_cuda_get_device()].cc)) {
        cudaStream_t stream = ctx.stream();
        int device;
        CUDA_CHECK(cudaGetDevice(&device));

        // Load TCQ codebooks for fused kernel (constant memory used by inline dequant)
        if (K->type == GGML_TYPE_TURBO3_TCQ || V->type == GGML_TYPE_TURBO3_TCQ) {
            static bool tcq3_fused_loaded[GGML_CUDA_MAX_DEVICES] = {};
            static int tcq_hot_f = -1; if (tcq_hot_f < 0) tcq_hot_f = getenv("TURBO_TCQ_HOTSWAP") ? 1 : 0;
            if (!tcq3_fused_loaded[device] || tcq_hot_f) {
                tcq3_fused_loaded[device] = true;
                turbo_tcq_load_kv_decode();
            }
        }
        if (K->type == GGML_TYPE_TURBO2_TCQ || V->type == GGML_TYPE_TURBO2_TCQ) {
            static bool tcq2_fused_loaded[GGML_CUDA_MAX_DEVICES] = {};
            static int tcq2_hot_f = -1; if (tcq2_hot_f < 0) tcq2_hot_f = getenv("TURBO_TCQ_HOTSWAP") ? 1 : 0;
            if (!tcq2_fused_loaded[device] || tcq2_hot_f) {
                tcq2_fused_loaded[device] = true;
                turbo2_tcq_load_kv_decode();
            }
        }

        // Pre-rotate Q: most fused turbo K types stay in WHT-rotated domain. The turbo1_tcq
        // fused loader emits original-domain K/V after inverse FWHT, matching its unfused path.
        ggml_tensor Q_rot_fused;
        ggml_tensor * orig_q_fused = nullptr;
        const bool fused_k_original_domain = K->type == GGML_TYPE_TURBO1_TCQ;
        if (!fused_k_original_domain && Q->ne[0] % 128 == 0) {
            const size_t q_size = ggml_nelements(Q) * sizeof(float);
            if (q_size > q_rot_buf_size[device]) {
                if (q_rot_buf[device]) CUDA_CHECK(cudaFree(q_rot_buf[device]));
                CUDA_CHECK(cudaMalloc(&q_rot_buf[device], q_size));
                q_rot_buf_size[device] = q_size;
            }
            const int64_t n_q_groups = ggml_nelements(Q) / 128;
            k_turbo_fwht_forward<<<(int)n_q_groups, 128, 0, stream>>>(
                (const float *)Q->data, q_rot_buf[device], ggml_nelements(Q));
            Q_rot_fused = *Q;
            Q_rot_fused.data = q_rot_buf[device];
            orig_q_fused = dst->src[0];
            dst->src[0] = &Q_rot_fused;
        }

        // TCQ decode alpha for V dequant
        if (V->type == GGML_TYPE_TURBO3_TCQ || V->type == GGML_TYPE_TURBO2_TCQ || V->type == GGML_TYPE_TURBO1_TCQ) {
            load_tcq_decode_alpha(device);
            if (d_tcq_decode_alpha_v_static == 0.0f) {
                float alpha = tcq_compute_alpha_v(V->type, V->ne[1]);
                cudaMemcpyToSymbol(d_tcq_decode_alpha_v_fattn, &alpha, sizeof(float));
            }
        }

#define TURBO_FUSED_DISPATCH(tK, tV) \
        if (K->type == tK && V->type == tV) { \
            if (Q->ne[0] == 128) \
                ggml_cuda_flash_attn_ext_mma_turbo_switch_ncols2<128, 128, tK, tV>(ctx, dst); \
            else \
                ggml_cuda_flash_attn_ext_mma_turbo_switch_ncols2<256, 256, tK, tV>(ctx, dst); \
        }
        // Asymmetric "q6 sweet spot" (6.124 bpw): turbo8 K + turbo4 V, D=256 only (no D=128 instance).
        if (K->type == GGML_TYPE_TURBO8_0 && V->type == GGML_TYPE_TURBO4_0) {
            ggml_cuda_flash_attn_ext_mma_turbo_switch_ncols2<256, 256, GGML_TYPE_TURBO8_0, GGML_TYPE_TURBO4_0>(ctx, dst);
        }
        else TURBO_FUSED_DISPATCH(GGML_TYPE_TURBO4_0,   GGML_TYPE_TURBO4_0)
        else TURBO_FUSED_DISPATCH(GGML_TYPE_TURBO8_0,   GGML_TYPE_TURBO8_0)
        else TURBO_FUSED_DISPATCH(GGML_TYPE_TURBO3_TCQ, GGML_TYPE_TURBO3_TCQ)
        else TURBO_FUSED_DISPATCH(GGML_TYPE_TURBO2_TCQ, GGML_TYPE_TURBO2_TCQ)
        else if (K->type == GGML_TYPE_TURBO1_TCQ && V->type == GGML_TYPE_TURBO1_TCQ) {
            ggml_cuda_flash_attn_ext_mma_turbo_switch_ncols2<128, 128, GGML_TYPE_TURBO1_TCQ, GGML_TYPE_TURBO1_TCQ>(ctx, dst);
        }
        else TURBO_FUSED_DISPATCH(GGML_TYPE_TURBO3_0,   GGML_TYPE_TURBO3_0)
        else TURBO_FUSED_DISPATCH(GGML_TYPE_TURBO2_0,   GGML_TYPE_TURBO2_0)
#undef TURBO_FUSED_DISPATCH

        if (orig_q_fused) dst->src[0] = orig_q_fused;
        return;
    }

    if (turbo_kv && !turbo_prefill_vec && Q->ne[1] > 1 && Q->ne[0] <= 256 && turing_mma_available(ggml_cuda_info().devices[ggml_cuda_get_device()].cc)) {
        // Prefill path: turbo4 K uses inverse FWHT dequant (original domain, no Q rotation),
        // turbo2/3 K uses simple dequant (rotated domain, Q pre-rotated). V un-rotation at graph level.
        ggml_cuda_turbo_prefill_attend(ctx, dst);
    } else {
        load_tcq_decode_alpha(ctx.device);

        // Update VEC __constant__ alpha for context-adaptive mode
        if (d_tcq_decode_alpha_v_static == 0.0f &&
            (V->type == GGML_TYPE_TURBO3_TCQ || V->type == GGML_TYPE_TURBO2_TCQ)) {
            float alpha = tcq_compute_alpha_v(V->type, V->ne[1]);
            cudaMemcpyToSymbol(d_tcq_decode_alpha_v_fattn, &alpha, sizeof(float));
        }

        // Load runtime codebooks for TCQ types (needed by both dequant and native VEC paths)
        if (K->type == GGML_TYPE_TURBO3_TCQ || V->type == GGML_TYPE_TURBO3_TCQ) {
            static bool tcq3_cb_loaded[GGML_CUDA_MAX_DEVICES] = {};
            if (!tcq3_cb_loaded[ctx.device]) {
                tcq3_cb_loaded[ctx.device] = true;
                turbo_tcq_load_kv_decode();
            }
        }
        if (K->type == GGML_TYPE_TURBO2_TCQ || V->type == GGML_TYPE_TURBO2_TCQ) {
            static bool tcq2_cb_loaded[GGML_CUDA_MAX_DEVICES] = {};
            static int tcq2_hot_d = -1; if (tcq2_hot_d < 0) tcq2_hot_d = getenv("TURBO_TCQ_HOTSWAP") ? 1 : 0;
            if (!tcq2_cb_loaded[ctx.device] || tcq2_hot_d) {
                tcq2_cb_loaded[ctx.device] = true;
                turbo2_tcq_load_kv_decode();
            }
        }

        cudaStream_t stream = ctx.stream();

        // Dequant turbo K/V to fp16 for decode: MMA tensor cores on fp16 beat VEC scalar
        // on turbo bits for dense models. Bandwidth savings from turbo's 3/16 footprint are
        // negligible relative to FFN compute (~1% slower native on Qwen3.5-27B, 3090).
        // Set GGML_TURBO_DECODE_NATIVE=1 to force native VEC path (may help bandwidth-limited configs).
        static const bool turbo_decode_native = (getenv("GGML_TURBO_DECODE_NATIVE") != nullptr);
        // Dequant turbo K/V to fp16 for D<=256 (any K/V combo), or D=512 only when BOTH
        // K and V are turbo (Gemma 4 ISWA global layers with K=V — Bug A2). Mixed q8_0+turbo
        // at D=512 stays on native VEC path because post-dequant q8_0 K + f16 V has no
        // working VEC template at D=512.
        const bool turbo_k_only = K->type == GGML_TYPE_TURBO2_0 || K->type == GGML_TYPE_TURBO3_0 || K->type == GGML_TYPE_TURBO4_0 || K->type == GGML_TYPE_TURBO8_0 || K->type == GGML_TYPE_TURBO3_TCQ || K->type == GGML_TYPE_TURBO2_TCQ || K->type == GGML_TYPE_TURBO1 || K->type == GGML_TYPE_TURBO1_NSN || K->type == GGML_TYPE_TURBO1_CQ || K->type == GGML_TYPE_TURBO1_TCQ;
        const bool turbo_v_only = V->type == GGML_TYPE_TURBO2_0 || V->type == GGML_TYPE_TURBO3_0 || V->type == GGML_TYPE_TURBO4_0 || V->type == GGML_TYPE_TURBO8_0 || V->type == GGML_TYPE_TURBO3_TCQ || V->type == GGML_TYPE_TURBO2_TCQ || V->type == GGML_TYPE_TURBO1 || V->type == GGML_TYPE_TURBO1_NSN || V->type == GGML_TYPE_TURBO1_CQ || V->type == GGML_TYPE_TURBO1_TCQ;
        // Mixed f16/q8_0 + turbo at D=512: dequant K (and turbo V) to f16 so FA dispatches as
        // F16/F16 D=512 (which exists). Without this, the native VEC templates needed are
        // either missing (F16↔turbo) or have buggy SASS on sm_120 PTX-JIT (Q8_0↔turbo4 etc).
        const bool k_is_f16_q8_or_turbo = (K->type == GGML_TYPE_F16) || (K->type == GGML_TYPE_Q8_0) || (K->type == GGML_TYPE_BF16) || turbo_k_only;
        const bool v_is_f16_q8_or_turbo = (V->type == GGML_TYPE_F16) || (V->type == GGML_TYPE_Q8_0) || (V->type == GGML_TYPE_BF16) || turbo_v_only;
        const bool both_dequantable_512 = k_is_f16_q8_or_turbo && v_is_f16_q8_or_turbo;
        // turbo8 has no native VEC kernel (no FATTN_VEC_CASES registration) — it is a
        // dequant-to-f16-only codec. GGML_TURBO_DECODE_NATIVE must not route it to the native
        // VEC path (would hit BEST_FATTN_KERNEL_VEC → unregistered template → GGML_ABORT), so
        // force dequant whenever turbo8 is on either side, regardless of the override. Other
        // turbo types do have native VEC kernels and honor the override.
        // turbo8 and turbo1 have no native VEC kernel (dequant-to-f16 only) — force dequant even
        // under GGML_TURBO_DECODE_NATIVE so we never route them to an unregistered VEC template.
        const bool turbo8_involved = K->type == GGML_TYPE_TURBO8_0 || V->type == GGML_TYPE_TURBO8_0 ||
                                     K->type == GGML_TYPE_TURBO1   || V->type == GGML_TYPE_TURBO1 ||
                                     K->type == GGML_TYPE_TURBO1_NSN || K->type == GGML_TYPE_TURBO1_CQ || K->type == GGML_TYPE_TURBO1_TCQ || V->type == GGML_TYPE_TURBO1_NSN || V->type == GGML_TYPE_TURBO1_CQ || V->type == GGML_TYPE_TURBO1_TCQ;
        const bool do_decode_dequant = (!turbo_decode_native || turbo8_involved) && turbo_kv && (Q->ne[0] <= 256 || (Q->ne[0] <= 512 && both_dequantable_512));

        half * k_fp16_dec = nullptr;
        half * v_fp16_dec = nullptr;
        ggml_tensor K_f16_dec, V_f16_dec;
        ggml_tensor * orig_k_decode = nullptr;
        ggml_tensor * orig_v_decode = nullptr;

        int device_dec;
        CUDA_CHECK(cudaGetDevice(&device_dec));

        if (do_decode_dequant) {
            // Mixed q8_0 + turbo: when one side is turbo (dequanted to f16) and the other is
            // q8_0, the FA kernel gets a (Q8_0, F16) pair, which produces garbage at D<=256
            // (no correct mixed VEC combo). Dequant the q8_0 side to f16 too so the pair is
            // (F16, F16) — matching the symmetric turbo path that goes through the same f16 FA.
            const bool k_needs_dequant = turbo_k_only || ((K->type == GGML_TYPE_Q8_0 || K->type == GGML_TYPE_BF16) && (Q->ne[0] > 256 || turbo_v_only));
            const bool v_needs_dequant = turbo_v_only || ((V->type == GGML_TYPE_Q8_0 || V->type == GGML_TYPE_BF16) && (Q->ne[0] > 256 || turbo_k_only));
            if (k_needs_dequant) {
                // Size the dequant buffer for the FULL cache (kv_size from the underlying root
                // tensor), not just the current n_kv. This prevents per-token reallocations as
                // the cache fills, which would invalidate any in-flight CUDA graph capture
                // pointing at the old device pointer (defensive against PR #21635-class bugs).
                // The first call sizes the buffer for the worst-case cache; subsequent calls
                // (including layers with smaller caches) reuse the same allocation.
                const ggml_tensor * k_root = K;
                while (k_root->view_src) k_root = k_root->view_src;
                const size_t k_max_bytes = (size_t) ggml_nelements(k_root) * sizeof(half);
                if (k_max_bytes > kv_dequant_k_buf_size[device_dec]) {
                    if (kv_dequant_k_buf[device_dec]) CUDA_CHECK(cudaFree(kv_dequant_k_buf[device_dec]));
                    CUDA_CHECK(cudaMalloc(&kv_dequant_k_buf[device_dec], k_max_bytes));
                    kv_dequant_k_buf_size[device_dec] = k_max_bytes;
                }
                k_fp16_dec = kv_dequant_k_buf[device_dec];
                // K dequant to fp16 in ORIGINAL (unrotated) domain via inverse FWHT.
                // All turbo K types use inv-FWHT kernels so K matches native f16/q8_0 layout
                // and Q stays unrotated. This mirrors the prefill path's encode→decode chain
                // and is the only path that works on Gemma 4 ISWA + K=V global layers.
                //
                // Bug #31 exception: K=turbo2 inv-FWHT decode produces correct values for V in
                // {turbo2, *_tcq} but a (still-undiagnosed) divergence with V in {turbo3, turbo4,
                // q8_0, f16} on Gemma 4 26B-A4B (degenerate single-token output, attention scores
                // collapse). The PREFILL path uses the rotated-domain kernel + Q rotation for
                // turbo2 K and works for every V type. Mirror that here for the failing V types.
                const bool k_t2_use_rotated = (K->type == GGML_TYPE_TURBO2_0) &&
                    (V->type == GGML_TYPE_TURBO3_0 || V->type == GGML_TYPE_TURBO4_0 ||
                     V->type == GGML_TYPE_Q8_0    || V->type == GGML_TYPE_F16);
                const bool k_t3_use_rotated = (K->type == GGML_TYPE_TURBO3_0) &&
                    (V->type == GGML_TYPE_TURBO2_0);
                dim3 grid_k(K->ne[1], K->ne[2], K->ne[3]);
                if (K->type == GGML_TYPE_TURBO2_0 && k_t2_use_rotated) {
                    // Rotated-domain dequant: K stays in WHT-rotated space; Q is pre-rotated below.
                    k_turbo2_dequant_f16<<<grid_k, K->ne[0], 0, stream>>>(
                        (const char *)K->data, k_fp16_dec, K->ne[0], K->ne[1], K->ne[2], K->nb[1], K->nb[2], K->nb[3]);
                } else if (K->type == GGML_TYPE_TURBO2_0) {
                    k_turbo2_dequant_f16_inv_fwht<<<grid_k, 128, 0, stream>>>(
                        (const char *)K->data, k_fp16_dec, K->ne[0], K->ne[1], K->ne[2], K->nb[1], K->nb[2], K->nb[3]);
                } else if (K->type == GGML_TYPE_TURBO3_0 && k_t3_use_rotated) {
                    // Rotated-domain dequant for K=t3 + V=t2 (same Bug #31 pattern, V side).
                    k_turbo3_dequant_f16<<<grid_k, K->ne[0], 0, stream>>>(
                        (const char *)K->data, k_fp16_dec, K->ne[0], K->ne[1], K->ne[2], K->nb[1], K->nb[2], K->nb[3]);
                } else if (K->type == GGML_TYPE_TURBO3_0) {
                    k_turbo3_dequant_f16_inv_fwht<<<grid_k, 128, 0, stream>>>(
                        (const char *)K->data, k_fp16_dec, K->ne[0], K->ne[1], K->ne[2], K->nb[1], K->nb[2], K->nb[3]);
                } else if (K->type == GGML_TYPE_TURBO4_0) {
                    k_turbo4_dequant_f16_inv_fwht<<<grid_k, 128, 0, stream>>>(
                        (const char *)K->data, k_fp16_dec, K->ne[0], K->ne[1], K->ne[2], K->nb[1], K->nb[2], K->nb[3]);
                } else if (K->type == GGML_TYPE_TURBO8_0) {
                    k_turbo8_dequant_f16_inv_fwht<<<grid_k, 128, 0, stream>>>(
                        (const char *)K->data, k_fp16_dec, K->ne[0], K->ne[1], K->ne[2], K->nb[1], K->nb[2], K->nb[3]);
                } else if (K->type == GGML_TYPE_TURBO1) {
                    k_turbo1_dequant_f16_inv_fwht<<<grid_k, 128, 0, stream>>>(
                        (const char *)K->data, k_fp16_dec, K->ne[0], K->ne[1], K->ne[2], K->nb[1], K->nb[2], K->nb[3]);
                } else if (K->type == GGML_TYPE_TURBO1_CQ) {
                    k_turbo1_cq_dequant_f16<<<grid_k, 128, 0, stream>>>(
                        (const char *)K->data, k_fp16_dec, K->ne[0], K->ne[1], K->ne[2], K->nb[1], K->nb[2], K->nb[3]);
                } else if (K->type == GGML_TYPE_TURBO1_TCQ) {
                    k_turbo1_tcq_dequant_f16<<<grid_k, 128, 0, stream>>>(
                        (const char *)K->data, k_fp16_dec, K->ne[0], K->ne[1], K->ne[2], K->nb[1], K->nb[2], K->nb[3], d_tcq_decode_alpha_k, 0);
                } else if (K->type == GGML_TYPE_TURBO1_NSN) {
                    const int kl_nsn = atoi(K->name + 9);
                    const float * o_k = (kl_nsn >= 0 && kl_nsn < PFHEAD_MAX_L)
                        ? turbo1_nsn_o_buf(device_dec, 1) + (size_t) kl_nsn * PFHEAD_MAX_C : nullptr;
                    k_turbo1_nsn_dequant_f16<<<grid_k, 128, 0, stream>>>(
                        (const char *)K->data, k_fp16_dec, K->ne[0], K->ne[1], K->ne[2], K->nb[1], K->nb[2], K->nb[3], o_k);
                } else if (K->type == GGML_TYPE_TURBO3_TCQ) {
                    k_turbo3_tcq_dequant_f16_inv_fwht<<<grid_k, 128, 0, stream>>>(
                        (const char *)K->data, k_fp16_dec, K->ne[0], K->ne[1], K->ne[2], K->nb[1], K->nb[2], K->nb[3], d_tcq_decode_alpha_k);
                } else if (K->type == GGML_TYPE_TURBO2_TCQ) {
                    k_turbo2_tcq_dequant_f16_inv_fwht<<<grid_k, 128, 0, stream>>>(
                        (const char *)K->data, k_fp16_dec, K->ne[0], K->ne[1], K->ne[2], K->nb[1], K->nb[2], K->nb[3], d_tcq_decode_alpha_k);
                } else if (K->type == GGML_TYPE_Q8_0) {
                    // Q8_0 K dequant: only fires at D=512 when V is turbo (no F16/Q8_0 D=512
                    // template, and (Q8_0, TURBO4_0) D=512 has buggy SASS on sm_120 PTX-JIT).
                    // Output goes into TKHE layout matching V_f16_dec → dispatches as F16/F16 D=512.
                    k_q8_0_dequant_f16_tkhe<<<grid_k, K->ne[0], 0, stream>>>(
                        (const char *)K->data, k_fp16_dec, K->ne[0], K->ne[1], K->ne[2], K->nb[1], K->nb[2], K->nb[3]);
                } else if (K->type == GGML_TYPE_BF16) {
                    // bf16 K cast to f16 (mixed bf16-K + turbo-V) → dispatches as F16/F16.
                    k_bf16_to_f16_tkhe<<<grid_k, K->ne[0], 0, stream>>>(
                        (const char *)K->data, k_fp16_dec, K->ne[0], K->ne[1], K->ne[2], K->nb[1], K->nb[2], K->nb[3]);
                }
                K_f16_dec = *K;
                K_f16_dec.type = GGML_TYPE_F16;
                K_f16_dec.data = k_fp16_dec;
                K_f16_dec.nb[0] = sizeof(half);
                K_f16_dec.nb[1] = K->ne[0] * K->ne[2] * sizeof(half);  // row stride: head_dim * n_head_kv (matches native cache)
                K_f16_dec.nb[2] = K->ne[0] * sizeof(half);             // head stride: head_dim (matches native cache)
                K_f16_dec.nb[3] = K->ne[0] * K->ne[1] * K->ne[2] * sizeof(half);
                orig_k_decode = dst->src[1];
                dst->src[1] = &K_f16_dec;
            }
            if (v_needs_dequant) {
                // Same kv_size-based sizing as K above — see comment there.
                const ggml_tensor * v_root = V;
                while (v_root->view_src) v_root = v_root->view_src;
                const size_t v_max_bytes = (size_t) ggml_nelements(v_root) * sizeof(half);
                if (v_max_bytes > kv_dequant_v_buf_size[device_dec]) {
                    if (kv_dequant_v_buf[device_dec]) CUDA_CHECK(cudaFree(kv_dequant_v_buf[device_dec]));
                    CUDA_CHECK(cudaMalloc(&kv_dequant_v_buf[device_dec], v_max_bytes));
                    kv_dequant_v_buf_size[device_dec] = v_max_bytes;
                }
                v_fp16_dec = kv_dequant_v_buf[device_dec];
                // V dequant to fp16. All turbo V stays in rotated domain — the graph-level
                // ggml_turbo_wht inverse op (added in build_attn) un-rotates the attention output.
                dim3 grid_v(V->ne[1], V->ne[2], V->ne[3]);
                if (V->type == GGML_TYPE_TURBO2_0) {
                    k_turbo2_dequant_f16<<<grid_v, V->ne[0], 0, stream>>>(
                        (const char *)V->data, v_fp16_dec, V->ne[0], V->ne[1], V->ne[2], V->nb[1], V->nb[2], V->nb[3]);
                } else if (V->type == GGML_TYPE_TURBO3_TCQ) {
                    k_turbo3_tcq_dequant_f16<<<grid_v, V->ne[0], 0, stream>>>(
                        (const char *)V->data, v_fp16_dec, V->ne[0], V->ne[1], V->ne[2], V->nb[1], V->nb[2], V->nb[3], tcq_compute_alpha_v(V->type, V->ne[1]), 1);
                } else if (V->type == GGML_TYPE_TURBO2_TCQ) {
                    k_turbo2_tcq_dequant_f16<<<grid_v, V->ne[0], 0, stream>>>(
                        (const char *)V->data, v_fp16_dec, V->ne[0], V->ne[1], V->ne[2], V->nb[1], V->nb[2], V->nb[3], tcq_compute_alpha_v(V->type, V->ne[1]), 1);
                } else if (V->type == GGML_TYPE_TURBO4_0) {
                    k_turbo4_dequant_f16<<<grid_v, V->ne[0], 0, stream>>>(
                        (const char *)V->data, v_fp16_dec, V->ne[0], V->ne[1], V->ne[2], V->nb[1], V->nb[2], V->nb[3]);
                } else if (V->type == GGML_TYPE_TURBO8_0) {
                    k_turbo8_dequant_f16<<<grid_v, V->ne[0], 0, stream>>>(
                        (const char *)V->data, v_fp16_dec, V->ne[0], V->ne[1], V->ne[2], V->nb[1], V->nb[2], V->nb[3]);
                } else if (V->type == GGML_TYPE_TURBO1) {
                    k_turbo1_dequant_f16_inv_fwht<<<grid_v, 128, 0, stream>>>(
                        (const char *)V->data, v_fp16_dec, V->ne[0], V->ne[1], V->ne[2], V->nb[1], V->nb[2], V->nb[3]);
                } else if (V->type == GGML_TYPE_TURBO1_CQ) {
                    k_turbo1_cq_dequant_f16<<<grid_v, 128, 0, stream>>>(
                        (const char *)V->data, v_fp16_dec, V->ne[0], V->ne[1], V->ne[2], V->nb[1], V->nb[2], V->nb[3]);
                } else if (V->type == GGML_TYPE_TURBO1_TCQ) {
                    k_turbo1_tcq_dequant_f16<<<grid_v, 128, 0, stream>>>(
                        (const char *)V->data, v_fp16_dec, V->ne[0], V->ne[1], V->ne[2], V->nb[1], V->nb[2], V->nb[3], tcq_compute_alpha_v(V->type, V->ne[1]), 1);
                } else if (V->type == GGML_TYPE_TURBO1_NSN) {
                    const int vl_nsn = atoi(V->name + 9);
                    const float * o_v = (vl_nsn >= 0 && vl_nsn < PFHEAD_MAX_L)
                        ? turbo1_nsn_o_buf(device_dec, 0) + (size_t) vl_nsn * PFHEAD_MAX_C : nullptr;
                    k_turbo1_nsn_dequant_f16<<<grid_v, 128, 0, stream>>>(
                        (const char *)V->data, v_fp16_dec, V->ne[0], V->ne[1], V->ne[2], V->nb[1], V->nb[2], V->nb[3], o_v);
                } else if (V->type == GGML_TYPE_TURBO3_0) {
                    k_turbo3_dequant_f16<<<grid_v, V->ne[0], 0, stream>>>(
                        (const char *)V->data, v_fp16_dec, V->ne[0], V->ne[1], V->ne[2], V->nb[1], V->nb[2], V->nb[3]);
                } else if (V->type == GGML_TYPE_Q8_0) {
                    // Q8_0 V dequant: only fires at D=512 when K is turbo (mirror of K=Q8_0 path).
                    k_q8_0_dequant_f16_tkhe<<<grid_v, V->ne[0], 0, stream>>>(
                        (const char *)V->data, v_fp16_dec, V->ne[0], V->ne[1], V->ne[2], V->nb[1], V->nb[2], V->nb[3]);
                } else if (V->type == GGML_TYPE_BF16) {
                    // bf16 V cast to f16 (mixed turbo-K + bf16-V) → dispatches as F16/F16.
                    k_bf16_to_f16_tkhe<<<grid_v, V->ne[0], 0, stream>>>(
                        (const char *)V->data, v_fp16_dec, V->ne[0], V->ne[1], V->ne[2], V->nb[1], V->nb[2], V->nb[3]);
                }
                V_f16_dec = *V;
                V_f16_dec.type = GGML_TYPE_F16;
                V_f16_dec.data = v_fp16_dec;
                V_f16_dec.nb[0] = sizeof(half);
                V_f16_dec.nb[1] = V->ne[0] * V->ne[2] * sizeof(half);  // row stride: head_dim * n_head_kv (matches native cache)
                V_f16_dec.nb[2] = V->ne[0] * sizeof(half);             // head stride: head_dim (matches native cache)
                V_f16_dec.nb[3] = V->ne[0] * V->ne[1] * V->ne[2] * sizeof(half);
                orig_v_decode = dst->src[2];
                dst->src[2] = &V_f16_dec;
                // Bug A1: nvcc 13 on sm_120a reorders these V_f16_dec.nb[*] stores past the FA
                // dispatcher → stale strides → <unused49> garbage. signal_fence is a pure
                // host-compiler barrier (zero machine instructions).
                std::atomic_signal_fence(std::memory_order_seq_cst);
            }
        }

        // Pre-rotate Q for turbo K stored in rotated domain.
        // When do_decode_dequant fires, all turbo K types are dequanted via inv-FWHT into
        // ORIGINAL domain → Q stays unrotated. When decode dequant is skipped (D>256 or
        // GGML_TURBO_DECODE_NATIVE), turbo K is consumed by the native vec turbo dot product,
        // which expects a pre-rotated Q — so rotate Q in that case.
        ggml_tensor Q_rot_decode;
        ggml_tensor * orig_q_decode = nullptr;
        const bool turbo_k_any = (K->type == GGML_TYPE_TURBO2_0 || K->type == GGML_TYPE_TURBO3_0 || K->type == GGML_TYPE_TURBO4_0 || K->type == GGML_TYPE_TURBO8_0 || K->type == GGML_TYPE_TURBO3_TCQ || K->type == GGML_TYPE_TURBO2_TCQ);
        // Bug #31 exception: when K=turbo2/turbo3 dequant fell back to the rotated kernel (see K
        // dispatch above), K is in WHT-rotated space, not original space, so Q must be pre-rotated.
        const bool k_uses_rotated_path = do_decode_dequant && (
            ((K->type == GGML_TYPE_TURBO2_0) &&
             (V->type == GGML_TYPE_TURBO3_0 || V->type == GGML_TYPE_TURBO4_0 ||
              V->type == GGML_TYPE_Q8_0    || V->type == GGML_TYPE_F16)) ||
            ((K->type == GGML_TYPE_TURBO3_0) && (V->type == GGML_TYPE_TURBO2_0)));
        const bool turbo_k_in_orig_domain = do_decode_dequant && turbo_k_any && !k_uses_rotated_path;
        if (turbo_k_any && !turbo_k_in_orig_domain && Q->ne[0] % 128 == 0) {
            const size_t q_size = ggml_nelements(Q) * sizeof(float);
            if (q_size > q_rot_buf_size[device_dec]) {
                if (q_rot_buf[device_dec]) CUDA_CHECK(cudaFree(q_rot_buf[device_dec]));
                CUDA_CHECK(cudaMalloc(&q_rot_buf[device_dec], q_size));
                q_rot_buf_size[device_dec] = q_size;
            }
            const int64_t n_q_groups = ggml_nelements(Q) / 128;
            k_turbo_fwht_forward<<<(int)n_q_groups, 128, 0, stream>>>(
                (const float *)Q->data, q_rot_buf[device_dec], ggml_nelements(Q));
            Q_rot_decode = *Q;
            Q_rot_decode.data = q_rot_buf[device_dec];
            orig_q_decode = dst->src[0];
            dst->src[0] = &Q_rot_decode;
        }

        switch (ggml_cuda_get_best_fattn_kernel(ggml_cuda_get_device(), dst)) {
            case BEST_FATTN_KERNEL_NONE:
                GGML_ABORT("fatal error");
            case BEST_FATTN_KERNEL_TILE:
                ggml_cuda_flash_attn_ext_tile(ctx, dst);
                break;
            case BEST_FATTN_KERNEL_VEC:
                ggml_cuda_flash_attn_ext_vec(ctx, dst);
                break;
            case BEST_FATTN_KERNEL_WMMA_F16:
                ggml_cuda_flash_attn_ext_wmma_f16(ctx, dst);
                break;
            case BEST_FATTN_KERNEL_MMA_F16:
                ggml_cuda_flash_attn_ext_mma_f16(ctx, dst);
                break;
        }

        if (orig_q_decode) dst->src[0] = orig_q_decode;
        if (orig_k_decode) dst->src[1] = orig_k_decode;
        if (orig_v_decode) dst->src[2] = orig_v_decode;
        // K/V fp16 buffers are persistent (grow-only), no free needed
    }

    // Output inverse rotation for turbo V types is handled at graph level
    // (ggml_turbo_wht op in llama-graph.cpp) to maintain CUDA graph compatibility.
}

bool ggml_cuda_flash_attn_ext_supported(int device, const ggml_tensor * dst) {
    return ggml_cuda_get_best_fattn_kernel(device, dst) != BEST_FATTN_KERNEL_NONE;
}
