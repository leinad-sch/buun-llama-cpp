#include "set-rows.cuh"
#include "cpy-utils.cuh"
#include "turbo-quant-cuda.cuh"
#include "turbo-tcq-alpha.cuh"
#include <cstring>
#include <cerrno>
#include <cctype>
#include <string>
#include <vector>
#if defined(_WIN32) && !defined(strtok_r)
#define strtok_r strtok_s
#endif
// Definition for the extern in turbo-quant-cuda.cuh. When true, the encode mean-sub tap is skipped
// (VBR transcode re-encode: its input is already stored-domain V - mu_V).
bool g_turbo_meansub_suppress = false;
// set by the InnerQ calibration when non-identity scales arm: the fused turbo1_tcq decode is
// not scale-aware, so the fattn dispatch routes t1 to the materialize path while calibrated
bool g_turbo_innerq_calibrated = false;

static void load_turbo4_alpha(int device) {
    static bool loaded[GGML_CUDA_MAX_DEVICES] = {};
    if (loaded[device]) return;
    loaded[device] = true;
    const char *s = getenv("TURBO4_ALPHA");
    if (!s) return;
    char *end;
    errno = 0;
    float a = strtof(s, &end);
    if (end == s || errno != 0 || a <= 0.0f || a >= 10.0f) {
        fprintf(stderr, "TURBO4: invalid TURBO4_ALPHA='%s'\n", s);
    } else {
        cudaMemcpyToSymbol(d_turbo4_alpha, &a, sizeof(float));
        fprintf(stderr, "TURBO4: alpha=%.3f (device %d)\n", a, device);
    }
}

static void load_tcq_norm_alpha(int device) {
    static bool loaded[GGML_CUDA_MAX_DEVICES] = {};
    if (loaded[device]) return;
    loaded[device] = true;

    // Context-adaptive decode-time alpha is the default. Force encode-time V alpha to 1.0
    // unless TURBO_TCQ_ENCODE_ALPHA=1 is explicitly set to use encode-time alpha instead.
    const char *encode_mode = getenv("TURBO_TCQ_ENCODE_ALPHA");
    if (!encode_mode) {
        float one = 1.0f;
        cudaMemcpyToSymbol(d_tcq_norm_alpha_v, &one, sizeof(float));
        if (device == 0) fprintf(stderr, "TCQ: encode V alpha=1.0 (context-adaptive decode-time alpha active)\n");
        // Still allow K alpha override
        const char *s = getenv("TURBO_TCQ_ALPHA");
        if (s) {
            char *end;
            errno = 0;
            float a = strtof(s, &end);
            if (end != s && errno == 0 && a > 0.0f && a < 10.0f) {
                cudaMemcpyToSymbol(d_tcq_norm_alpha, &a, sizeof(float));
            }
        }
        return;
    }

    const char *s = getenv("TURBO_TCQ_ALPHA");
    const char *sv = getenv("TURBO_TCQ_ALPHA_V");
    if (!s && !sv) return;
    float alpha_k = 1.0f;
    bool k_set = false;
    if (s) {
        char *end;
        errno = 0;
        float a = strtof(s, &end);
        if (end == s || errno != 0 || a <= 0.0f || a >= 10.0f) {
            fprintf(stderr, "TCQ: invalid TURBO_TCQ_ALPHA='%s'\n", s);
        } else {
            alpha_k = a;
            k_set = true;
            cudaMemcpyToSymbol(d_tcq_norm_alpha, &alpha_k, sizeof(float));
        }
    }
    if (sv) {
        char *end;
        errno = 0;
        float a = strtof(sv, &end);
        if (end == sv || errno != 0 || a <= 0.0f || a >= 10.0f) {
            fprintf(stderr, "TCQ: invalid TURBO_TCQ_ALPHA_V='%s'\n", sv);
        } else {
            cudaMemcpyToSymbol(d_tcq_norm_alpha_v, &a, sizeof(float));
            fprintf(stderr, "TCQ: norm alpha K=%.3f V=%.3f\n", alpha_k, a);
            return;
        }
    }
    // TURBO_TCQ_ALPHA set but not TURBO_TCQ_ALPHA_V: V matches K for backwards compat
    if (k_set) {
        cudaMemcpyToSymbol(d_tcq_norm_alpha_v, &alpha_k, sizeof(float));
        fprintf(stderr, "TCQ: norm alpha K=V=%.3f\n", alpha_k);
    }
}

// TCQ error dump for autocorrelation analysis (TURBO_TCQ_DUMP_ERRORS=N)
// Only active on the first device that triggers it (diagnostic tool, not perf-critical)
static int    tcq_dump_n = 0;
static int    tcq_dump_device = -1;
static float * tcq_dump_x_host = nullptr;
static uint8_t * tcq_dump_out_host = nullptr;
static float * tcq_dump_x_dev = nullptr;
static uint8_t * tcq_dump_out_dev = nullptr;

static void tcq_error_dump_flush() {
    if (tcq_dump_n == 0 || tcq_dump_device < 0) return;
    ggml_cuda_set_device(tcq_dump_device);
    cudaMemcpy(tcq_dump_x_host, tcq_dump_x_dev, tcq_dump_n * 128 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(tcq_dump_out_host, tcq_dump_out_dev, tcq_dump_n * 128 * sizeof(uint8_t), cudaMemcpyDeviceToHost);
    FILE * f = fopen("/tmp/tcq_errors.bin", "wb");
    if (f) {
        int32_t header[1] = { tcq_dump_n };
        fwrite(header, sizeof(int32_t), 1, f);
        fwrite(tcq_dump_x_host, sizeof(float), tcq_dump_n * 128, f);
        fwrite(tcq_dump_out_host, sizeof(uint8_t), tcq_dump_n * 128, f);
        fclose(f);
        fprintf(stderr, "TCQ: dumped %d groups to /tmp/tcq_errors.bin\n", tcq_dump_n);
    }
    cudaFree(tcq_dump_x_dev);
    cudaFree(tcq_dump_out_dev);
    free(tcq_dump_x_host);
    free(tcq_dump_out_host);
}

static void init_tcq_error_dump(int device) {
    static bool loaded[GGML_CUDA_MAX_DEVICES] = {};
    if (loaded[device]) return;
    loaded[device] = true;
    const char *s = getenv("TURBO_TCQ_DUMP_ERRORS");
    if (!s) return;
    int n = atoi(s);
    if (n <= 0 || n > 500000) return;
    // Only allocate dump buffers on the first device that requests them
    if (tcq_dump_device >= 0) {
        // Already allocated on another device — just set the symbol pointers for this device
        cudaMemcpyToSymbol(d_tcq_dump_max, &n, sizeof(int));
        return;
    }
    tcq_dump_n = n;
    tcq_dump_device = device;
    tcq_dump_x_host = (float *)malloc(n * 128 * sizeof(float));
    tcq_dump_out_host = (uint8_t *)malloc(n * 128 * sizeof(uint8_t));
    cudaMalloc(&tcq_dump_x_dev, n * 128 * sizeof(float));
    cudaMalloc(&tcq_dump_out_dev, n * 128 * sizeof(uint8_t));
    cudaMemcpyToSymbol(d_tcq_dump_x_buf, &tcq_dump_x_dev, sizeof(float*));
    cudaMemcpyToSymbol(d_tcq_dump_out_buf, &tcq_dump_out_dev, sizeof(uint8_t*));
    cudaMemcpyToSymbol(d_tcq_dump_max, &n, sizeof(int));
    atexit(tcq_error_dump_flush);
    fprintf(stderr, "TCQ: will dump errors for first %d groups to /tmp/tcq_errors.bin\n", n);
}

typedef void (*set_rows_kernel_t)(const char * src, char * dst);

// Generic quantized set_rows kernel template
template <typename idx_t, typename block_type, int qk, void (*quantize_func)(const float *, block_type *)>
static __global__ void k_set_rows_quant(const float * __restrict__ src0,
                                        const idx_t * __restrict__ src1,
                                        block_type * __restrict__ dst,
                                        const int64_t ne_total,
                                        const int64_t ne10,
                                        const int64_t ne11,
                                        const int64_t ne12,
                                        const int64_t ne13,
                                        const int64_t s01,
                                        const int64_t s02,
                                        const int64_t s03,
                                        const int64_t s10,
                                        const int64_t s11,
                                        const int64_t s12,
                                        const int64_t s1,
                                        const int64_t s2,
                                        const int64_t s3,
                                        const uint3   ne00,
                                        const uint3   ne01,
                                        const uint3   ne02,
                                        const uint3   ne11_fd,
                                        const uint3   ne12_fd,
                                        const float * __restrict__ kmean_mu) {
    const int64_t i = int64_t(blockDim.x) * blockIdx.x + threadIdx.x;

    if (i >= ne_total) {
        return;
    }

    const int64_t i_base = i * qk;
    uint32_t      tmp    = (uint32_t) i_base;
    uint2         div_mod;

    div_mod           = fast_div_modulo(tmp, ne00);
    const int64_t i00 = div_mod.y;
    tmp               = div_mod.x;

    div_mod           = fast_div_modulo(tmp, ne01);
    const int64_t i01 = div_mod.y;
    tmp               = div_mod.x;

    div_mod           = fast_div_modulo(tmp, ne02);
    const int64_t i02 = div_mod.y;
    const int64_t i03 = div_mod.x;

    const int64_t i12 = fastmodulo((uint32_t) i03, ne12_fd);
    const int64_t i11 = fastmodulo((uint32_t) i02, ne11_fd);
    const int64_t i10 = i01;

    ggml_cuda_pdl_sync();
    const int64_t dst_row = *(src1 + i10*s10 + i11*s11 + i12*s12);
    if (dst_row < 0) {
        return;
    }

    const float * src0_row = src0 + i01*s01 + i02*s02 + i03*s03;
    block_type * dst_row_ptr = dst + (dst_row*s1 + i02*s2 + i03*s3) / sizeof(block_type);

    const float * src_block = src0_row + i00;
    block_type * dst_block = dst_row_ptr + i00 / qk;

    // Affine tap: raw-domain per-head mean subtract before the block quantizer's normalize+FWHT.
    // kmean_mu != nullptr only for turbo4/turbo8 dispatches (null for standard quants → no-op).
    if (kmean_mu != nullptr) {
        float tapped[qk];
        #pragma unroll
        for (int j = 0; j < qk; j++) tapped[j] = src_block[j] - kmean_mu[i00 + j];
        quantize_func(tapped, dst_block);
    } else {
        quantize_func(src_block, dst_block);
    }

    GGML_UNUSED(ne10);
    GGML_UNUSED(ne11);
    GGML_UNUSED(ne12);
    GGML_UNUSED(ne13);
}

// Template dispatch function for quantized set_rows
template<typename idx_t, typename block_type, int qk, void (*quantize_func)(const float*, block_type*)>
static void set_rows_cuda_quant(
        const float * src0_d, const idx_t * src1_d, block_type * dst_d,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream, const float * kmean_mu = nullptr) {

    GGML_ASSERT(ne00 % qk == 0);
    const int64_t ne_total = (ne00 * ne01 * ne02 * ne03) / qk;
    const int num_blocks = (ne_total + CUDA_SET_ROWS_BLOCK_SIZE - 1) / CUDA_SET_ROWS_BLOCK_SIZE;
    const dim3 block_size(CUDA_SET_ROWS_BLOCK_SIZE);
    const dim3 grid_size(num_blocks);

    const int64_t s01 = nb01/sizeof(float);
    const int64_t s02 = nb02/sizeof(float);
    const int64_t s03 = nb03/sizeof(float);
    const int64_t s10 = nb10/sizeof(idx_t);
    const int64_t s11 = nb11/sizeof(idx_t);
    const int64_t s12 = nb12/sizeof(idx_t);
    const int64_t s1  = nb1;
    const int64_t s2  = nb2;
    const int64_t s3  = nb3;

    if (ne_total > 0 && ne00 > 0 && ne01 > 0 && ne02 > 0 && ne11 > 0 && ne12 > 0) {
        const uint3 ne00_fd = init_fastdiv_values((uint32_t) ne00);
        const uint3 ne01_fd = init_fastdiv_values((uint32_t) ne01);
        const uint3 ne02_fd = init_fastdiv_values((uint32_t) ne02);
        const uint3 ne11_fd = init_fastdiv_values((uint32_t) ne11);
        const uint3 ne12_fd = init_fastdiv_values((uint32_t) ne12);

        k_set_rows_quant<idx_t, block_type, qk, quantize_func><<<grid_size, block_size, 0, stream>>>(
            src0_d, src1_d, dst_d, ne_total, ne10, ne11, ne12, ne13, s01, s02, s03, s10, s11, s12, s1, s2, s3, ne00_fd,
            ne01_fd, ne02_fd, ne11_fd, ne12_fd, kmean_mu);
    }
}



// Affine-tap mean vector for a turbo4/turbo8 set_rows dst (per-layer slice of the K or V mean
// table, selected by cache_k_/cache_v_ name). Returns nullptr (no tap) unless the dst is a
// cache_{k,v}_l<N> tensor within table bounds AND a mean table is loaded (TURBO_KMEAN/VMEAN_SUB).
static inline const float * turbo_tap_mu(ggml_backend_cuda_context & ctx, const ggml_tensor * dst, int64_t ne00) {
    // Tap tier gate: no mean-sub at 8-bit — measured zero median gain and +9.5% mean KLD cost
    // (native flat A/B, q27 16k×8ch, 2026-07-05). Tap stays on for t4 and coarser (t4 median
    // −33%, t3 −39%). The graph-side V restore carries the matching TURBO8_0 exclusion.
    if (dst->type == GGML_TYPE_TURBO8_0) return nullptr;
    if (strncmp(dst->name, "cache_k_l", 9) != 0 && strncmp(dst->name, "cache_v_l", 9) != 0) return nullptr;
    const int pf_layer = atoi(dst->name + 9);
    if (pf_layer < 0 || pf_layer >= PFHEAD_MAX_L || ne00 > PFHEAD_MAX_C) return nullptr;
    const int iq_is_k = (strncmp(dst->name, "cache_k_", 8) == 0) ? 1 : 0;
    const float * tbl = iq_is_k ? turbo_kmean_table(ctx.device) : turbo_vmean_table_enc(ctx.device);
    return tbl ? tbl + (size_t) pf_layer * PFHEAD_MAX_C : nullptr;
}

template <typename src_t, typename idx_t, typename dst_t>
static __global__ void k_set_rows(const src_t * src0_ptr,
                                  const idx_t * src1_ptr,
                                  dst_t * dst_ptr,
                                  const int64_t ne_total,
                                  const int64_t ne10,
                                  const int64_t ne11,
                                  const int64_t ne12,
                                  const int64_t ne13,
                                  const int64_t s01,
                                  const int64_t s02,
                                  const int64_t s03,
                                  const int64_t s10,
                                  const int64_t s11,
                                  const int64_t s12,
                                  const int64_t s1,
                                  const int64_t s2,
                                  const int64_t s3,
                                  const uint3   ne00,
                                  const uint3   ne01,
                                  const uint3   ne02,
                                  const uint3   ne11_fd,
                                  const uint3   ne12_fd) {
    const src_t * GGML_CUDA_RESTRICT src0 = src0_ptr;
    const idx_t * GGML_CUDA_RESTRICT src1 = src1_ptr;
    dst_t       * GGML_CUDA_RESTRICT dst  = dst_ptr;
    const int64_t i = int64_t(blockDim.x) * blockIdx.x + threadIdx.x;

    if (i >= ne_total) {
        return;
    }

    uint32_t tmp = (uint32_t) i;
    uint2    div_mod;

    div_mod           = fast_div_modulo(tmp, ne00);
    const int64_t i00 = div_mod.y;
    tmp               = div_mod.x;

    div_mod           = fast_div_modulo(tmp, ne01);
    const int64_t i01 = div_mod.y;
    tmp               = div_mod.x;

    div_mod           = fast_div_modulo(tmp, ne02);
    const int64_t i02 = div_mod.y;
    const int64_t i03 = div_mod.x;

    const int64_t i12 = fastmodulo((uint32_t) i03, ne12_fd);
    const int64_t i11 = fastmodulo((uint32_t) i02, ne11_fd);
    const int64_t i10 = i01;

    ggml_cuda_pdl_sync();
    const int64_t dst_row = *(src1 + i10*s10 + i11*s11 + i12*s12);
    ggml_cuda_pdl_lc();
    if (dst_row < 0) {
        return;
    }

    const src_t * src0_row = src0 + i01*s01 + i02*s02 + i03*s03;
    dst_t * dst_row_ptr    = dst + dst_row*s1 + i02*s2 + i03*s3;

    dst_row_ptr[i00] = ggml_cuda_cast<dst_t>(src0_row[i00]);

    GGML_UNUSED(ne10);
    GGML_UNUSED(ne11);
    GGML_UNUSED(ne12);
    GGML_UNUSED(ne13);
}

template<typename src_t, typename idx_t, typename dst_t>
static void set_rows_cuda(
        const src_t * src0_d, const idx_t * src1_d, dst_t * dst_d,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream) {

    const int64_t ne_total = ne00 * ne01 * ne02 * ne03;
    const int num_blocks = (ne_total + CUDA_SET_ROWS_BLOCK_SIZE - 1) / CUDA_SET_ROWS_BLOCK_SIZE;
    const dim3 block_size(CUDA_SET_ROWS_BLOCK_SIZE);
    const dim3 grid_size(num_blocks);


    const int64_t s01 = nb01/sizeof(src_t);
    const int64_t s02 = nb02/sizeof(src_t);
    const int64_t s03 = nb03/sizeof(src_t);
    const int64_t s10 = nb10/sizeof(idx_t);
    const int64_t s11 = nb11/sizeof(idx_t);
    const int64_t s12 = nb12/sizeof(idx_t);
    const int64_t s1  = nb1/sizeof(dst_t);
    const int64_t s2  = nb2/sizeof(dst_t);
    const int64_t s3  = nb3/sizeof(dst_t);

    if (ne_total > 0 && ne00 > 0 && ne01 > 0 && ne02 > 0 && ne11 > 0 && ne12 > 0) {
        const uint3 ne00_fd = init_fastdiv_values((uint32_t) ne00);
        const uint3 ne01_fd = init_fastdiv_values((uint32_t) ne01);
        const uint3 ne02_fd = init_fastdiv_values((uint32_t) ne02);
        const uint3 ne11_fd = init_fastdiv_values((uint32_t) ne11);
        const uint3 ne12_fd = init_fastdiv_values((uint32_t) ne12);

        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_size, block_size, 0, stream);
        ggml_cuda_kernel_launch(k_set_rows<src_t, idx_t, dst_t>, launch_params,
            src0_d, src1_d, dst_d, ne_total, ne10, ne11, ne12, ne13, s01,
            s02, s03, s10, s11, s12, s1, s2, s3, ne00_fd, ne01_fd, ne02_fd,
            ne11_fd, ne12_fd);
    }
}

// Per-device backtrace buffer for Viterbi (replaces 32KB shared memory per block)
static uint8_t * tcq_bt_buf[GGML_CUDA_MAX_DEVICES] = {};
static int64_t   tcq_bt_buf_bytes[GGML_CUDA_MAX_DEVICES] = {};

static void ensure_tcq_bt_buf(int device, int64_t bytes_needed) {
    if (bytes_needed <= tcq_bt_buf_bytes[device]) return;
    if (tcq_bt_buf[device]) cudaFree(tcq_bt_buf[device]);
    cudaMalloc(&tcq_bt_buf[device], bytes_needed);
    tcq_bt_buf_bytes[device] = bytes_needed;
}

// === EXP-16 ragged reconstruct-to-f16 quality harness ===
// Stores KV as f16, but injects per-row quantization error per a static
// (position, layer, K/V) -> tier schedule from TURBO_RAGGED_SCHEDULE. Measures the
// KLD cost of per-row precision without any ragged storage or mixed-dtype fattn.
#define RAGGED_TIER_TURBO1_TCQ 11
#define RAGGED_TIER_TURBO2_TCQ 22
#define RAGGED_TIER_TURBO3_TCQ 33

static bool ragged_has_turbo1_tcq[GGML_CUDA_MAX_DEVICES] = {};
static bool ragged_has_turbo2_tcq[GGML_CUDA_MAX_DEVICES] = {};
static bool ragged_has_turbo3_tcq[GGML_CUDA_MAX_DEVICES] = {};

static bool ragged_schedule_active() {
    const char * e = getenv("TURBO_RAGGED_SCHEDULE");
    return e && e[0];
}

static bool ragged_read_schedule_file(const char * path, std::string & out) {
    FILE * f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "RAGGED schedule: cannot open %s (%s)\n", path, strerror(errno));
        return false;
    }
    if (fseek(f, 0, SEEK_END) != 0) {
        fprintf(stderr, "RAGGED schedule: cannot seek %s (%s)\n", path, strerror(errno));
        fclose(f);
        return false;
    }
    const long n = ftell(f);
    if (n < 0) {
        fprintf(stderr, "RAGGED schedule: cannot size %s (%s)\n", path, strerror(errno));
        fclose(f);
        return false;
    }
    rewind(f);
    out.resize((size_t) n);
    if (n > 0) {
        const size_t got = fread(&out[0], 1, (size_t) n, f);
        if (got != (size_t) n) {
            fprintf(stderr, "RAGGED schedule: short read %s (%zu/%ld bytes)\n", path, got, n);
            fclose(f);
            out.clear();
            return false;
        }
    }
    fclose(f);
    return true;
}

static bool ragged_load_schedule_text(const char * env, std::string & out) {
    if (!env || !env[0]) return false;
    if (env[0] == '@') {
        return ragged_read_schedule_file(env + 1, out);
    }
    if (!strchr(env, '=') && !strchr(env, ';')) {
        return ragged_read_schedule_file(env, out);
    }
    out = env;
    return true;
}

static int ragged_parse_tier(const char * s) {
    while (*s && isspace((unsigned char) *s)) s++;

    char norm[64];
    int n = 0;
    for (const char * p = s; *p && !isspace((unsigned char) *p) && n < (int) sizeof(norm) - 1; p++) {
        if (*p == '_' || *p == '-') continue;
        norm[n++] = (char) tolower((unsigned char) *p);
    }
    norm[n] = 0;

    if (strcmp(norm, "t1tcq") == 0 || strcmp(norm, "1tcq") == 0 ||
        strcmp(norm, "tcq1") == 0 || strcmp(norm, "turbo1tcq") == 0) {
        return RAGGED_TIER_TURBO1_TCQ;
    }
    if (strcmp(norm, "t3tcq") == 0 || strcmp(norm, "3tcq") == 0 ||
        strcmp(norm, "tcq3") == 0 || strcmp(norm, "turbo3tcq") == 0) {
        return RAGGED_TIER_TURBO3_TCQ;
    }
    if (strcmp(norm, "t2tcq") == 0 || strcmp(norm, "2tcq") == 0 ||
        strcmp(norm, "tcq2") == 0 || strcmp(norm, "turbo2tcq") == 0) {
        return RAGGED_TIER_TURBO2_TCQ;
    }

    if (*s == 't' || *s == 'T') s++;
    if (*s == 'f' || *s == 'F') return 16;   // fp16 / f16
    const int v = atoi(s);
    return (v == 16 || v == 8 || v == 4 || v == 3 || v == 2) ? v : 16;
}

static const char * ragged_tier_name(int tier) {
    switch (tier) {
        case 16: return "f16";
        case 8:  return "t8";
        case 4:  return "t4";
        case 3:  return "t3";
        case 2:  return "t2";
        case RAGGED_TIER_TURBO1_TCQ: return "t1tcq";
        case RAGGED_TIER_TURBO3_TCQ: return "t3tcq";
        case RAGGED_TIER_TURBO2_TCQ: return "t2tcq";
        default: return "unknown";
    }
}

static bool ragged_is_kv_cache(const char * name, int * layer, int * is_k) {
    const bool isk = strncmp(name, "cache_k_", 8) == 0;
    const bool isv = strncmp(name, "cache_v_", 8) == 0;
    if (!isk && !isv) return false;
    *is_k = isk ? 1 : 0;
    *layer = 0;
    const char * p = strstr(name, "_l");
    if (p) *layer = atoi(p + 2);
    return true;
}

// === TorQuant certificate harvest (task #141) ===
// TURBO_CERT_DUMP=<dir>: dump raw pre-FWHT (post-RoPE) K/V rows per layer as fp32,
// codec-independent (fires for ANY dst type incl. f16 KV). TURBO_CERT_DUMP_ROWS caps
// rows per (layer, K/V) file (default 4096). Files: <dir>/{k,v}_l%03d_w%d.f32
static char cert_dir[1024] = {0};
static int cert_state = 0; // 0 = unchecked, -1 = off, 1 = on
static int64_t cert_cap = 4096;
static int64_t cert_rows_written[2][160] = {};

static void cert_dump_rows(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const char * name) {
    if (cert_state == 0) {
        const char * e = getenv("TURBO_CERT_DUMP");
        if (!e || !e[0]) { cert_state = -1; return; }
        strncpy(cert_dir, e, sizeof(cert_dir) - 1);
        if (const char * r = getenv("TURBO_CERT_DUMP_ROWS")) cert_cap = atoll(r);
        cert_state = 1;
        fprintf(stderr, "CERT_DUMP: dir=%s cap=%lld rows per (layer, K/V)\n", cert_dir, (long long)cert_cap);
    }
    if (cert_state == -1) return;
    int layer, is_k;
    if (!ragged_is_kv_cache(name, &layer, &is_k)) return;
    if (layer < 0 || layer >= 160) return;
    int64_t & done = cert_rows_written[is_k][layer];
    if (done >= cert_cap) return;
    const int64_t w = src0->ne[0];
    int64_t n = src0->ne[1] * src0->ne[2] * src0->ne[3];
    if (n > cert_cap - done) n = cert_cap - done;
    if (n <= 0 || w <= 0) return;
    std::vector<float> h((size_t)(w * n));
    cudaStreamSynchronize(ctx.stream());
    int64_t got = 0;
    for (int64_t i3 = 0; i3 < src0->ne[3] && got < n; i3++) {
        for (int64_t i2 = 0; i2 < src0->ne[2] && got < n; i2++) {
            int64_t rows = src0->ne[1];
            if (rows > n - got) rows = n - got;
            cudaMemcpy2D(h.data() + got * w, w * sizeof(float),
                (const char *)src0->data + i2 * src0->nb[2] + i3 * src0->nb[3], src0->nb[1],
                w * sizeof(float), rows, cudaMemcpyDeviceToHost);
            got += rows;
        }
    }
    char path[1200];
    snprintf(path, sizeof(path), "%s/%s_l%03d_w%lld.f32", cert_dir, is_k ? "k" : "v", layer, (long long)w);
    FILE * f = fopen(path, done == 0 ? "wb" : "ab");
    if (!f) { fprintf(stderr, "CERT_DUMP: cannot open %s\n", path); cert_state = -1; return; }
    fwrite(h.data(), sizeof(float), (size_t)(w * got), f);
    fclose(f);
    done += got;
}

// Parse TURBO_RAGGED_SCHEDULE once per device and upload to constant memory.
// Format: "default=t4;band=LO-HI:TIER[:Llo-Lhi][:cLo-Hi][:k|v];...". TIER in
// {fp16,t8,t4,t3,t2,t1tcq,t2tcq,t3tcq}. cLo-Hi = coord-block (128-FWHT) range within a row; for the
// per-head axis, head h occupies cb [h*d_head/128, (h+1)*d_head/128).
static void load_ragged_schedule(int device) {
    static bool loaded[GGML_CUDA_MAX_DEVICES] = {};
    if (loaded[device]) return;
    loaded[device] = true;
    const char * env = getenv("TURBO_RAGGED_SCHEDULE");
    if (!env || !env[0]) return;

    std::string schedule;
    if (!ragged_load_schedule_text(env, schedule)) return;

    ragged_band bands[RAGGED_MAX_BANDS];
    int nbands = 0;
    int dropped_bands = 0;
    int default_tier = 16;
    bool has_t1tcq = false;
    bool has_t2tcq = false;
    bool has_t3tcq = false;

    std::vector<char> buf(schedule.begin(), schedule.end());
    buf.push_back('\0');
    char * sp = nullptr;
    for (char * tok = strtok_r(buf.data(), ";", &sp); tok; tok = strtok_r(nullptr, ";", &sp)) {
        while (*tok && isspace((unsigned char) *tok)) tok++;
        if (strncmp(tok, "default=", 8) == 0) {
            default_tier = ragged_parse_tier(tok + 8);
            has_t1tcq = has_t1tcq || default_tier == RAGGED_TIER_TURBO1_TCQ;
            has_t2tcq = has_t2tcq || default_tier == RAGGED_TIER_TURBO2_TCQ;
            has_t3tcq = has_t3tcq || default_tier == RAGGED_TIER_TURBO3_TCQ;
        } else if (strncmp(tok, "band=", 5) == 0) {
            if (nbands >= RAGGED_MAX_BANDS) {
                dropped_bands++;
                continue;
            }
            ragged_band b = { 0, 2000000000, 0, 100000, 0, 100000, 0, 16 };
            char * sp2 = nullptr;
            int fi = 0;
            for (char * fld = strtok_r(tok + 5, ":", &sp2); fld; fld = strtok_r(nullptr, ":", &sp2), fi++) {
                if (fi == 0) {
                    int lo, hi;
                    if (sscanf(fld, "%d-%d", &lo, &hi) == 2) { b.pos_lo = lo; b.pos_hi = hi; }
                } else if (fi == 1) {
                    b.tier = ragged_parse_tier(fld);
                    has_t1tcq = has_t1tcq || b.tier == RAGGED_TIER_TURBO1_TCQ;
                    has_t2tcq = has_t2tcq || b.tier == RAGGED_TIER_TURBO2_TCQ;
                    has_t3tcq = has_t3tcq || b.tier == RAGGED_TIER_TURBO3_TCQ;
                } else if (fld[0] == 'L' || fld[0] == 'l') {
                    int a, c;
                    if (sscanf(fld + 1, "%d-%d", &a, &c) == 2) { b.lay_lo = a; b.lay_hi = c; }
                } else if (fld[0] == 'c' || fld[0] == 'C') {
                    int a, c;
                    if (sscanf(fld + 1, "%d-%d", &a, &c) == 2) { b.cb_lo = a; b.cb_hi = c; }
                } else if (fld[0] == 'k' || fld[0] == 'K') {
                    b.kv = 1;
                } else if (fld[0] == 'v' || fld[0] == 'V') {
                    b.kv = 2;
                }
            }
            bands[nbands++] = b;
        }
    }
    if (nbands > 0) cudaMemcpyToSymbol(d_ragged_bands, bands, sizeof(ragged_band) * nbands);
    cudaMemcpyToSymbol(d_ragged_nbands, &nbands, sizeof(int));
    cudaMemcpyToSymbol(d_ragged_default_tier, &default_tier, sizeof(int));
    ragged_has_turbo1_tcq[device] = has_t1tcq;
    ragged_has_turbo2_tcq[device] = has_t2tcq;
    ragged_has_turbo3_tcq[device] = has_t3tcq;
    fprintf(stderr, "RAGGED schedule (device %d): default_tier=%s(%d), %d bands\n",
            device, ragged_tier_name(default_tier), default_tier, nbands);
    if (dropped_bands > 0) {
        fprintf(stderr, "RAGGED schedule: dropped %d bands after RAGGED_MAX_BANDS=%d\n",
                dropped_bands, RAGGED_MAX_BANDS);
    }
    for (int i = 0; i < nbands; i++) {
        fprintf(stderr, "  band[%d] pos[%d,%d) lay[%d,%d] cb[%d,%d) kv=%d tier=%s(%d)\n",
                i, bands[i].pos_lo, bands[i].pos_hi, bands[i].lay_lo, bands[i].lay_hi,
                bands[i].cb_lo, bands[i].cb_hi, bands[i].kv, ragged_tier_name(bands[i].tier), bands[i].tier);
    }
}

// Content/token axis (EXP-15d): load a per-(window,position) tier mask from
// TURBO_RAGGED_CONTENT_MASK once per device. File = int32 magic 'RCM1', int32
// n_windows, int32 n_ctx, then n_windows*n_ctx bytes (tier per cell, 0=default).
// The mask seeds ragged_lookup_tier; positional bands in TURBO_RAGGED_SCHEDULE
// still override it. d_ragged_window selects the active window (set per chunk).
static void load_ragged_content_mask(int device) {
    static bool loaded[GGML_CUDA_MAX_DEVICES] = {};
    if (loaded[device]) return;
    loaded[device] = true;
    const char * path = getenv("TURBO_RAGGED_CONTENT_MASK");
    if (!path || !path[0]) return;

    FILE * f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "RAGGED cmask: cannot open %s (%s)\n", path, strerror(errno)); return; }
    int32_t hdr[3] = {0,0,0};
    if (fread(hdr, sizeof(int32_t), 3, f) != 3 || hdr[0] != 0x52434d31) {
        fprintf(stderr, "RAGGED cmask: bad header in %s\n", path); fclose(f); return;
    }
    const int n_windows = hdr[1];
    const int n_ctx     = hdr[2];
    const size_t n = (size_t) n_windows * (size_t) n_ctx;
    if (n_windows <= 0 || n_ctx <= 0) { fprintf(stderr, "RAGGED cmask: empty dims\n"); fclose(f); return; }

    unsigned char * host = (unsigned char *) malloc(n);
    if (!host || fread(host, 1, n, f) != n) {
        fprintf(stderr, "RAGGED cmask: short read (%zu cells)\n", n); free(host); fclose(f); return;
    }
    fclose(f);

    bool has_t1tcq = false;
    bool has_t2tcq = false;
    bool has_t3tcq = false;
    for (size_t i = 0; i < n; i++) {
        has_t1tcq = has_t1tcq || host[i] == RAGGED_TIER_TURBO1_TCQ;
        has_t2tcq = has_t2tcq || host[i] == RAGGED_TIER_TURBO2_TCQ;
        has_t3tcq = has_t3tcq || host[i] == RAGGED_TIER_TURBO3_TCQ;
    }
    ragged_has_turbo1_tcq[device] = ragged_has_turbo1_tcq[device] || has_t1tcq;
    ragged_has_turbo2_tcq[device] = ragged_has_turbo2_tcq[device] || has_t2tcq;
    ragged_has_turbo3_tcq[device] = ragged_has_turbo3_tcq[device] || has_t3tcq;

    unsigned char * dptr = nullptr;
    cudaMalloc(&dptr, n);
    cudaMemcpy(dptr, host, n, cudaMemcpyHostToDevice);
    free(host);
    cudaMemcpyToSymbol(d_ragged_cmask, &dptr, sizeof(dptr));
    cudaMemcpyToSymbol(d_ragged_cmask_nctx, &n_ctx, sizeof(int));
    fprintf(stderr, "RAGGED cmask (device %d): %d windows x %d ctx loaded from %s\n",
            device, n_windows, n_ctx, path);
}

extern "C" void ggml_cuda_ragged_set_window(int window);

// Host setter: select the active content-mask window (= KLD chunk index).
// Called from the perplexity chunk loop. No-op if no mask is loaded.
extern "C" void ggml_cuda_ragged_set_window(int window) {
    cudaMemcpyToSymbol(d_ragged_window, &window, sizeof(int));
}

// One thread per 128-coord block: round-trip degraded blocks to original space
// (inverse FWHT) and write f16; copy protected blocks straight to f16.
template<typename src_t, typename idx_t>
static __global__ void k_set_rows_ragged_roundtrip(
        const src_t * __restrict__ src0, const idx_t * __restrict__ src1, half * __restrict__ dst,
        const int64_t n_blk_total,
        const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t s10, const int64_t s11, const int64_t s12,
        const int64_t s1, const int64_t s2, const int64_t s3,
        const uint3 bpr_fd, const uint3 ne01_fd, const uint3 ne02_fd,
        const uint3 ne11_fd, const uint3 ne12_fd,
        const int layer, const int is_k, const int v_rotated, const float * __restrict__ kvmean_mu) {
    const int64_t bidx = int64_t(blockDim.x) * blockIdx.x + threadIdx.x;
    if (bidx >= n_blk_total) return;

    uint32_t tmp = (uint32_t) bidx;
    uint2 dm;
    dm = fast_div_modulo(tmp, bpr_fd);  const int64_t cb  = dm.y; tmp = dm.x;
    dm = fast_div_modulo(tmp, ne01_fd); const int64_t i01 = dm.y; tmp = dm.x;
    dm = fast_div_modulo(tmp, ne02_fd); const int64_t i02 = dm.y; const int64_t i03 = dm.x;

    const int64_t i12 = fastmodulo((uint32_t) i03, ne12_fd);
    const int64_t i11 = fastmodulo((uint32_t) i02, ne11_fd);
    const int64_t i10 = i01;
    const int64_t dst_row = *(src1 + i10*s10 + i11*s11 + i12*s12);
    if (dst_row < 0) return;

    const int tier = ragged_lookup_tier(dst_row, layer, is_k, (int) cb);

    const src_t * src_blk = src0 + i01*s01 + i02*s02 + i03*s03 + cb*128;
    half * dst_blk = dst + dst_row*s1 + i02*s2 + i03*s3 + cb*128;
    const bool write_rotated_v = !is_k && v_rotated;

    if (tier >= 16) {
        if (write_rotated_v) {
            float src_local[128];
            for (int j = 0; j < 128; j++) {
                float v = ggml_cuda_cast<float>(src_blk[j]);
                if (kvmean_mu != nullptr) v -= kvmean_mu[cb*128 + j];
                src_local[j] = v;
            }
            turbo_rotate_forward_cuda(src_local, d_turbo_wht_signs1, d_turbo_wht_signs2);
            for (int j = 0; j < 128; j++) dst_blk[j] = __float2half(src_local[j]);
            return;
        }
        for (int j = 0; j < 128; j++) dst_blk[j] = ggml_cuda_cast<half>(src_blk[j]);
        return;
    }
    // Tap tier gate: t8 rows quantize RAW (no centering benefit at 8-bit — 2026-07-05 A/B).
    // The container contract stays uniform: under rotv=1 the graph adds mu_V back for the whole
    // layer, so gated V rows store (Q(x) − mu) — center AFTER the roundtrip instead of before.
    const bool tap_tier = (tier != 8);
    float src_local[128];
    for (int j = 0; j < 128; j++) {
        float v = ggml_cuda_cast<float>(src_blk[j]);
        if (tap_tier && kvmean_mu != nullptr) v -= kvmean_mu[cb*128 + j];
        src_local[j] = v;
    }
    float out[128];
    turbo_roundtrip_block_to_orig(src_local, out, tier, is_k);
    if (!tap_tier && !is_k && kvmean_mu != nullptr && v_rotated) {
        for (int j = 0; j < 128; j++) out[j] -= kvmean_mu[cb*128 + j];
    }
    if (write_rotated_v) {
        turbo_rotate_forward_cuda(out, d_turbo_wht_signs1, d_turbo_wht_signs2);
    } else if (tap_tier && !is_k && kvmean_mu != nullptr) {
        for (int j = 0; j < 128; j++) out[j] += kvmean_mu[cb*128 + j];
    }
    for (int j = 0; j < 128; j++) dst_blk[j] = __float2half(out[j]);
}

static void * ragged_tcq_tmp3[GGML_CUDA_MAX_DEVICES] = {};
static void * ragged_tcq_tmp2[GGML_CUDA_MAX_DEVICES] = {};
static void * ragged_tcq_tmp1[GGML_CUDA_MAX_DEVICES] = {};
static int64_t ragged_tcq_tmp3_bytes[GGML_CUDA_MAX_DEVICES] = {};
static int64_t ragged_tcq_tmp2_bytes[GGML_CUDA_MAX_DEVICES] = {};
static int64_t ragged_tcq_tmp1_bytes[GGML_CUDA_MAX_DEVICES] = {};

static bool ragged_v_rotated_enabled() {
    static int enabled = -1;
    if (enabled < 0) {
        const char * e = getenv("TURBO_RAGGED_V_ROTATED");
        enabled = (e && atoi(e) != 0) ? 1 : 0;
    }
    return enabled != 0;
}

static void ensure_ragged_tmp(int device, void ** ptr, int64_t * cap, int64_t bytes_needed) {
    GGML_UNUSED(device);
    if (bytes_needed <= *cap) return;
    if (*ptr) cudaFree(*ptr);
    cudaMalloc(ptr, bytes_needed);
    *cap = bytes_needed;
}

// V decode alpha per TCQ tier (turbo-tcq-alpha.cuh = the single source). Per-tier env
// overrides (TURBO_TCQ_DECODE_ALPHA_V1/V2/V3) allow mixed-tier ragged schedules to pin one
// tier's alpha without the global TURBO_TCQ_DECODE_ALPHA_V contaminating the other tiers.
static float ragged_tcq_decode_alpha(int tier, int is_k) {
    static bool loaded = false;
    static float alpha_k = 1.0f;
    static float alpha_v_override = 0.0f;
    static float alpha_v_tier[3] = { TURBO_TCQ_ALPHA_V_T1, TURBO_TCQ_ALPHA_V_T2, TURBO_TCQ_ALPHA_V_T3 }; // t1tcq, t2tcq, t3tcq
    if (!loaded) {
        loaded = true;
        if (const char * s = getenv("TURBO_TCQ_DECODE_ALPHA_K")) {
            char * end;
            const float a = strtof(s, &end);
            if (end != s && a > 0.0f && a < 10.0f) alpha_k = a;
        }
        if (const char * s = getenv("TURBO_TCQ_DECODE_ALPHA_V")) {
            char * end;
            const float a = strtof(s, &end);
            if (end != s && a > 0.0f && a < 10.0f) alpha_v_override = a;
        }
        const char * tier_env[3] = { "TURBO_TCQ_DECODE_ALPHA_V1", "TURBO_TCQ_DECODE_ALPHA_V2", "TURBO_TCQ_DECODE_ALPHA_V3" };
        for (int i = 0; i < 3; i++) {
            if (const char * s = getenv(tier_env[i])) {
                char * end;
                const float a = strtof(s, &end);
                if (end != s && a > 0.0f && a < 10.0f) alpha_v_tier[i] = a;
            }
        }
        fprintf(stderr, "RAGGED tcq decode alpha: K=%.4f V(t1,t2,t3)=(%.4f,%.4f,%.4f)%s\n",
                alpha_k, alpha_v_tier[0], alpha_v_tier[1], alpha_v_tier[2],
                alpha_v_override > 0.0f ? " [global V override active]" : "");
    }
    if (is_k) return alpha_k;
    if (alpha_v_override > 0.0f) return alpha_v_override;
    if (tier == RAGGED_TIER_TURBO1_TCQ) return alpha_v_tier[0];
    return tier == RAGGED_TIER_TURBO2_TCQ ? alpha_v_tier[1] : alpha_v_tier[2];
}

template<typename idx_t>
static int ragged_tcq1_shared_bt(int device) {
    static int use_shared[GGML_CUDA_MAX_DEVICES] = {};
    static bool checked[GGML_CUDA_MAX_DEVICES] = {};
    if (checked[device]) return use_shared[device];
    checked[device] = true;
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    const char * env = getenv("TURBO_TCQ_SHARED_BT");
    if (!env || atoi(env) != 0) {
        constexpr int bytes = 128 * 128;
        int max_shared_optin = 0;
        CUDA_CHECK(cudaDeviceGetAttribute(&max_shared_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
        if (max_shared_optin >= bytes) {
            CUDA_SET_SHARED_MEMORY_LIMIT(k_set_rows_turbo1_tcq<idx_t>, bytes);
            use_shared[device] = 1;
        }
    }
#endif
    return use_shared[device];
}

template<typename idx_t>
static int ragged_tcq3_shared_bt(int device) {
    static int use_shared[GGML_CUDA_MAX_DEVICES] = {};
    static bool checked[GGML_CUDA_MAX_DEVICES] = {};
    if (checked[device]) return use_shared[device];
    checked[device] = true;
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    const char * env = getenv("TURBO_TCQ_SHARED_BT");
    if (!env || atoi(env) != 0) {
        constexpr int bytes = 128 * 64;
        int max_shared_optin = 0;
        CUDA_CHECK(cudaDeviceGetAttribute(&max_shared_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
        if (max_shared_optin >= bytes) {
            CUDA_SET_SHARED_MEMORY_LIMIT(k_set_rows_turbo3_tcq<idx_t>, bytes);
            use_shared[device] = 1;
        }
    }
#endif
    return use_shared[device];
}

template<typename idx_t>
static int ragged_tcq2_shared_bt(int device) {
    static int use_shared[GGML_CUDA_MAX_DEVICES] = {};
    static bool checked[GGML_CUDA_MAX_DEVICES] = {};
    if (checked[device]) return use_shared[device];
    checked[device] = true;
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    const char * env = getenv("TURBO_TCQ_SHARED_BT");
    if (!env || atoi(env) != 0) {
        constexpr int bytes = 128 * 64;
        int max_shared_optin = 0;
        CUDA_CHECK(cudaDeviceGetAttribute(&max_shared_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
        if (max_shared_optin >= bytes) {
            CUDA_SET_SHARED_MEMORY_LIMIT(k_set_rows_turbo2_tcq<idx_t>, bytes);
            use_shared[device] = 1;
        }
    }
#endif
    return use_shared[device];
}

template<typename idx_t>
static __global__ void k_ragged_turbo3_tcq_overlay(
        const idx_t * __restrict__ src1, const block_turbo3_tcq * __restrict__ qtmp, half * __restrict__ dst,
        const int64_t n_blk_total,
        const int64_t s10, const int64_t s11, const int64_t s12,
        const int64_t qs1, const int64_t qs2, const int64_t qs3,
        const int64_t s1, const int64_t s2, const int64_t s3,
        const uint3 bpr_fd, const uint3 ne01_fd, const uint3 ne02_fd,
        const uint3 ne11_fd, const uint3 ne12_fd,
        const int layer, const int is_k, const int v_rotated, const float alpha, const float * __restrict__ kvmean_mu) {
    const int64_t bidx = int64_t(blockDim.x) * blockIdx.x + threadIdx.x;
    if (bidx >= n_blk_total) return;

    uint32_t tmp = (uint32_t) bidx;
    uint2 dm;
    dm = fast_div_modulo(tmp, bpr_fd);  const int64_t cb  = dm.y; tmp = dm.x;
    dm = fast_div_modulo(tmp, ne01_fd); const int64_t i01 = dm.y; tmp = dm.x;
    dm = fast_div_modulo(tmp, ne02_fd); const int64_t i02 = dm.y; const int64_t i03 = dm.x;

    const int64_t i12 = fastmodulo((uint32_t) i03, ne12_fd);
    const int64_t i11 = fastmodulo((uint32_t) i02, ne11_fd);
    const int64_t dst_row = *(src1 + i01*s10 + i11*s11 + i12*s12);
    if (dst_row < 0) return;

    if (ragged_lookup_tier(dst_row, layer, is_k, (int) cb) != RAGGED_TIER_TURBO3_TCQ) return;

    const block_turbo3_tcq * blk = (const block_turbo3_tcq *)((const char *)qtmp + dst_row*qs1 + i02*qs2 + i03*qs3) + cb;
    half * dst_blk = dst + dst_row*s1 + i02*s2 + i03*s3 + cb*128;

    float xr[128];
    const float norm = __half2float(blk->norm) * alpha;
    for (int t = 0; t < 128; t++) {
        const int bit_pos = t * 3;
        const int byte_idx = bit_pos / 8;
        const int bit_off = bit_pos % 8;
        const uint16_t raw = (uint16_t)blk->qs[byte_idx] | ((uint16_t)blk->qs[byte_idx + 1] << 8);
        const int state = (raw >> bit_off) & 0x1FF;
        xr[t] = (is_k ? d_turbo3_tcq_codebook : d_turbo3_tcq_codebook_v)[state];
    }
    if (!is_k && v_rotated) {
        for (int j = 0; j < 128; j++) dst_blk[j] = __float2half(xr[j] * norm);
        return;
    }
    turbo_rotate_forward_cuda(xr, d_turbo_wht_signs2, d_turbo_wht_signs1);
    for (int j = 0; j < 128; j++) {
        float v = xr[j] * d_innerq_channel_scale_inv[j] * norm;
        if (!is_k && kvmean_mu != nullptr) v += kvmean_mu[cb*128 + j];
        dst_blk[j] = __float2half(v);
    }
}

template<typename idx_t>
static __global__ void k_ragged_turbo2_tcq_overlay(
        const idx_t * __restrict__ src1, const block_turbo2_tcq * __restrict__ qtmp, half * __restrict__ dst,
        const int64_t n_blk_total,
        const int64_t s10, const int64_t s11, const int64_t s12,
        const int64_t qs1, const int64_t qs2, const int64_t qs3,
        const int64_t s1, const int64_t s2, const int64_t s3,
        const uint3 bpr_fd, const uint3 ne01_fd, const uint3 ne02_fd,
        const uint3 ne11_fd, const uint3 ne12_fd,
        const int layer, const int is_k, const int v_rotated, const float alpha, const float * __restrict__ kvmean_mu) {
    const int64_t bidx = int64_t(blockDim.x) * blockIdx.x + threadIdx.x;
    if (bidx >= n_blk_total) return;

    uint32_t tmp = (uint32_t) bidx;
    uint2 dm;
    dm = fast_div_modulo(tmp, bpr_fd);  const int64_t cb  = dm.y; tmp = dm.x;
    dm = fast_div_modulo(tmp, ne01_fd); const int64_t i01 = dm.y; tmp = dm.x;
    dm = fast_div_modulo(tmp, ne02_fd); const int64_t i02 = dm.y; const int64_t i03 = dm.x;

    const int64_t i12 = fastmodulo((uint32_t) i03, ne12_fd);
    const int64_t i11 = fastmodulo((uint32_t) i02, ne11_fd);
    const int64_t dst_row = *(src1 + i01*s10 + i11*s11 + i12*s12);
    if (dst_row < 0) return;

    if (ragged_lookup_tier(dst_row, layer, is_k, (int) cb) != RAGGED_TIER_TURBO2_TCQ) return;

    const block_turbo2_tcq * blk = (const block_turbo2_tcq *)((const char *)qtmp + dst_row*qs1 + i02*qs2 + i03*qs3) + cb;
    half * dst_blk = dst + dst_row*s1 + i02*s2 + i03*s3 + cb*128;

    float xr[128];
    const float norm = __half2float(blk->norm) * alpha;
    for (int t = 0; t < 128; t++) {
        const int bit_pos = t * 2;
        const int byte_idx = bit_pos / 8;
        const int bit_off = bit_pos % 8;
        const uint16_t raw = (uint16_t)blk->qs[byte_idx] | ((uint16_t)blk->qs[byte_idx + 1] << 8);
        const int state = (raw >> bit_off) & 0xFF;
        xr[t] = (is_k ? d_turbo2_tcq_codebook : d_turbo2_tcq_codebook_v)[state];
    }
    if (!is_k && v_rotated) {
        for (int j = 0; j < 128; j++) dst_blk[j] = __float2half(xr[j] * norm);
        return;
    }
    turbo_rotate_forward_cuda(xr, d_turbo_wht_signs2, d_turbo_wht_signs1);
    for (int j = 0; j < 128; j++) {
        float v = xr[j] * d_innerq_channel_scale_inv[j] * norm;
        if (!is_k && kvmean_mu != nullptr) v += kvmean_mu[cb*128 + j];
        dst_blk[j] = __float2half(v);
    }
}

static __device__ __forceinline__ float ragged_fwht128_butterfly_inplace(float val, float * smem) {
    const int tid = threadIdx.x;

    #pragma unroll
    for (int h = 1; h <= 16; h *= 2) {
        const float other = __shfl_xor_sync(0xFFFFFFFFULL, val, h);
        val = (tid & h) ? (other - val) : (val + other);
    }

    smem[tid] = val;
    __syncthreads();
    val = (tid & 32) ? (smem[tid - 32] - val) : (val + smem[tid + 32]);
    __syncthreads();

    smem[tid] = val;
    __syncthreads();
    val = (tid & 64) ? (smem[tid - 64] - val) : (val + smem[tid + 64]);
    __syncthreads();

    return val;
}

static __device__ __forceinline__ void ragged_fwht128_store_half(float val, half * dst_base) {
    const int tid = threadIdx.x;
    const float neighbor = __shfl_xor_sync(0xFFFFFFFFULL, val, 1);
    if ((tid & 1) == 0) {
        const half2 packed = __floats2half2_rn(val, neighbor);
        *((half2 *)(dst_base + tid)) = packed;
    }
}

template<typename idx_t>
static __global__ void k_ragged_turbo1_tcq_overlay(
        const idx_t * __restrict__ src1, const block_turbo1_tcq * __restrict__ qtmp, half * __restrict__ dst,
        const int64_t n_blk_total,
        const int64_t s10, const int64_t s11, const int64_t s12,
        const int64_t qs1, const int64_t qs2, const int64_t qs3,
        const int64_t s1, const int64_t s2, const int64_t s3,
        const uint3 bpr_fd, const uint3 ne01_fd, const uint3 ne02_fd,
        const uint3 ne11_fd, const uint3 ne12_fd,
        const int layer, const int is_k, const int v_rotated, const float alpha) {
    const int64_t bidx = blockIdx.x;
    if (bidx >= n_blk_total) return;
    const int tid = threadIdx.x;

    uint32_t tmp = (uint32_t) bidx;
    uint2 dm;
    dm = fast_div_modulo(tmp, bpr_fd);  const int64_t cb  = dm.y; tmp = dm.x;
    dm = fast_div_modulo(tmp, ne01_fd); const int64_t i01 = dm.y; tmp = dm.x;
    dm = fast_div_modulo(tmp, ne02_fd); const int64_t i02 = dm.y; const int64_t i03 = dm.x;

    const int64_t i12 = fastmodulo((uint32_t) i03, ne12_fd);
    const int64_t i11 = fastmodulo((uint32_t) i02, ne11_fd);
    const int64_t dst_row = *(src1 + i01*s10 + i11*s11 + i12*s12);
    if (dst_row < 0) return;

    if (ragged_lookup_tier(dst_row, layer, is_k, (int) cb) != RAGGED_TIER_TURBO1_TCQ) return;

    const block_turbo1_tcq * blk = (const block_turbo1_tcq *)((const char *)qtmp + dst_row*qs1 + i02*qs2 + i03*qs3) + cb;
    half * dst_blk = dst + dst_row*s1 + i02*s2 + i03*s3 + cb*128;

    const float norm = __half2float(blk->norm) * alpha;
    const float * cbk = is_k ? d_turbo1_tcq_codebook : d_turbo1_tcq_codebook_v;
    const int byte_idx = tid >> 3;
    const int bit_off = tid & 7;
    const uint16_t raw = (uint16_t)blk->qs[byte_idx] | ((uint16_t)blk->qs[byte_idx + 1] << 8);
    const int state = (raw >> bit_off) & 0xFF;
    const float recon = norm * cbk[state];
    if (!is_k && v_rotated) {
        ragged_fwht128_store_half(recon, dst_blk);
        return;
    }

    __shared__ float smem[128];
    constexpr float inv_sqrt_128 = 0.08838834764831845f;
    float val = ragged_fwht128_butterfly_inplace(recon * d_turbo_wht_signs2[tid], smem);
    val = val * inv_sqrt_128 * d_turbo_wht_signs1[tid] * d_innerq_channel_scale_inv[tid];
    ragged_fwht128_store_half(val, dst_blk);
}

// Cooperative turbo4 KV-write encode: 128 threads per 128-elem block, one element per thread.
// Replaces the generic 1-thread-per-block set_rows path for turbo4, which launched only ~tens of
// threads in decode (few blocks) and spilled a 512 B/thread local array (~131 ms/decode). Same
// encode math as quantize_f32_turbo4_0_block (norm-normalize -> signs1 -> FWHT -> signs2 -> 4-bit
// centroid -> norm correction), parallelized with the ragged_fwht128 butterfly primitive above.
// Reductions are tree-order (vs serial) -> ULP-level, well within a 4-bit codec's noise. Bypassed
// (falls back to the generic path) when TURBO_EXTRACT post-rotation dumping is armed.
template<typename idx_t>
static __global__ void k_set_rows_turbo4_coop(
        const float * __restrict__ src0, const idx_t * __restrict__ src1,
        block_turbo4_0 * __restrict__ dst,
        const int64_t ne00, const int64_t ne01, const int64_t ne02,
        const int64_t ne11, const int64_t ne12,
        const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t s10, const int64_t s11, const int64_t s12,
        const int64_t s1,  const int64_t s2,  const int64_t s3,
        const float * __restrict__ kmean_mu) {
    const int     tid   = threadIdx.x;                 // 0..127, element within block
    const int64_t nbpr  = ne00 / QK_TURBO4;            // turbo4 blocks per row
    const int64_t g     = blockIdx.x;                  // flat block index
    const int64_t i_blk = g % nbpr;
    int64_t       t     = g / nbpr;
    const int64_t i01   = t % ne01; t /= ne01;
    const int64_t i02   = t % ne02; const int64_t i03 = t / ne02;
    const int64_t i12   = i03 % ne12;                  // matches k_set_rows_quant index decode
    const int64_t i11   = i02 % ne11;
    const int64_t i10   = i01;

    ggml_cuda_pdl_sync();
    const int64_t dst_row = *(src1 + i10*s10 + i11*s11 + i12*s12);
    if (dst_row < 0) return;

    const float *    src_row = src0 + i01*s01 + i02*s02 + i03*s03;
    block_turbo4_0 * blk     = (block_turbo4_0 *)((char *)dst + (dst_row*s1 + i02*s2 + i03*s3)) + i_blk;
    const int64_t    ch      = i_blk * QK_TURBO4 + tid;   // absolute channel within the row

    // load + affine tap (raw-domain per-head mean subtract; kmean_mu null => no-op)
    float v = src_row[ch];
    if (kmean_mu) v -= kmean_mu[ch];

    // ||tapped|| via a 128-thread tree reduction
    __shared__ float rsm[QK_TURBO4];
    rsm[tid] = v * v; __syncthreads();
    #pragma unroll
    for (int s = QK_TURBO4/2; s > 0; s >>= 1) { if (tid < s) rsm[tid] += rsm[tid + s]; __syncthreads(); }
    const float norm     = sqrtf(rsm[0]);
    const float inv_norm = norm > 1e-10f ? 1.0f / norm : 0.0f;
    __syncthreads();

    // normalize -> signs1 -> FWHT -> inv_sqrt128 -> signs2  (matches turbo_rotate_forward_cuda)
    __shared__ float smem[QK_TURBO4];
    float xr = ragged_fwht128_butterfly_inplace(v * inv_norm * d_turbo_wht_signs1[tid], smem);
    xr = xr * 0.08838834764831845f * d_turbo_wht_signs2[tid];

    // 4-bit centroid
    const uint8_t idx = turbo_find_nearest_4bit(xr);

    // reconstruction norm (sum of centroid^2) for the norm correction
    const float c = d_turbo_centroids_4bit[idx];
    __syncthreads();
    rsm[tid] = c * c; __syncthreads();
    #pragma unroll
    for (int s = QK_TURBO4/2; s > 0; s >>= 1) { if (tid < s) rsm[tid] += rsm[tid + s]; __syncthreads(); }
    const float recon_norm = sqrtf(rsm[0]);

    // pack two 4-bit indices per byte (even thread packs itself + its odd neighbor)
    __shared__ uint8_t sidx[QK_TURBO4];
    sidx[tid] = idx; __syncthreads();
    if ((tid & 1) == 0) blk->qs[tid >> 1] = (uint8_t)((sidx[tid + 1] << 4) | sidx[tid]);
    if (tid == 0) {
        const float corrected = recon_norm > 1e-10f ? norm / recon_norm : norm;
        blk->norm = __float2half(corrected * d_turbo4_alpha);
    }
}

// Cooperative turbo8 KV-write encode (mirror of k_set_rows_turbo4_coop). Same 1-thread-per-block
// underutilization existed for turbo8 (8-bit VBR top tier / t8:t4 asymmetric). turbo8 has NO
// affine tap (mean-sub gated off at 8-bit) and no pair-packing (one byte per element). Encode math
// matches quantize_f32_turbo8_0_block: norm-normalize -> signs1 -> FWHT -> signs2 -> per-block
// absmax scale -> 8-bit (companded override or uniform grid). The absmax reduction is exact (max is
// associative); only the norm sum is tree-order (ULP). Falls back to generic when TURBO_EXTRACT armed.
template<typename idx_t>
static __global__ void k_set_rows_turbo8_coop(
        const float * __restrict__ src0, const idx_t * __restrict__ src1,
        block_turbo8_0 * __restrict__ dst,
        const int64_t ne00, const int64_t ne01, const int64_t ne02,
        const int64_t ne11, const int64_t ne12,
        const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t s10, const int64_t s11, const int64_t s12,
        const int64_t s1,  const int64_t s2,  const int64_t s3) {
    const int     tid   = threadIdx.x;
    const int64_t nbpr  = ne00 / QK_TURBO8;
    const int64_t g     = blockIdx.x;
    const int64_t i_blk = g % nbpr;
    int64_t       t     = g / nbpr;
    const int64_t i01   = t % ne01; t /= ne01;
    const int64_t i02   = t % ne02; const int64_t i03 = t / ne02;
    const int64_t i12   = i03 % ne12;
    const int64_t i11   = i02 % ne11;
    const int64_t i10   = i01;

    ggml_cuda_pdl_sync();
    const int64_t dst_row = *(src1 + i10*s10 + i11*s11 + i12*s12);
    if (dst_row < 0) return;

    const float *    src_row = src0 + i01*s01 + i02*s02 + i03*s03;
    block_turbo8_0 * blk     = (block_turbo8_0 *)((char *)dst + (dst_row*s1 + i02*s2 + i03*s3)) + i_blk;
    const int64_t    ch      = i_blk * QK_TURBO8 + tid;

    const float v = src_row[ch];   // no affine tap at 8-bit

    // ||x|| via tree reduction (for normalize)
    __shared__ float rsm[QK_TURBO8];
    rsm[tid] = v * v; __syncthreads();
    #pragma unroll
    for (int s = QK_TURBO8/2; s > 0; s >>= 1) { if (tid < s) rsm[tid] += rsm[tid + s]; __syncthreads(); }
    const float norm     = sqrtf(rsm[0]);
    const float inv_norm = norm > 1e-10f ? 1.0f / norm : 0.0f;
    __syncthreads();

    // normalize -> signs1 -> FWHT -> inv_sqrt128 -> signs2
    __shared__ float smem[QK_TURBO8];
    float xr = ragged_fwht128_butterfly_inplace(v * inv_norm * d_turbo_wht_signs1[tid], smem);
    xr = xr * 0.08838834764831845f * d_turbo_wht_signs2[tid];

    // per-block absmax (exact: max is associative + commutative)
    __syncthreads();
    rsm[tid] = fabsf(xr); __syncthreads();
    #pragma unroll
    for (int s = QK_TURBO8/2; s > 0; s >>= 1) { if (tid < s) rsm[tid] = fmaxf(rsm[tid], rsm[tid + s]); __syncthreads(); }
    const float scale     = rsm[0] > 1e-10f ? rsm[0] : 1e-10f;
    const float inv_scale = 1.0f / scale;

    // 8-bit quantize: companded book (TURBO_CB_T8) or the stock uniform 256-level grid
    uint8_t q;
    if (d_turbo8_cb_override) {
        const float vv = xr * inv_scale;
        int idx = 0;
        #pragma unroll
        for (int step = 128; step >= 1; step >>= 1) {
            const int cand = idx + step;
            if (cand <= 255 && d_turbo_mid_8bit[cand - 1] < vv) idx = cand;
        }
        q = (uint8_t)idx;
    } else {
        int idx = (int)lrintf(xr * inv_scale * 127.5f + 127.5f);
        idx = idx < 0 ? 0 : (idx > 255 ? 255 : idx);
        q = (uint8_t)idx;
    }
    blk->qs[tid] = q;
    if (tid == 0) blk->norm = __float2half(norm * scale);
}

template<typename src_t, typename idx_t>
static void set_rows_cuda(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const src_t * src0_d = (const src_t *)src0->data;
    const idx_t * src1_d = (const idx_t *)src1->data;

    GGML_TENSOR_BINARY_OP_LOCALS

    cudaStream_t stream = ctx.stream();


    if (dst->type == GGML_TYPE_F32) {
        set_rows_cuda(
            src0_d, src1_d, (float*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_F16) {
        int rg_layer, rg_is_k;
        if (ragged_schedule_active() && ragged_is_kv_cache(dst->name, &rg_layer, &rg_is_k) && (ne00 % 128 == 0)) {
            load_ragged_schedule(ctx.device);
            load_ragged_content_mask(ctx.device);
            load_turbo4_alpha(ctx.device);
            static bool ragged_innerq_ready[GGML_CUDA_MAX_DEVICES] = {};
            if (!ragged_innerq_ready[ctx.device]) {
                turbo_innerq_init(); // identity scales unless a real turbo SET_ROWS run calibrates them
                ragged_innerq_ready[ctx.device] = true;
            }
            const int64_t bpr = ne00 / 128;
            const int64_t n_blk_total = bpr * ne01 * ne02 * ne03;
            const int num_blocks = (n_blk_total + CUDA_SET_ROWS_BLOCK_SIZE - 1) / CUDA_SET_ROWS_BLOCK_SIZE;
            const int64_t s01 = nb01/sizeof(src_t), s02 = nb02/sizeof(src_t), s03 = nb03/sizeof(src_t);
            const int64_t s10 = nb10/sizeof(idx_t), s11 = nb11/sizeof(idx_t), s12 = nb12/sizeof(idx_t);
            const int64_t s1 = nb1/sizeof(half), s2 = nb2/sizeof(half), s3 = nb3/sizeof(half);
            if (n_blk_total > 0 && ne01 > 0 && ne02 > 0 && ne11 > 0 && ne12 > 0) {
                const uint3 bpr_fd  = init_fastdiv_values((uint32_t) bpr);
                const uint3 ne01_fd = init_fastdiv_values((uint32_t) ne01);
                const uint3 ne02_fd = init_fastdiv_values((uint32_t) ne02);
                const uint3 ne11_fd = init_fastdiv_values((uint32_t) ne11);
                const uint3 ne12_fd = init_fastdiv_values((uint32_t) ne12);
                const int ragged_v_rotated = (!rg_is_k && ragged_v_rotated_enabled()) ? 1 : 0;
                const float * kvmean_mu = nullptr;
                if (rg_layer >= 0 && rg_layer < PFHEAD_MAX_L && ne00 <= PFHEAD_MAX_C) {
                    const float * tbl = rg_is_k ? turbo_kmean_table(ctx.device) : turbo_vmean_table_enc(ctx.device);
                    if (tbl) kvmean_mu = tbl + (size_t) rg_layer * PFHEAD_MAX_C;
                }
                k_set_rows_ragged_roundtrip<src_t, idx_t><<<num_blocks, CUDA_SET_ROWS_BLOCK_SIZE, 0, stream>>>(
                    src0_d, src1_d, (half*)dst->data, n_blk_total,
                    s01, s02, s03, s10, s11, s12, s1, s2, s3,
                    bpr_fd, ne01_fd, ne02_fd, ne11_fd, ne12_fd, rg_layer, rg_is_k, ragged_v_rotated, kvmean_mu);

                if (ragged_has_turbo3_tcq[ctx.device] || ragged_has_turbo2_tcq[ctx.device] || ragged_has_turbo1_tcq[ctx.device]) {
                    const int64_t tmp_groups = bpr * ne1 * ne2 * ne3;
                    const int64_t s01_f = nb01/sizeof(float), s02_f = nb02/sizeof(float), s03_f = nb03/sizeof(float);
                    const int64_t s10_i = nb10/sizeof(idx_t), s11_i = nb11/sizeof(idx_t), s12_i = nb12/sizeof(idx_t);
                    const uint3 ne00_fd = init_fastdiv_values((uint32_t) ne00);
                    if (ragged_has_turbo1_tcq[ctx.device]) {
                        static bool tcq1_cb_loaded[GGML_CUDA_MAX_DEVICES] = {};
                        static int tcq1_hot_e = -1; if (tcq1_hot_e < 0) tcq1_hot_e = getenv("TURBO1_TCQ_HOTSWAP") ? 1 : 0;
                        if (!tcq1_cb_loaded[ctx.device] || tcq1_hot_e) {
                            turbo1_tcq_load_kv_encode();
                            tcq1_cb_loaded[ctx.device] = true;
                        }
                        const int64_t qs1 = bpr * (int64_t) sizeof(block_turbo1_tcq);
                        const int64_t qs2 = ne1 * qs1;
                        const int64_t qs3 = ne2 * qs2;
                        ensure_ragged_tmp(ctx.device, &ragged_tcq_tmp1[ctx.device], &ragged_tcq_tmp1_bytes[ctx.device],
                                tmp_groups * (int64_t) sizeof(block_turbo1_tcq));
                        const int use_shared = ragged_tcq1_shared_bt<idx_t>(ctx.device);
                        if (!use_shared) ensure_tcq_bt_buf(ctx.device, n_blk_total * 128 * 128);
                        // V tap: encode subtracts mu, rotated-V storage emits coefficients, and the
                        // graph-level un-rotation restores mu_V (same contract as t2/t3 overlays).
                        // The rotv=0 overlay path still has no V add-back — tap requires rotv=1.
                        k_set_rows_turbo1_tcq<idx_t><<<(int)n_blk_total, 256, use_shared ? 128 * 128 : 0, stream>>>(
                            (const float *) src0_d, src1_d, (block_turbo1_tcq *) ragged_tcq_tmp1[ctx.device],
                            n_blk_total, tcq_bt_buf[ctx.device], use_shared, rg_is_k, kvmean_mu,
                            s01_f, s02_f, s03_f, s10_i, s11_i, s12_i, qs1, qs2, qs3,
                            ne00_fd, ne01_fd, ne02_fd, ne11_fd, ne12_fd);
                        k_ragged_turbo1_tcq_overlay<idx_t><<<(int)n_blk_total, 128, 0, stream>>>(
                            src1_d, (const block_turbo1_tcq *) ragged_tcq_tmp1[ctx.device], (half *) dst->data, n_blk_total,
                            s10_i, s11_i, s12_i, qs1, qs2, qs3, s1, s2, s3,
                            bpr_fd, ne01_fd, ne02_fd, ne11_fd, ne12_fd, rg_layer, rg_is_k, ragged_v_rotated,
                            ragged_tcq_decode_alpha(RAGGED_TIER_TURBO1_TCQ, rg_is_k));
                    }

                    if (ragged_has_turbo3_tcq[ctx.device]) {
                        static bool tcq3_cb_loaded[GGML_CUDA_MAX_DEVICES] = {};
                        static int tcq3_hot_e = -1; if (tcq3_hot_e < 0) tcq3_hot_e = getenv("TURBO_TCQ_HOTSWAP") ? 1 : 0;
                        if (!tcq3_cb_loaded[ctx.device] || tcq3_hot_e) {
                            turbo_tcq_load_kv_encode();
                            if (!tcq3_cb_loaded[ctx.device]) {
                                tcq3_cb_loaded[ctx.device] = true;
                                load_tcq_norm_alpha(ctx.device);
                                init_tcq_error_dump(ctx.device);
                            }
                        }
                        const int64_t qs1 = bpr * (int64_t) sizeof(block_turbo3_tcq);
                        const int64_t qs2 = ne1 * qs1;
                        const int64_t qs3 = ne2 * qs2;
                        ensure_ragged_tmp(ctx.device, &ragged_tcq_tmp3[ctx.device], &ragged_tcq_tmp3_bytes[ctx.device],
                                tmp_groups * (int64_t) sizeof(block_turbo3_tcq));
                        const int use_shared = ragged_tcq3_shared_bt<idx_t>(ctx.device);
                        if (!use_shared) ensure_tcq_bt_buf(ctx.device, n_blk_total * 128 * 64);
                        k_set_rows_turbo3_tcq<idx_t><<<(int)n_blk_total, TCQ3_ENC_NT, use_shared ? 128 * 64 : 0, stream>>>(
                            (const float *) src0_d, src1_d, (block_turbo3_tcq *) ragged_tcq_tmp3[ctx.device],
                            n_blk_total, tcq_bt_buf[ctx.device], use_shared, ne00, ne01, ne02, ne10, ne11, ne12, ne13,
                            s01_f, s02_f, s03_f, s10_i, s11_i, s12_i, rg_is_k, kvmean_mu, qs1, qs2, qs3,
                            ne00_fd, ne01_fd, ne02_fd, ne11_fd, ne12_fd);
                        k_ragged_turbo3_tcq_overlay<idx_t><<<num_blocks, CUDA_SET_ROWS_BLOCK_SIZE, 0, stream>>>(
                            src1_d, (const block_turbo3_tcq *) ragged_tcq_tmp3[ctx.device], (half *) dst->data, n_blk_total,
                            s10_i, s11_i, s12_i, qs1, qs2, qs3, s1, s2, s3,
                            bpr_fd, ne01_fd, ne02_fd, ne11_fd, ne12_fd, rg_layer, rg_is_k, ragged_v_rotated,
                            ragged_tcq_decode_alpha(RAGGED_TIER_TURBO3_TCQ, rg_is_k), kvmean_mu);
                    }

                    if (ragged_has_turbo2_tcq[ctx.device]) {
                        static bool tcq2_cb_loaded[GGML_CUDA_MAX_DEVICES] = {};
                        static int tcq2_hot_e = -1; if (tcq2_hot_e < 0) tcq2_hot_e = getenv("TURBO_TCQ_HOTSWAP") ? 1 : 0;
                        if (!tcq2_cb_loaded[ctx.device] || tcq2_hot_e) {
                            turbo2_tcq_load_kv_encode();
                            if (!tcq2_cb_loaded[ctx.device]) {
                                tcq2_cb_loaded[ctx.device] = true;
                                load_tcq_norm_alpha(ctx.device);
                                init_tcq_error_dump(ctx.device);
                            }
                        }
                        const int64_t qs1 = bpr * (int64_t) sizeof(block_turbo2_tcq);
                        const int64_t qs2 = ne1 * qs1;
                        const int64_t qs3 = ne2 * qs2;
                        ensure_ragged_tmp(ctx.device, &ragged_tcq_tmp2[ctx.device], &ragged_tcq_tmp2_bytes[ctx.device],
                                tmp_groups * (int64_t) sizeof(block_turbo2_tcq));
                        const int use_shared = ragged_tcq2_shared_bt<idx_t>(ctx.device);
                        if (!use_shared) ensure_tcq_bt_buf(ctx.device, n_blk_total * 128 * 64);
                        k_set_rows_turbo2_tcq<idx_t><<<(int)n_blk_total, 256, use_shared ? 128 * 64 : 0, stream>>>(
                            (const float *) src0_d, src1_d, (block_turbo2_tcq *) ragged_tcq_tmp2[ctx.device],
                            n_blk_total, tcq_bt_buf[ctx.device], use_shared, ne00, ne01, ne02, ne10, ne11, ne12, ne13,
                            s01_f, s02_f, s03_f, s10_i, s11_i, s12_i, rg_is_k, kvmean_mu, qs1, qs2, qs3,
                            ne00_fd, ne01_fd, ne02_fd, ne11_fd, ne12_fd);
                        k_ragged_turbo2_tcq_overlay<idx_t><<<num_blocks, CUDA_SET_ROWS_BLOCK_SIZE, 0, stream>>>(
                            src1_d, (const block_turbo2_tcq *) ragged_tcq_tmp2[ctx.device], (half *) dst->data, n_blk_total,
                            s10_i, s11_i, s12_i, qs1, qs2, qs3, s1, s2, s3,
                            bpr_fd, ne01_fd, ne02_fd, ne11_fd, ne12_fd, rg_layer, rg_is_k, ragged_v_rotated,
                            ragged_tcq_decode_alpha(RAGGED_TIER_TURBO2_TCQ, rg_is_k), kvmean_mu);
                    }
                }
            }
        } else {
            set_rows_cuda(
                src0_d, src1_d, (half*)dst->data,
                ne00, ne01, ne02, ne03,
                ne10, ne11, ne12, ne13,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream
            );
        }
    } else if (dst->type == GGML_TYPE_BF16) {
        set_rows_cuda(
            src0_d, src1_d, (nv_bfloat16*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q4_0) {
        set_rows_cuda_quant<idx_t, block_q4_0, QK4_0, quantize_f32_q4_0_block>(
            src0_d, src1_d, (block_q4_0*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q4_1) {
        set_rows_cuda_quant<idx_t, block_q4_1, QK4_1, quantize_f32_q4_1_block>(
            src0_d, src1_d, (block_q4_1*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q5_0) {
        set_rows_cuda_quant<idx_t, block_q5_0, QK5_0, quantize_f32_q5_0_block>(
            src0_d, src1_d, (block_q5_0*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q5_1) {
        set_rows_cuda_quant<idx_t, block_q5_1, QK5_1, quantize_f32_q5_1_block>(
            src0_d, src1_d, (block_q5_1*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q8_0) {
        set_rows_cuda_quant<idx_t, block_q8_0, QK8_0, quantize_f32_q8_0_block>(
            src0_d, src1_d, (block_q8_0*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_IQ4_NL) {
        set_rows_cuda_quant<idx_t, block_iq4_nl, QK4_NL, quantize_f32_iq4_nl_block>(
            src0_d, src1_d, (block_iq4_nl*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_TURBO2_0) {
        GGML_ASSERT(ne00 % QK_TURBO2_GROUP == 0);
        const int64_t ne_total_groups = (ne00 * ne01 * ne02 * ne03) / QK_TURBO2_GROUP;
        const int num_blocks_grid = (ne_total_groups + CUDA_SET_ROWS_BLOCK_SIZE - 1) / CUDA_SET_ROWS_BLOCK_SIZE;
        const int64_t s01_f = nb01/sizeof(float); const int64_t s02_f = nb02/sizeof(float); const int64_t s03_f = nb03/sizeof(float);
        const int64_t s10_i = nb10/sizeof(idx_t); const int64_t s11_i = nb11/sizeof(idx_t); const int64_t s12_i = nb12/sizeof(idx_t);
        if (ne_total_groups > 0 && ne00 > 0 && ne01 > 0 && ne02 > 0 && ne11 > 0 && ne12 > 0) {
            const uint3 ne00_fd = init_fastdiv_values((uint32_t) ne00);
            const uint3 ne01_fd = init_fastdiv_values((uint32_t) ne01);
            const uint3 ne02_fd = init_fastdiv_values((uint32_t) ne02);
            const uint3 ne11_fd = init_fastdiv_values((uint32_t) ne11);
            const uint3 ne12_fd = init_fastdiv_values((uint32_t) ne12);
            const float * kmean_mu = nullptr;
            if (ne00 <= PFHEAD_MAX_C) {
                const bool is_k = strncmp(dst->name, "cache_k_l", 9) == 0;
                const bool is_v = strncmp(dst->name, "cache_v_l", 9) == 0;
                if (is_k || is_v) {
                    const float * tbl = is_k ? turbo_kmean_table(ctx.device) : turbo_vmean_table_enc(ctx.device);
                    const int kl = atoi(dst->name + 9);
                    if (tbl && kl >= 0 && kl < PFHEAD_MAX_L) kmean_mu = tbl + (size_t) kl * PFHEAD_MAX_C;
                }
            }
            k_set_rows_turbo2<idx_t><<<num_blocks_grid, CUDA_SET_ROWS_BLOCK_SIZE, 0, stream>>>(
                src0_d, src1_d, (block_turbo2_0 *)dst->data,
                ne_total_groups, ne00, ne01, ne02, ne10, ne11, ne12, ne13,
                s01_f, s02_f, s03_f, s10_i, s11_i, s12_i, kmean_mu, nb1, nb2, nb3,
                ne00_fd, ne01_fd, ne02_fd, ne11_fd, ne12_fd);
        }
    } else if (dst->type == GGML_TYPE_TURBO3_0) {
        GGML_ASSERT(ne00 % QK_TURBO3_GROUP == 0);
        const int64_t ne_total_groups = (ne00 * ne01 * ne02 * ne03) / QK_TURBO3_GROUP;
        const int num_blocks_grid = (ne_total_groups + CUDA_SET_ROWS_BLOCK_SIZE - 1) / CUDA_SET_ROWS_BLOCK_SIZE;
        const int64_t s01_f = nb01/sizeof(float); const int64_t s02_f = nb02/sizeof(float); const int64_t s03_f = nb03/sizeof(float);
        const int64_t s10_i = nb10/sizeof(idx_t); const int64_t s11_i = nb11/sizeof(idx_t); const int64_t s12_i = nb12/sizeof(idx_t);
        const int iq_is_k = (strncmp(dst->name, "cache_k_", 8) == 0) ? 1 : 0;
        if (ne_total_groups > 0 && ne00 > 0 && ne01 > 0 && ne02 > 0 && ne11 > 0 && ne12 > 0) {
            const uint3 ne00_fd = init_fastdiv_values((uint32_t) ne00);
            const uint3 ne01_fd = init_fastdiv_values((uint32_t) ne01);
            const uint3 ne02_fd = init_fastdiv_values((uint32_t) ne02);
            const uint3 ne11_fd = init_fastdiv_values((uint32_t) ne11);
            const uint3 ne12_fd = init_fastdiv_values((uint32_t) ne12);
            int pf_layer = -1;
            if (strncmp(dst->name, "cache_k_l", 9) == 0 || strncmp(dst->name, "cache_v_l", 9) == 0) {
                pf_layer = atoi(dst->name + 9);
            }
            const float * kmean_mu = nullptr;
            if (pf_layer >= 0 && pf_layer < PFHEAD_MAX_L && ne00 <= PFHEAD_MAX_C) {
                const float * tbl = iq_is_k ? turbo_kmean_table(ctx.device) : turbo_vmean_table_enc(ctx.device);
                if (tbl) kmean_mu = tbl + (size_t) pf_layer * PFHEAD_MAX_C;
            }
            k_set_rows_turbo3<idx_t><<<num_blocks_grid, CUDA_SET_ROWS_BLOCK_SIZE, 0, stream>>>(
                src0_d, src1_d, (block_turbo3_0 *)dst->data,
                ne_total_groups, ne00, ne01, ne02, ne10, ne11, ne12, ne13,
                s01_f, s02_f, s03_f, s10_i, s11_i, s12_i, iq_is_k, kmean_mu, pf_layer, nb1, nb2, nb3,
                ne00_fd, ne01_fd, ne02_fd, ne11_fd, ne12_fd);
        }
    } else if (dst->type == GGML_TYPE_TURBO4_0) {
        load_turbo4_alpha(ctx.device);
        const float * mu = turbo_tap_mu(ctx, dst, ne00);
        // Fast path: cooperative 128-thread encode (see k_set_rows_turbo4_coop). Fall back to the
        // generic 1-thread-per-block path only when TURBO_EXTRACT post-rotation dumping is armed
        // (that path lives inside quantize_f32_turbo4_0_block).
        static const bool t4_extract = []() { const char * e = getenv("TURBO_EXTRACT"); return e && atoi(e) > 0; }();
        const int64_t t4_total = (ne00 * ne01 * ne02 * ne03) / QK_TURBO4;
        if (!t4_extract && (ne00 % QK_TURBO4) == 0 && t4_total > 0) {
            k_set_rows_turbo4_coop<idx_t><<<(unsigned)t4_total, QK_TURBO4, 0, stream>>>(
                src0_d, src1_d, (block_turbo4_0 *)dst->data,
                ne00, ne01, ne02, ne11, ne12,
                (int64_t)(nb01/sizeof(float)), (int64_t)(nb02/sizeof(float)), (int64_t)(nb03/sizeof(float)),
                (int64_t)(nb10/sizeof(idx_t)), (int64_t)(nb11/sizeof(idx_t)), (int64_t)(nb12/sizeof(idx_t)),
                (int64_t)nb1, (int64_t)nb2, (int64_t)nb3, mu);
        } else {
            set_rows_cuda_quant<idx_t, block_turbo4_0, QK_TURBO4, quantize_f32_turbo4_0_block>(
                src0_d, src1_d, (block_turbo4_0*)dst->data,
                ne00, ne01, ne02, ne03, ne10, ne11, ne12, ne13,
                nb01, nb02, nb03, nb10, nb11, nb12, nb1, nb2, nb3, stream, mu);
        }
    } else if (dst->type == GGML_TYPE_TURBO8_0) {
        // Fast path: cooperative 128-thread encode (see k_set_rows_turbo8_coop). Falls back to the
        // generic 1-thread-per-block path when TURBO_EXTRACT post-rotation dumping is armed.
        static const bool t8_extract = []() { const char * e = getenv("TURBO_EXTRACT"); return e && atoi(e) > 0; }();
        const int64_t t8_total = (ne00 * ne01 * ne02 * ne03) / QK_TURBO8;
        if (!t8_extract && (ne00 % QK_TURBO8) == 0 && t8_total > 0) {
            k_set_rows_turbo8_coop<idx_t><<<(unsigned)t8_total, QK_TURBO8, 0, stream>>>(
                src0_d, src1_d, (block_turbo8_0 *)dst->data,
                ne00, ne01, ne02, ne11, ne12,
                (int64_t)(nb01/sizeof(float)), (int64_t)(nb02/sizeof(float)), (int64_t)(nb03/sizeof(float)),
                (int64_t)(nb10/sizeof(idx_t)), (int64_t)(nb11/sizeof(idx_t)), (int64_t)(nb12/sizeof(idx_t)),
                (int64_t)nb1, (int64_t)nb2, (int64_t)nb3);
        } else {
            set_rows_cuda_quant<idx_t, block_turbo8_0, QK_TURBO8, quantize_f32_turbo8_0_block>(
                src0_d, src1_d, (block_turbo8_0*)dst->data,
                ne00, ne01, ne02, ne03, ne10, ne11, ne12, ne13,
                nb01, nb02, nb03, nb10, nb11, nb12, nb1, nb2, nb3, stream, turbo_tap_mu(ctx, dst, ne00));
        }
    } else if (dst->type == GGML_TYPE_TURBO3_TCQ) {
        GGML_ASSERT(ne00 % QK_TURBO3_TCQ == 0);
        const int64_t ne_total_groups = (ne00 * ne01 * ne02 * ne03) / QK_TURBO3_TCQ;
        // Runtime codebook loading: TURBO_TCQ_CB overrides compiled-in codebook (per-device)
        {
            static bool tcq_cb_loaded[GGML_CUDA_MAX_DEVICES] = {};
            static int tcq_hot_e = -1; if (tcq_hot_e < 0) tcq_hot_e = getenv("TURBO_TCQ_HOTSWAP") ? 1 : 0;
            if (!tcq_cb_loaded[ctx.device] || tcq_hot_e) {
                turbo_tcq_load_kv_encode();                 // self-gates on mtime under hotswap
                if (!tcq_cb_loaded[ctx.device]) {
                    tcq_cb_loaded[ctx.device] = true;
                    load_tcq_norm_alpha(ctx.device);
                    init_tcq_error_dump(ctx.device);
                }
            }
            {   // TURBO_TCQ_TIEHI tie-break probe (this TU's __device__ flag)
                static bool tiehi_set[GGML_CUDA_MAX_DEVICES] = {};
                if (!tiehi_set[ctx.device]) {
                    tiehi_set[ctx.device] = true;
                    static const int tiehi = getenv("TURBO_TCQ_TIEHI") ? 1 : 0;
                    if (tiehi) {
                        cudaMemcpyToSymbol(d_tcq_tiehi, &tiehi, sizeof(int));
                        fprintf(stderr, "TCQ encode: tie-break probe ON (prefer-high predecessor)\n");
                    }
                }
            }
        }
        // TCQ Viterbi encode: 512 threads per block. The TCQ3 backtrace stores
        // one predecessor for each 64-state low-bit group per step.
        const int64_t s01_f = nb01/sizeof(float); const int64_t s02_f = nb02/sizeof(float); const int64_t s03_f = nb03/sizeof(float);
        const int64_t s10_i = nb10/sizeof(idx_t); const int64_t s11_i = nb11/sizeof(idx_t); const int64_t s12_i = nb12/sizeof(idx_t);
        const int iq_is_k = (strncmp(dst->name, "cache_k_", 8) == 0) ? 1 : 0;
        if (ne_total_groups > 0 && ne00 > 0 && ne01 > 0 && ne02 > 0 && ne11 > 0 && ne12 > 0) {
            static int tcq3_use_shared_bt[GGML_CUDA_MAX_DEVICES] = {};
            static bool tcq3_bt_checked[GGML_CUDA_MAX_DEVICES] = {};
            constexpr int tcq3_bt_shared_bytes = 128 * 64;
            if (!tcq3_bt_checked[ctx.device]) {
                tcq3_bt_checked[ctx.device] = true;
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
                const char * tcq_shared_bt_env = getenv("TURBO_TCQ_SHARED_BT");
                if (!tcq_shared_bt_env || atoi(tcq_shared_bt_env) != 0) {
                    int max_shared_optin = 0;
                    CUDA_CHECK(cudaDeviceGetAttribute(&max_shared_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, ctx.device));
                    if (max_shared_optin >= tcq3_bt_shared_bytes) {
                        CUDA_SET_SHARED_MEMORY_LIMIT(k_set_rows_turbo3_tcq<idx_t>, tcq3_bt_shared_bytes);
                        tcq3_use_shared_bt[ctx.device] = 1;
                        fprintf(stderr, "TCQ encode: using shared-memory backtrace (%d bytes/block)\n", tcq3_bt_shared_bytes);
                    } else {
                        fprintf(stderr, "TCQ encode: shared-memory backtrace unavailable, only %d bytes/block are available\n", max_shared_optin);
                    }
                }
#endif
            }
            if (!tcq3_use_shared_bt[ctx.device]) {
                ensure_tcq_bt_buf(ctx.device, ne_total_groups * 128 * 64);
            }
            const uint3 ne00_fd = init_fastdiv_values((uint32_t) ne00);
            const uint3 ne01_fd = init_fastdiv_values((uint32_t) ne01);
            const uint3 ne02_fd = init_fastdiv_values((uint32_t) ne02);
            const uint3 ne11_fd = init_fastdiv_values((uint32_t) ne11);
            const uint3 ne12_fd = init_fastdiv_values((uint32_t) ne12);
            const int shared_bytes = tcq3_use_shared_bt[ctx.device] ? tcq3_bt_shared_bytes : 0;
            const float * kvmean_mu = nullptr;
            if (strncmp(dst->name, "cache_k_l", 9) == 0 || strncmp(dst->name, "cache_v_l", 9) == 0) {
                const int pf_layer = atoi(dst->name + 9);
                if (pf_layer >= 0 && pf_layer < PFHEAD_MAX_L && ne00 <= PFHEAD_MAX_C) {
                    const float * tbl = iq_is_k ? turbo_kmean_table(ctx.device) : turbo_vmean_table_enc(ctx.device);
                    if (tbl) kvmean_mu = tbl + (size_t) pf_layer * PFHEAD_MAX_C;
                }
            }
            k_set_rows_turbo3_tcq<idx_t><<<(int)ne_total_groups, TCQ3_ENC_NT, shared_bytes, stream>>>(
                src0_d, src1_d, (block_turbo3_tcq *)dst->data,
                ne_total_groups, tcq_bt_buf[ctx.device], tcq3_use_shared_bt[ctx.device], ne00, ne01, ne02, ne10, ne11, ne12, ne13,
                s01_f, s02_f, s03_f, s10_i, s11_i, s12_i, iq_is_k, kvmean_mu, nb1, nb2, nb3,
                ne00_fd, ne01_fd, ne02_fd, ne11_fd, ne12_fd);
        }
    } else if (dst->type == GGML_TYPE_TURBO2_TCQ) {
        GGML_ASSERT(ne00 % QK_TURBO2_TCQ == 0);
        const int64_t ne_total_groups = (ne00 * ne01 * ne02 * ne03) / QK_TURBO2_TCQ;
        // Runtime codebook loading: TURBO_TCQ_CB_K/_V (?? TURBO_TCQ_CB) override the compiled-in
        // 2-bit K/V codebooks (per-device). V defaults to a copy of the compiled-in K book.
        {
            static bool tcq2_cb_loaded[GGML_CUDA_MAX_DEVICES] = {};
            static int tcq2_hot_e = -1; if (tcq2_hot_e < 0) tcq2_hot_e = getenv("TURBO_TCQ_HOTSWAP") ? 1 : 0;
            if (!tcq2_cb_loaded[ctx.device] || tcq2_hot_e) {
                turbo2_tcq_load_kv_encode();                // self-gates on mtime under hotswap
                if (!tcq2_cb_loaded[ctx.device]) {
                    tcq2_cb_loaded[ctx.device] = true;
                    load_tcq_norm_alpha(ctx.device);
                    init_tcq_error_dump(ctx.device);
                }
            }
        }
        // 2-bit TCQ Viterbi encode: 256 threads per block. Compressed backtrace
        // stores one predecessor per 64 low-state groups per step (same as turbo3_tcq).
        const int64_t s01_f = nb01/sizeof(float); const int64_t s02_f = nb02/sizeof(float); const int64_t s03_f = nb03/sizeof(float);
        const int64_t s10_i = nb10/sizeof(idx_t); const int64_t s11_i = nb11/sizeof(idx_t); const int64_t s12_i = nb12/sizeof(idx_t);
        const int iq_is_k = (strncmp(dst->name, "cache_k_", 8) == 0) ? 1 : 0;
        if (ne_total_groups > 0 && ne00 > 0 && ne01 > 0 && ne02 > 0 && ne11 > 0 && ne12 > 0) {
            static int tcq2_use_shared_bt[GGML_CUDA_MAX_DEVICES] = {};
            static bool tcq2_bt_checked[GGML_CUDA_MAX_DEVICES] = {};
            constexpr int tcq2_bt_shared_bytes = 128 * 64;
            if (!tcq2_bt_checked[ctx.device]) {
                tcq2_bt_checked[ctx.device] = true;
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
                const char * tcq_shared_bt_env = getenv("TURBO_TCQ_SHARED_BT");
                if (!tcq_shared_bt_env || atoi(tcq_shared_bt_env) != 0) {
                    int max_shared_optin = 0;
                    CUDA_CHECK(cudaDeviceGetAttribute(&max_shared_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, ctx.device));
                    if (max_shared_optin >= tcq2_bt_shared_bytes) {
                        CUDA_SET_SHARED_MEMORY_LIMIT(k_set_rows_turbo2_tcq<idx_t>, tcq2_bt_shared_bytes);
                        tcq2_use_shared_bt[ctx.device] = 1;
                    }
                }
#endif
            }
            if (!tcq2_use_shared_bt[ctx.device]) {
                ensure_tcq_bt_buf(ctx.device, ne_total_groups * 128 * 64);
            }
            const uint3 ne00_fd = init_fastdiv_values((uint32_t) ne00);
            const uint3 ne01_fd = init_fastdiv_values((uint32_t) ne01);
            const uint3 ne02_fd = init_fastdiv_values((uint32_t) ne02);
            const uint3 ne11_fd = init_fastdiv_values((uint32_t) ne11);
            const uint3 ne12_fd = init_fastdiv_values((uint32_t) ne12);
            const int shared_bytes = tcq2_use_shared_bt[ctx.device] ? tcq2_bt_shared_bytes : 0;
            const float * kvmean_mu = nullptr;
            if (strncmp(dst->name, "cache_k_l", 9) == 0 || strncmp(dst->name, "cache_v_l", 9) == 0) {
                const int pf_layer = atoi(dst->name + 9);
                if (pf_layer >= 0 && pf_layer < PFHEAD_MAX_L && ne00 <= PFHEAD_MAX_C) {
                    const float * tbl = iq_is_k ? turbo_kmean_table(ctx.device) : turbo_vmean_table_enc(ctx.device);
                    if (tbl) kvmean_mu = tbl + (size_t) pf_layer * PFHEAD_MAX_C;
                }
            }
            k_set_rows_turbo2_tcq<idx_t><<<(int)ne_total_groups, 256, shared_bytes, stream>>>(
                src0_d, src1_d, (block_turbo2_tcq *)dst->data,
                ne_total_groups, tcq_bt_buf[ctx.device], tcq2_use_shared_bt[ctx.device], ne00, ne01, ne02, ne10, ne11, ne12, ne13,
                s01_f, s02_f, s03_f, s10_i, s11_i, s12_i, iq_is_k, kvmean_mu, nb1, nb2, nb3,
                ne00_fd, ne01_fd, ne02_fd, ne11_fd, ne12_fd);
        }
    } else if (dst->type == GGML_TYPE_TURBO1_TCQ) {
        GGML_ASSERT(ne00 % QK_TURBO1_TCQ == 0);
        const int64_t ne_total_groups = (ne00 * ne01 * ne02 * ne03) / QK_TURBO1_TCQ;
        {
            static bool tcq1_cb_loaded[GGML_CUDA_MAX_DEVICES] = {};
            static int tcq1_hot_e = -1; if (tcq1_hot_e < 0) tcq1_hot_e = getenv("TURBO1_TCQ_HOTSWAP") ? 1 : 0;
            if (!tcq1_cb_loaded[ctx.device] || tcq1_hot_e) {
                turbo1_tcq_load_kv_encode();
                tcq1_cb_loaded[ctx.device] = true;
            }
        }
        const int64_t s01_f = nb01/sizeof(float); const int64_t s02_f = nb02/sizeof(float); const int64_t s03_f = nb03/sizeof(float);
        const int64_t s10_i = nb10/sizeof(idx_t); const int64_t s11_i = nb11/sizeof(idx_t); const int64_t s12_i = nb12/sizeof(idx_t);
        const int iq_is_k = (strncmp(dst->name, "cache_k_", 8) == 0) ? 1 : 0;
        if (ne_total_groups > 0 && ne00 > 0 && ne01 > 0 && ne02 > 0 && ne11 > 0 && ne12 > 0) {
            static int tcq1_use_shared_bt[GGML_CUDA_MAX_DEVICES] = {};
            static bool tcq1_bt_checked[GGML_CUDA_MAX_DEVICES] = {};
            constexpr int tcq1_bt_shared_bytes = 128 * 128;   // k=1: 128 low-7-bit pred groups (vs 128*64 for k=2)
            if (!tcq1_bt_checked[ctx.device]) {
                tcq1_bt_checked[ctx.device] = true;
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
                const char * tcq_shared_bt_env = getenv("TURBO_TCQ_SHARED_BT");
                if (!tcq_shared_bt_env || atoi(tcq_shared_bt_env) != 0) {
                    int max_shared_optin = 0;
                    CUDA_CHECK(cudaDeviceGetAttribute(&max_shared_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, ctx.device));
                    if (max_shared_optin >= tcq1_bt_shared_bytes) {
                        CUDA_SET_SHARED_MEMORY_LIMIT(k_set_rows_turbo1_tcq<idx_t>, tcq1_bt_shared_bytes);
                        tcq1_use_shared_bt[ctx.device] = 1;
                    }
                }
#endif
            }
            if (!tcq1_use_shared_bt[ctx.device]) {
                ensure_tcq_bt_buf(ctx.device, ne_total_groups * 128 * 128);
            }
            const uint3 ne00_fd = init_fastdiv_values((uint32_t) ne00);
            const uint3 ne01_fd = init_fastdiv_values((uint32_t) ne01);
            const uint3 ne02_fd = init_fastdiv_values((uint32_t) ne02);
            const uint3 ne11_fd = init_fastdiv_values((uint32_t) ne11);
            const uint3 ne12_fd = init_fastdiv_values((uint32_t) ne12);
            const int shared_bytes = tcq1_use_shared_bt[ctx.device] ? tcq1_bt_shared_bytes : 0;
            const float * kvmean_mu = nullptr;
            if (strncmp(dst->name, "cache_k_l", 9) == 0 || strncmp(dst->name, "cache_v_l", 9) == 0) {
                const int pf_layer = atoi(dst->name + 9);
                if (pf_layer >= 0 && pf_layer < PFHEAD_MAX_L && ne00 <= PFHEAD_MAX_C) {
                    const float * tbl = iq_is_k ? turbo_kmean_table(ctx.device) : turbo_vmean_table_enc(ctx.device);
                    if (tbl) kvmean_mu = tbl + (size_t) pf_layer * PFHEAD_MAX_C;
                }
            }
            k_set_rows_turbo1_tcq<idx_t><<<(int)ne_total_groups, 256, shared_bytes, stream>>>(
                src0_d, src1_d, (block_turbo1_tcq *)dst->data,
                ne_total_groups, tcq_bt_buf[ctx.device], tcq1_use_shared_bt[ctx.device], iq_is_k, kvmean_mu,
                s01_f, s02_f, s03_f, s10_i, s11_i, s12_i, nb1, nb2, nb3,
                ne00_fd, ne01_fd, ne02_fd, ne11_fd, ne12_fd);
        }
    } else {
        GGML_ABORT("unsupported type %s", ggml_type_name(dst->type));
    }
}

template<>
void set_rows_cuda<half, int32_t>(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const half    * src0_d = (const half *)src0->data;
    const int32_t * src1_d = (const int32_t *)src1->data;

    GGML_TENSOR_BINARY_OP_LOCALS

    cudaStream_t stream = ctx.stream();


    if (dst->type == GGML_TYPE_F16) {
        set_rows_cuda(
            src0_d, src1_d, (half*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else {
        GGML_ABORT("unsupported type %s", ggml_type_name(dst->type));
    }
}

template<>
void set_rows_cuda<half, int64_t>(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const half    * src0_d = (const half *)src0->data;
    const int64_t * src1_d = (const int64_t *)src1->data;

    GGML_TENSOR_BINARY_OP_LOCALS

    cudaStream_t stream = ctx.stream();


    if (dst->type == GGML_TYPE_F16) {
        set_rows_cuda(
            src0_d, src1_d, (half*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else {
        GGML_ABORT("unsupported type %s", ggml_type_name(dst->type));
    }
}


// InnerQ calibration state machine (driven by TURBO_INNERQ env var)
static int innerq_state = 0; // 0=uninit, 1=calibrating, 2=active, -1=disabled
static int innerq_tokens_seen = 0;
// Total set_rows tokens across all layers; TURBO_INNERQ_CAL_TOKENS overrides (PFHEAD probe
// dumps need longer windows so every KV layer gets enough rows for stable per-channel means).
static int innerq_calibration_tokens() {
    static int v = 0;
    if (v == 0) {
        const char * env = getenv("TURBO_INNERQ_CAL_TOKENS");
        v = (env && atoi(env) > 0) ? atoi(env) : 100000;
    }
    return v;
}
#define INNERQ_CALIBRATION_TOKENS innerq_calibration_tokens()

void ggml_cuda_op_set_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F16));
    GGML_ASSERT(src1->type == GGML_TYPE_I64 || src1->type == GGML_TYPE_I32);

    turbo_vanilla_cb_load_encode();  // TURBO_CB_T2/3/4/8: this TU's encode centroids + mids

    if (cert_state != -1) {
        cert_dump_rows(ctx, src0, dst->name);
    }

    // Post-rotation extraction: thread-safe one-time init
    if (h_extract_state == 0 && (dst->type == GGML_TYPE_TURBO3_0 || dst->type == GGML_TYPE_TURBO4_0 || dst->type == GGML_TYPE_TURBO8_0 || dst->type == GGML_TYPE_TURBO3_TCQ || dst->type == GGML_TYPE_TURBO2_TCQ || dst->type == GGML_TYPE_TURBO1_TCQ)) {
        std::call_once(h_extract_init_flag, []() {
            const char * env = getenv("TURBO_EXTRACT");
            if (env && atoi(env) > 0) {
                turbo_extract_init(atoi(env));
            } else {
                h_extract_state = -1;
            }
        });
    }
    // Check if extraction buffer is full
    if (h_extract_state == 1) {
        std::lock_guard<std::mutex> lock(h_extract_check_mutex);
        turbo_extract_check_done();
    }

    // InnerQ: one-time init on first turbo SET_ROWS call. turbo1_tcq must be here too: its K
    // inverse-FWHT decode multiplies by d_innerq_channel_scale_inv_fattn, which stays
    // zero-initialized (→ K decoded as all-zeros) unless turbo_innerq_init() uploads identity.
    if (innerq_state == 0 && (dst->type == GGML_TYPE_TURBO2_0 || dst->type == GGML_TYPE_TURBO3_0 || dst->type == GGML_TYPE_TURBO4_0 || dst->type == GGML_TYPE_TURBO8_0 || dst->type == GGML_TYPE_TURBO3_TCQ || dst->type == GGML_TYPE_TURBO2_TCQ || dst->type == GGML_TYPE_TURBO1_TCQ)) {
        static const char * env = getenv("TURBO_INNERQ");
        if (env && atoi(env) > 0) {
            turbo_innerq_init();
            turbo_innerq_start_calibration();
            innerq_state = 1;
            fprintf(stderr, "InnerQ: calibration started (collecting %d tokens)\n", INNERQ_CALIBRATION_TOKENS);
        } else {
            turbo_innerq_init(); // identity scales
            innerq_state = -1;
        }
    }

    // Track calibration progress
    if (innerq_state == 1 && (dst->type == GGML_TYPE_TURBO3_0 || dst->type == GGML_TYPE_TURBO4_0 || dst->type == GGML_TYPE_TURBO8_0 || dst->type == GGML_TYPE_TURBO3_TCQ || dst->type == GGML_TYPE_TURBO2_TCQ)) {
        innerq_tokens_seen += dst->src[0]->ne[1];
        if (innerq_tokens_seen >= INNERQ_CALIBRATION_TOKENS) {
            turbo_innerq_finalize_calibration();
            innerq_state = 2;
            fprintf(stderr, "InnerQ: calibration complete, scales active\n");
        }
    }

    if (src0->type == GGML_TYPE_F32) {
        if (src1->type == GGML_TYPE_I64) {
            set_rows_cuda<float, int64_t>(ctx, src0, src1, dst);
        } else {
            set_rows_cuda<float, int32_t>(ctx, src0, src1, dst);
        }
    } else if (src0->type == GGML_TYPE_F16) {
        if (src1->type == GGML_TYPE_I64) {
            set_rows_cuda<half, int64_t>(ctx, src0, src1, dst);
        } else {
            set_rows_cuda<half, int32_t>(ctx, src0, src1, dst);
        }
    } else {
        GGML_ABORT("unsupported type %s", ggml_type_name(src0->type));
    }
}
