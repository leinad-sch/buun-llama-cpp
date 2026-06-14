// pflash-dump: offline PFlash importance harness for EXP-15d (token/content axis).
//
// Loads the Qwen3-0.6B proxy scorer as a normal llama_model, runs it over fixed
// n_ctx windows of a corpus, and captures the real post-RoPE per-layer Q/K via the
// llama eval callback (cb_eval) — the same hook lucebox's kvzap-test uses. From Q/K
// it computes the PFlash "attention-received" importance per 128-token block
// (mean-K per block, dot Q, max over GQA heads + over query rows, summed over the
// causal query blocks and over all layers). One importance value per (window, block).
//
// Output: a binary dump consumed by gen_masks.py to build the per-config RCM tier
// masks (TURBO_RAGGED_CONTENT_MASK) for configs 3/4/5.
//
// Windows are decoded with the KV cache cleared between them and RoPE positions
// 0..n_ctx-1, mirroring llama-perplexity's per-chunk eval, so block b covers
// positions [b*128, (b+1)*128) — directly aligned with the KLD content mask.

#include "llama.h"
#include "ggml.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <fstream>
#include <algorithm>
#include <chrono>

static constexpr int BLOCK_SZ = 128;

struct qk_capture {
    int n_layers  = 0;
    int n_head    = 0;  // from model API (reliable)
    int n_kv_head = 0;  // from model API
    int d_head    = 0;  // discovered from the first accepted 3D tensor's ne[0]
    int n_ctx     = 0;
    int cur_pos   = 0;  // start position of the batch currently being decoded
    bool skip_q   = false;  // head-stats mode only needs K; don't request/store Q
    // Per-layer fixed buffers for the CURRENT window, written by absolute position
    // (idempotent overwrite — the graph fires each named Qcur/Kcur node several times).
    // Q[l] layout: [pos][head][d]   K[l] layout: [pos][kv_head][d]
    std::vector<std::vector<float>> Q;
    std::vector<std::vector<float>> K;

    void ensure_alloc() {
        if (d_head == 0 || (int) K.size() != n_layers) return;
        const size_t qn = (size_t) n_ctx * n_head    * d_head;
        const size_t kn = (size_t) n_ctx * n_kv_head * d_head;
        if (!skip_q) for (auto & v : Q) if (v.size() != qn) v.assign(qn, 0.0f);
        for (auto & v : K) if (v.size() != kn) v.assign(kn, 0.0f);
    }
    void reset_window() {
        if (!skip_q) for (auto & v : Q) std::fill(v.begin(), v.end(), 0.0f);
        for (auto & v : K) std::fill(v.begin(), v.end(), 0.0f);
    }
};

// Parse "Qcur-<il>" / "Kcur-<il>"; returns layer index or -1, sets is_k. Rejects the
// auto-named "Qcur-N (view)" / reshape duplicates (anything with a trailing space/suffix).
static int parse_qk_name(const char * name, bool & is_k) {
    const char * p = nullptr;
    if      (strncmp(name, "Qcur-", 5) == 0) { is_k = false; p = name + 5; }
    else if (strncmp(name, "Kcur-", 5) == 0) { is_k = true;  p = name + 5; }
    else return -1;
    int il = 0;
    if (*p < '0' || *p > '9') return -1;
    for (; *p; p++) {
        if (*p < '0' || *p > '9') return -1; // suffix like " (view)" -> reject
        il = il * 10 + (*p - '0');
    }
    return il;
}

static bool qk_eval_callback(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * cap = (qk_capture *) user_data;
    bool is_k;
    const int il = parse_qk_name(t->name, is_k);

    if (ask) {
        // request only the per-layer Qcur/Kcur nodes (K only in head-stats mode)
        return il >= 0 && il < cap->n_layers && (is_k || !cap->skip_q);
    }
    if (il < 0 || il >= cap->n_layers) return true;
    if (!is_k && cap->skip_q) return true;
    if (t->type != GGML_TYPE_F32) return true;

    const int heads = is_k ? cap->n_kv_head : cap->n_head;
    // Accept only the post-RoPE 3D form [d_head, heads, n_tokens]; skip the 2D
    // pre-reshape node and any views that share the same name.
    if (t->ne[1] != heads || t->ne[2] < 1 || t->ne[3] != 1) return true;

    const int dh = (int) t->ne[0];
    const int T  = (int) t->ne[2];
    if (cap->d_head == 0) { cap->d_head = dh; cap->ensure_alloc(); }
    if (dh != cap->d_head) return true;

    auto & dst = is_k ? cap->K[il] : cap->Q[il];
    const size_t per_tok = (size_t) heads * dh;
    const size_t off     = (size_t) cap->cur_pos * per_tok;
    if (off + (size_t) T * per_tok > dst.size()) return true; // out of range guard
    ggml_backend_tensor_get(t, dst.data() + off, 0, (size_t) T * per_tok * sizeof(float));
    return true;
}

// Accumulate this window's per-block importance into imp_out[n_blocks].
static void reduce_window_importance(const qk_capture & cap, int seq_len,
        std::vector<float> & imp_out) {
    const int n_head       = cap.n_head;
    const int n_kv_head    = cap.n_kv_head;
    const int d_head       = cap.d_head;
    const int heads_per_kv = n_head / n_kv_head;
    const int n_blocks     = (seq_len + BLOCK_SZ - 1) / BLOCK_SZ;
    const float scale      = 1.0f / sqrtf((float) d_head);

    imp_out.assign(n_blocks, 0.0f);

    for (int l = 0; l < cap.n_layers; l++) {
        const float * Q = cap.Q[l].data(); // [pos][head][d]
        const float * K = cap.K[l].data(); // [pos][kv_head][d]

        // mean-K per (block, kv_head): [n_blocks * n_kv_head * d_head]
        std::vector<float> meanK((size_t) n_blocks * n_kv_head * d_head, 0.0f);
        for (int b = 0; b < n_blocks; b++) {
            const int t0 = b * BLOCK_SZ;
            const int t1 = std::min(t0 + BLOCK_SZ, seq_len);
            const int cnt = t1 - t0;
            if (cnt <= 0) continue;
            for (int h = 0; h < n_kv_head; h++) {
                float * mk = &meanK[((size_t) b * n_kv_head + h) * d_head];
                for (int t = t0; t < t1; t++) {
                    const float * kp = &K[((size_t) t * n_kv_head + h) * d_head];
                    for (int d = 0; d < d_head; d++) mk[d] += kp[d];
                }
                const float inv = 1.0f / (float) cnt;
                for (int d = 0; d < d_head; d++) mk[d] *= inv;
            }
        }

        // attention received by each kv block, summed over causal query blocks + heads.
        std::vector<float> imp_l(n_blocks, 0.0f);
        #pragma omp parallel for schedule(dynamic)
        for (int b = 0; b < n_blocks; b++) {
            float acc = 0.0f;
            for (int h = 0; h < n_kv_head; h++) {
                const float * mk = &meanK[((size_t) b * n_kv_head + h) * d_head];
                for (int qb = b; qb < n_blocks; qb++) {  // causal: query block >= key block
                    const int qt0 = qb * BLOCK_SZ;
                    const int qt1 = std::min(qt0 + BLOCK_SZ, seq_len);
                    float block_max = -INFINITY;
                    for (int t = qt0; t < qt1; t++) {
                        for (int hoff = 0; hoff < heads_per_kv; hoff++) {
                            const int qhead = h * heads_per_kv + hoff;
                            const float * qp = &Q[((size_t) t * n_head + qhead) * d_head];
                            float dot = 0.0f;
                            for (int d = 0; d < d_head; d++) dot += qp[d] * mk[d];
                            dot *= scale;
                            if (dot > block_max) block_max = dot;
                        }
                    }
                    if (block_max > -INFINITY) acc += block_max;
                }
            }
            imp_l[b] = acc;
        }
        for (int b = 0; b < n_blocks; b++) imp_out[b] += imp_l[b];
    }
}

// Accumulate this window's per-TOKEN importance into imp_out[seq_len].
// Full O(T^2) attention: each key token t gets the attention it receives, summed over
// every causal query q>=t (no block pooling on either side, no mean-K) — max over the
// GQA query heads sharing kv_head h, summed over kv_heads, summed over layers. This is
// the generous test: block-mean-K dilutes a sparse outlier key by 1/128, per-token does
// not. Cost is real attention compute (minutes), not the block shortcut's seconds.
static void reduce_window_importance_pertoken(const qk_capture & cap, int seq_len,
        std::vector<float> & imp_out) {
    const int n_head       = cap.n_head;
    const int n_kv_head    = cap.n_kv_head;
    const int d_head       = cap.d_head;
    const int heads_per_kv = n_head / n_kv_head;
    const float scale      = 1.0f / sqrtf((float) d_head);

    imp_out.assign(seq_len, 0.0f);

    for (int l = 0; l < cap.n_layers; l++) {
        const float * Q = cap.Q[l].data(); // [pos][head][d]
        const float * K = cap.K[l].data(); // [pos][kv_head][d]

        std::vector<float> imp_l(seq_len, 0.0f);
        #pragma omp parallel for schedule(dynamic)
        for (int t = 0; t < seq_len; t++) {           // key token
            float acc = 0.0f;
            for (int h = 0; h < n_kv_head; h++) {
                const float * kp = &K[((size_t) t * n_kv_head + h) * d_head];
                for (int q = t; q < seq_len; q++) {   // causal: query pos >= key pos
                    float best = -INFINITY;
                    for (int hoff = 0; hoff < heads_per_kv; hoff++) {
                        const int qhead = h * heads_per_kv + hoff;
                        const float * qp = &Q[((size_t) q * n_head + qhead) * d_head];
                        float dot = 0.0f;
                        for (int d = 0; d < d_head; d++) dot += qp[d] * kp[d];
                        dot *= scale;
                        if (dot > best) best = dot;
                    }
                    acc += best;
                }
            }
            imp_l[t] = acc;
        }
        for (int t = 0; t < seq_len; t++) imp_out[t] += imp_l[t];
    }
}

// Per-(layer, kv_head) K-covariance accumulator for the EXP-15e head d_eff gate.
// Stores running first + second moments of the post-RoPE per-head K vectors, sampled
// every `stride` positions. d_eff is computed at the end as the participation ratio
// PR = tr(C)^2 / ||C||_F^2 (rotation-invariant, so FWHT/eigenbasis don't matter; no
// eigensolve needed). PR ~= d_head means the head fills its dims (protect, hard to
// quant); PR << d_head means variance concentrates in few dims (tolerates aggression).
struct head_cov {
    int n_layers = 0, n_kv_head = 0, d = 0;
    std::vector<double> sum;    // [L*H*d]   running Σ k
    std::vector<double> sumsq;  // [L*H*d*d] running Σ k k^T
    std::vector<long>   count;  // [L*H]
    void init(int L, int H, int dh) {
        n_layers = L; n_kv_head = H; d = dh;
        sum.assign((size_t) L * H * dh, 0.0);
        sumsq.assign((size_t) L * H * dh * dh, 0.0);
        count.assign((size_t) L * H, 0);
    }
};

static void accumulate_head_cov(const qk_capture & cap, int seq_len, int stride,
        head_cov & hc) {
    const int H = cap.n_kv_head, d = cap.d_head;
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int l = 0; l < cap.n_layers; l++) {
        for (int h = 0; h < H; h++) {
            const float * K = cap.K[l].data();
            double * s  = &hc.sum[((size_t) l * H + h) * d];
            double * sq = &hc.sumsq[((size_t) l * H + h) * d * d];
            long cnt = 0;
            for (int t = 0; t < seq_len; t += stride) {
                const float * kp = &K[((size_t) t * H + h) * d];
                for (int i = 0; i < d; i++) {
                    const double ki = kp[i];
                    s[i] += ki;
                    double * sqi = &sq[(size_t) i * d];
                    for (int j = 0; j < d; j++) sqi[j] += ki * kp[j];
                }
                cnt++;
            }
            hc.count[(size_t) l * H + h] += cnt;
        }
    }
}

// Finalize: emit CSV "layer,kv_head,n_samples,total_var,d_eff,top_coord_frac".
static void finalize_head_stats(const head_cov & hc, const std::string & out_path) {
    const int L = hc.n_layers, H = hc.n_kv_head, d = hc.d;
    std::ofstream out(out_path);
    out << "layer,kv_head,n_samples,total_var,d_eff,top_coord_frac\n";
    for (int l = 0; l < L; l++) {
        for (int h = 0; h < H; h++) {
            const long n = hc.count[(size_t) l * H + h];
            const double * s  = &hc.sum[((size_t) l * H + h) * d];
            const double * sq = &hc.sumsq[((size_t) l * H + h) * d * d];
            if (n < 2) { out << l << "," << h << ",0,0,0,0\n"; continue; }
            const double inv = 1.0 / (double) n;
            // C_ij = E[k_i k_j] - E[k_i] E[k_j]; trC and ||C||_F^2 and max diag.
            double trC = 0.0, froF2 = 0.0, maxdiag = 0.0;
            for (int i = 0; i < d; i++) {
                const double mi = s[i] * inv;
                for (int j = 0; j < d; j++) {
                    const double cij = sq[(size_t) i * d + j] * inv - mi * (s[j] * inv);
                    froF2 += cij * cij;
                    if (i == j) {
                        trC += cij;
                        if (cij > maxdiag) maxdiag = cij;
                    }
                }
            }
            const double d_eff = froF2 > 0 ? (trC * trC) / froF2 : 0.0;
            const double top   = trC > 0 ? maxdiag / trC : 0.0;
            out << l << "," << h << "," << n << ","
                << trC << "," << d_eff << "," << top << "\n";
        }
    }
    out.close();
    fprintf(stderr, "pflash-dump: wrote head-stats CSV %s (%d layers x %d kv_heads, d=%d)\n",
        out_path.c_str(), L, H, d);
}

static void usage(const char * prog) {
    fprintf(stderr,
        "usage: %s -m <scorer.gguf> -f <corpus.txt> -o <out.bin>\n"
        "          [-c n_ctx=8192] [-b n_batch=2048] [-ngl 99]\n"
        "          [--windows N] [--per-token]\n"
        "          [--head-stats] [--head-stride N=4]   # EXP-15e: per-head K d_eff CSV\n", prog);
}

int main(int argc, char ** argv) {
    std::string model_path, corpus_path, out_path;
    int n_ctx = 8192, n_batch = 2048, n_gpu_layers = 99;
    int max_windows = -1;
    bool per_token = false;
    bool head_stats = false;
    int head_stride = 4;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> const char * { return (i + 1 < argc) ? argv[++i] : ""; };
        if      (a == "-m")          model_path  = next();
        else if (a == "-f")          corpus_path = next();
        else if (a == "-o")          out_path    = next();
        else if (a == "-c")          n_ctx       = atoi(next());
        else if (a == "-b")          n_batch     = atoi(next());
        else if (a == "-ngl")        n_gpu_layers= atoi(next());
        else if (a == "--windows")   max_windows = atoi(next());
        else if (a == "--per-token") per_token   = true;
        else if (a == "--head-stats") head_stats = true;
        else if (a == "--head-stride") head_stride = atoi(next());
        else { usage(argv[0]); return 1; }
    }
    if (model_path.empty() || corpus_path.empty() || out_path.empty()) { usage(argv[0]); return 1; }

    llama_backend_init();

    auto mparams = llama_model_default_params();
    mparams.n_gpu_layers = n_gpu_layers;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) { fprintf(stderr, "pflash-dump: failed to load %s\n", model_path.c_str()); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    // read + tokenize the whole corpus (BOS prepended), matching llama-perplexity windowing.
    std::string text;
    { std::ifstream f(corpus_path);
      if (!f) { fprintf(stderr, "pflash-dump: cannot open %s\n", corpus_path.c_str()); return 1; }
      text.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()); }

    std::vector<llama_token> all_tokens(text.size() + 256);
    int n_all = llama_tokenize(vocab, text.c_str(), text.size(),
        all_tokens.data(), all_tokens.size(), true, false);
    if (n_all < 0) {
        all_tokens.resize(-n_all);
        n_all = llama_tokenize(vocab, text.c_str(), text.size(),
            all_tokens.data(), all_tokens.size(), true, false);
    }
    all_tokens.resize(n_all);
    int n_windows = n_all / n_ctx;
    if (max_windows > 0 && n_windows > max_windows) n_windows = max_windows;
    fprintf(stderr, "pflash-dump: %d tokens -> %d windows of %d\n", n_all, n_windows, n_ctx);
    if (n_windows < 1) { fprintf(stderr, "pflash-dump: corpus too short\n"); return 1; }

    qk_capture cap;
    cap.n_layers  = llama_model_n_layer(model);
    cap.n_head    = llama_model_n_head(model);
    cap.n_kv_head = llama_model_n_head_kv(model);
    cap.n_ctx     = n_ctx;
    cap.skip_q    = head_stats;  // head d_eff gate needs K only
    // d_head is discovered from the first accepted 3D tensor (n_embd_head_k isn't exposed,
    // and n_embd/n_head is wrong for Qwen3); buffers are allocated then.
    cap.Q.resize(cap.n_layers);
    cap.K.resize(cap.n_layers);

    auto cparams = llama_context_default_params();
    cparams.n_ctx           = n_ctx;
    cparams.n_batch         = n_batch;
    cparams.n_ubatch        = n_batch;
    cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED; // keep Qcur/Kcur as plain F32 nodes
    cparams.cb_eval         = qk_eval_callback;
    cparams.cb_eval_user_data = &cap;
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) { fprintf(stderr, "pflash-dump: failed to create context\n"); return 1; }

    const int block_sz = per_token ? 1 : BLOCK_SZ;
    const int n_units  = per_token ? n_ctx : (n_ctx + BLOCK_SZ - 1) / BLOCK_SZ;
    std::vector<float> all_imp((size_t) n_windows * n_units, 0.0f);
    head_cov hc;
    if (head_stats)
        fprintf(stderr, "pflash-dump: HEAD-STATS mode (K-only d_eff gate, stride %d)\n", head_stride);
    else
        fprintf(stderr, "pflash-dump: %s reduction, %d units/window\n",
            per_token ? "PER-TOKEN (O(T^2) full attention)" : "per-128-block", n_units);

    for (int w = 0; w < n_windows; w++) {
        auto t0 = std::chrono::high_resolution_clock::now();
        cap.reset_window();
        llama_memory_clear(llama_get_memory(ctx), true);

        llama_token * wtok = all_tokens.data() + (size_t) w * n_ctx;
        for (int pos = 0; pos < n_ctx; pos += n_batch) {
            int n_eval = std::min(n_batch, n_ctx - pos);
            cap.cur_pos = pos;
            llama_batch batch = llama_batch_get_one(wtok + pos, n_eval);
            if (llama_decode(ctx, batch) != 0) {
                fprintf(stderr, "pflash-dump: decode failed window %d pos %d\n", w, pos);
                return 1;
            }
        }

        if (w == 0) {
            if (cap.d_head == 0) {
                fprintf(stderr, "pflash-dump: no 3D Qcur/Kcur captured — tensor names/shape differ "
                    "(expected 'Qcur-<il>' with ne[1]==n_head)\n");
                return 1;
            }
            fprintf(stderr, "pflash-dump: model %d layers, %d heads (%d kv), d_head %d\n",
                cap.n_layers, cap.n_head, cap.n_kv_head, cap.d_head);
            if (head_stats) hc.init(cap.n_layers, cap.n_kv_head, cap.d_head);
        }

        if (head_stats) {
            accumulate_head_cov(cap, n_ctx, head_stride, hc);
        } else {
            std::vector<float> imp;
            if (per_token) reduce_window_importance_pertoken(cap, n_ctx, imp);
            else           reduce_window_importance(cap, n_ctx, imp);
            std::copy(imp.begin(), imp.end(), all_imp.begin() + (size_t) w * n_units);
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        fprintf(stderr, "pflash-dump: window %d/%d done (%.1fs)\n", w + 1, n_windows,
            std::chrono::duration<double>(t1 - t0).count());
    }

    if (head_stats) {
        finalize_head_stats(hc, out_path);
    } else {
        // write dump: magic 'PFI1', int32 n_windows, n_ctx, block_size, n_units, then floats
        std::ofstream out(out_path, std::ios::binary);
        const int32_t magic = 0x31494650; // 'PFI1' little-endian
        int32_t hdr[5] = { magic, n_windows, n_ctx, block_sz, n_units };
        out.write((const char *) hdr, sizeof(hdr));
        out.write((const char *) all_imp.data(), all_imp.size() * sizeof(float));
        out.close();
        fprintf(stderr, "pflash-dump: wrote %s (%d windows x %d units, block_sz %d)\n",
            out_path.c_str(), n_windows, n_units, block_sz);
    }

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
