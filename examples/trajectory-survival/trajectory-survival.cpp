// Trajectory-survival metric harness (roadmap P1.1 / Codex §3.1).
//
// Paired GREEDY autoregressive decode: a reference context (F16/F16 KV) and a
// quant context (e.g. turbo3_tcq) decode the SAME prompt in lockstep. We record
// the first generation step at which their greedy argmax tokens differ
// (first-divergence). Up to that step both models are fed the identical shared
// token, so their KV caches hold the same token sequence but different quantized
// representations -- the divergence is purely accumulated KV-quant error.
//
// Anchor check: run with -ctk f16 -ctv f16 (quant == reference) -> must NEVER
// diverge (survival == 1.0, first-div == H for every prompt). turbo8 should
// survive far longer than turbo3.

#include "llama.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

static void print_usage(char ** argv) {
    fprintf(stderr,
        "\nusage: %s -m model.gguf [-f prompts.txt] [-n n_predict] [-ctk type] [-ctv type]\n"
        "         [-ngl n] [--n-prefix K] [--max-prompts N]\n"
        "  reference KV is always f16/f16; -ctk/-ctv select the quant KV to compare.\n"
        "  prompts.txt: one prompt per line (blank lines skipped).\n\n",
        argv[0]);
}

// Replicate common/arg.cpp::kv_cache_type_from_str without linking common:
// match the requested string against ggml_type_name over the type table.
static ggml_type type_from_str(const std::string & s) {
    for (int i = 0; i < GGML_TYPE_COUNT; i++) {
        const ggml_type t = (ggml_type) i;
        const char * n = ggml_type_name(t);
        if (n && s == n) {
            return t;
        }
    }
    fprintf(stderr, "error: unsupported KV cache type '%s'\n", s.c_str());
    exit(1);
}

static llama_token greedy_argmax(llama_context * ctx, int32_t n_vocab) {
    const float * logits = llama_get_logits_ith(ctx, -1);
    llama_token best = 0;
    float best_l = logits[0];
    for (int32_t i = 1; i < n_vocab; i++) {
        if (logits[i] > best_l) {
            best_l = logits[i];
            best   = i;
        }
    }
    return best;
}

int main(int argc, char ** argv) {
    std::string model_path;
    std::string prompts_path;
    std::string ctk = "turbo3_tcq";
    std::string ctv = "turbo3_tcq";
    int ngl         = 99;
    int n_predict   = 128;
    int n_prefix    = 0;    // 0 = use full prompt
    int max_prompts = 0;    // 0 = no cap

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&](const char * name) -> std::string {
            if (i + 1 >= argc) { fprintf(stderr, "error: %s needs a value\n", name); exit(1); }
            return argv[++i];
        };
        if      (a == "-m")            model_path   = next("-m");
        else if (a == "-f")            prompts_path = next("-f");
        else if (a == "-ctk")          ctk          = next("-ctk");
        else if (a == "-ctv")          ctv          = next("-ctv");
        else if (a == "-n")            n_predict    = std::stoi(next("-n"));
        else if (a == "-ngl")          ngl          = std::stoi(next("-ngl"));
        else if (a == "--n-prefix")    n_prefix     = std::stoi(next("--n-prefix"));
        else if (a == "--max-prompts") max_prompts  = std::stoi(next("--max-prompts"));
        else { print_usage(argv); return 1; }
    }
    if (model_path.empty()) { print_usage(argv); return 1; }

    // collect prompts
    std::vector<std::string> prompts;
    if (!prompts_path.empty()) {
        std::ifstream f(prompts_path);
        if (!f) { fprintf(stderr, "error: cannot open prompts file '%s'\n", prompts_path.c_str()); return 1; }
        std::string line;
        while (std::getline(f, line)) {
            if (!line.empty()) prompts.push_back(line);
            if (max_prompts > 0 && (int) prompts.size() >= max_prompts) break;
        }
    } else {
        prompts.push_back("The history of the Roman Empire is a long and complex story that begins");
    }
    if (prompts.empty()) { fprintf(stderr, "error: no prompts\n"); return 1; }

    ggml_backend_load_all();

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = ngl;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), model_params);
    if (!model) { fprintf(stderr, "error: unable to load model\n"); return 1; }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);

    // tokenize all prompts up front; track max prefix length to size the context
    std::vector<std::vector<llama_token>> toks(prompts.size());
    int max_n_prompt = 1;
    for (size_t p = 0; p < prompts.size(); p++) {
        const std::string & s = prompts[p];
        int n = -llama_tokenize(vocab, s.c_str(), s.size(), NULL, 0, true, true);
        std::vector<llama_token> t(n);
        if (llama_tokenize(vocab, s.c_str(), s.size(), t.data(), t.size(), true, true) < 0) {
            fprintf(stderr, "error: tokenize failed for prompt %zu\n", p);
            return 1;
        }
        if (n_prefix > 0 && (int) t.size() > n_prefix) t.resize(n_prefix);
        if (t.empty()) t.push_back(llama_vocab_bos(vocab));
        max_n_prompt = std::max(max_n_prompt, (int) t.size());
        toks[p] = std::move(t);
    }

    const int n_ctx = max_n_prompt + n_predict + 8;

    auto make_ctx = [&](ggml_type tk, ggml_type tv) -> llama_context * {
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx   = n_ctx;
        cp.n_batch = max_n_prompt;
        cp.type_k  = tk;
        cp.type_v  = tv;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;  // turbo KV consumed in fused-MMA FA kernels
        cp.no_perf = true;
        return llama_init_from_model(model, cp);
    };

    llama_context * ctx_ref = make_ctx(GGML_TYPE_F16, GGML_TYPE_F16);
    llama_context * ctx_q   = make_ctx(type_from_str(ctk), type_from_str(ctv));
    if (!ctx_ref || !ctx_q) { fprintf(stderr, "error: failed to create contexts\n"); return 1; }

    llama_memory_t mem_ref = llama_get_memory(ctx_ref);
    llama_memory_t mem_q   = llama_get_memory(ctx_q);

    printf("# trajectory-survival  ref=f16/f16  quant=%s/%s  H=%d  prompts=%zu  n_ctx=%d\n",
           ctk.c_str(), ctv.c_str(), n_predict, prompts.size(), n_ctx);
    printf("# TRAJ <idx> <n_prompt> <first_div> <diverged>\n");

    std::vector<int> first_div(prompts.size(), n_predict);

    for (size_t p = 0; p < prompts.size(); p++) {
        llama_memory_clear(mem_ref, true);
        llama_memory_clear(mem_q,   true);

        std::vector<llama_token> & pt = toks[p];

        llama_batch b_ref = llama_batch_get_one(pt.data(), pt.size());
        llama_batch b_q   = llama_batch_get_one(pt.data(), pt.size());
        if (llama_decode(ctx_ref, b_ref) || llama_decode(ctx_q, b_q)) {
            fprintf(stderr, "error: prefill decode failed (prompt %zu)\n", p);
            return 1;
        }

        int fd = n_predict;  // survived all H steps unless we break
        llama_token tr_prev = 0, tq_prev = 0;
        for (int h = 0; h < n_predict; h++) {
            llama_token tr = greedy_argmax(ctx_ref, n_vocab);
            llama_token tq = greedy_argmax(ctx_q,   n_vocab);
            if (tr != tq) { fd = h; break; }
            if (llama_vocab_is_eog(vocab, tr)) { fd = n_predict; break; } // agreed to EOG: counts as survived
            tr_prev = tr; tq_prev = tq;
            llama_batch nb_ref = llama_batch_get_one(&tr_prev, 1);
            llama_batch nb_q   = llama_batch_get_one(&tq_prev, 1);
            if (llama_decode(ctx_ref, nb_ref) || llama_decode(ctx_q, nb_q)) {
                fprintf(stderr, "error: step decode failed (prompt %zu, h %d)\n", p, h);
                return 1;
            }
        }
        first_div[p] = fd;
        printf("TRAJ %zu %d %d %d\n", p, (int) pt.size(), fd, fd < n_predict ? 1 : 0);
        fflush(stdout);
    }

    // ---- summary ----
    const int N = (int) prompts.size();
    std::vector<int> fd = first_div;
    std::sort(fd.begin(), fd.end());

    auto survival_at = [&](int h) -> double {
        int c = 0;
        for (int v : first_div) if (v > h) c++;
        return (double) c / N;
    };

    int n_div = 0; long sum = 0;
    for (int v : first_div) { if (v < n_predict) n_div++; sum += v; }

    // CVaR5: mean of the worst (earliest-diverging) 5% of prompts
    int k = std::max(1, N / 20);
    long cvar_sum = 0;
    for (int i = 0; i < k; i++) cvar_sum += fd[i];
    const double cvar5 = (double) cvar_sum / k;
    const double median = fd[N / 2];

    printf("\n# SUMMARY quant=%s/%s prompts=%d H=%d\n", ctk.c_str(), ctv.c_str(), N, n_predict);
    printf("SUMMARY diverged_frac=%.4f survive_H=%.4f survive_32=%.4f survive_64=%.4f "
           "median_firstdiv=%.1f mean_firstdiv=%.2f cvar5_firstdiv=%.2f\n",
           (double) n_div / N, survival_at(n_predict - 1),
           survival_at(31), survival_at(63),
           median, (double) sum / N, cvar5);

    llama_free(ctx_q);
    llama_free(ctx_ref);
    llama_model_free(model);
    return 0;
}
