# Implementation spec: `gemma4-dflash-draft` arch support

**Goal:** make this fork load + run the Lucebox `gemma-4-31B-it-DFlash-q8_0.gguf` as a DFlash
speculative-decoding drafter for `gemma-4-31B-it-Q4_K_M.gguf`. Currently fails with
`unknown model architecture: 'gemma4-dflash-draft'`.

**Branch/worktree:** `feature/gemma4-dflash-draft` @ `/home/blice/cuda-llama/claude-research-hub/fork-gemma4dflash`
(off merged trunk 17ea3baff, which already has both `gemma4` and `dflash-draft` archs).

**Approach:** the gemma4 DFlash draft = our existing `dflash-draft` KV-injection graph +
gemma4's numerical conventions. Both templates are in-tree. NO external reference needed — the draft
was trained to mimic gemma-4-31B, so mirror our own `src/models/gemma4.cpp` conventions.

---

## The target GGUF (corsair `/home/buun/models/gemma-4-31B-it-DFlash-q8_0.gguf`)

- `general.architecture = gemma4-dflash-draft`
- hparams (all keyed `gemma4-dflash-draft.*`): block_count=5, embedding_length=5376,
  feed_forward_length=10752, attention.head_count=64, head_count_kv=8, key_length=128,
  value_length=128, vocab_size=262144, layer_norm_rms_epsilon=1e-6, rope.freq_base=1000000,
  attention.sliding_window=2048, final_logit_softcapping=30.0, tie_word_embeddings=True,
  dflash.n_target_layers=60, dflash.block_size=16, dflash.mask_token_id=4,
  dflash.target_layer_ids=[6 ints]
- tensors per layer `blk.N.`: attn_norm, attn_q, attn_k, attn_v, attn_output, attn_q_norm,
  attn_k_norm, ffn_norm, ffn_gate, ffn_up, ffn_down. Global: **`dflash.fc.weight`**,
  **`dflash.hidden_norm.weight`** (DOT form!), output_norm.weight.
  NO token_embd, NO output (tied → shares target's), NO attn_post_norm, NO ffn_post_norm.

## Templates (read these)
- `src/models/dflash_draft.cpp` — the DFlash draft graph to clone. KV-injection: `dflash_fc` projects
  concatenated target hidden [n_target_features, ctx_len] → `dflash_hidden_norm`; per layer Q from
  block/noise tokens, K/V concatenated from (target-context via fused_target) + (noise), asymmetric
  non-causal mask, RoPE, QK-norm. Handles SWA already.
- `src/models/gemma4.cpp` — gemma4 conventions (line refs approx):
  - embed scale: `inpL=cast_bf16; scale(bf16_round(sqrt(n_embd))); cast_f32` (~184-186)
  - attention scale = `hparams.f_attention_scale = 1.0` (~12), passed to build_attn
  - QK-norm on Q and K (RMS) (~238, 259)
  - FFN activation = **GELU** (LLM_FFN_GELU) (~308/363) — NOT silu
  - final logit softcap: `scale(1/cap); tanh; scale(cap)`, cap=f_final_logit_softcapping=30 (~433-437)

## The 6 grafts (dflash_draft graph -> gemma4-dflash graph)
1. **pre-FFN norm = `ffn_norm`** (the draft has ffn_norm, NOT attn_post_norm). Structure:
   attn_norm → attn → +residual → ffn_norm → ffn → +residual. No post-norms.
2. **embedding scale** sqrt(n_embd) BF16-rounded on the block-token embeds (build_inp_embd result).
3. **attention scale = 1.0** (hparams.f_attention_scale) in build_attn_mha, NOT 1/sqrt(head_dim).
4. **FFN = GELU** (LLM_FFN_GELU, LLM_FFN_PAR), not SILU.
5. **final logit softcapping** (tanh, cap=30) after the output projection.
6. keep QK-norm + RoPE (freq_base=1M flows from hparams automatically).
   ⚠ VERIFY during impl: (a) does gemma4 apply the embed scale to the DRAFT's block tokens the same
   way? (b) exact attention-scale semantics — gemma folds query pre-scaling; confirm build_attn_mha
   with scale=1.0 matches. Draft ACCEPTANCE RATE is the correctness gate; iterate on 2/3 if low.

## Files to change (mirror the `DFLASH_DRAFT` wiring)
1. `src/llama-arch.h`: add `LLM_ARCH_GEMMA4_DFLASH_DRAFT` enum (near LLM_ARCH_DFLASH_DRAFT).
2. `src/llama-arch.cpp`:
   - name map: `{ LLM_ARCH_GEMMA4_DFLASH_DRAFT, "gemma4-dflash-draft" }` (near line 141).
   - LLM_TENSOR_NAMES block for the arch: attn_norm/q/k/v/output, attn_q_norm/k_norm, ffn_norm/
     gate/up/down, output_norm, and **DFLASH_FC → "dflash.fc"**, **DFLASH_HIDDEN_NORM →
     "dflash.hidden_norm"** (DOT — our dflash-draft uses underscore `dflash_fc`, line ~506; this arch
     needs the dot form to match the GGUF). Mirror the DFLASH_DRAFT block otherwise.
3. `src/models/models.h`: declare `struct llm_build_gemma4_dflash_draft : llm_graph_context` +
   `struct llama_model_gemma4_dflash_draft : llama_model_base` (mirror lines 570-575).
4. `src/models/gemma4_dflash_draft.cpp` (NEW): copy dflash_draft.cpp, apply the 6 grafts. In
   load_arch_hparams: reuse dflash keys + read LLM_KV_FINAL_LOGIT_SOFTCAPPING + set
   f_attention_scale=1.0 + sliding_window. ⚠ n_target_features: Lucebox has `dflash.n_target_layers`
   not `.n_target_features`; derive n_target_features from the `dflash.fc` tensor ne[0] (or key), and
   set dflash_n_target_features accordingly so the fc tensor + target_hidden dims match. In
   load_arch_tensors: NO attn_post_norm/ffn_post_norm/tok_embd/output; ADD ffn_norm; dflash_fc dims
   {n_target_features, n_embd}.
5. `src/llama-model.cpp`: add GEMMA4_DFLASH_DRAFT to (a) the create-model switch (~306, `return new
   llama_model_gemma4_dflash_draft(params)`), (b) the load-hparams dispatch (calls load_arch_hparams),
   (c) the graph-build `res=nullptr` list (~2087), (d) the ~2587 list (MIMO2/STEP35/DFLASH_DRAFT/...).
   Grep all `LLM_ARCH_DFLASH_DRAFT` occurrences and add the twin.
6. Drafter registration: find where COMMON_SPECULATIVE_TYPE_DFLASH validates/uses the draft model's
   arch (common/speculative.cpp — the DFlash drafter creation). It must accept GEMMA4_DFLASH_DRAFT as
   a valid dflash draft (it currently keys off dflash-draft somewhere). ⚠ INVESTIGATE: grep
   speculative.cpp for how it identifies a dflash draft model (arch check, or hparams.dflash_block_size>0).

## Build + test (corsair, gfx1151)
- Baseline keeper build (for the target model): `/home/buun/bench/integrate-turbo-sync_17ea3baff_20260709194014/repo/build/bin`
- New build: rsync this worktree -> fresh corsair dir -> `cmake -S . -B build -DGGML_HIP=ON
  -DAMDGPU_TARGETS=gfx1151 -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF` -> build llama-server.
- Test: `llama-server -m /home/buun/models/gemma-4-31B-it-Q4_K_M.gguf -ngl 999 -c 4096
  --spec-dflash-default -md /home/buun/models/gemma-4-31B-it-DFlash-q8_0.gguf --port 8150`
  Then a /completion — check: (1) loads (no unknown-arch), (2) coherent output, (3) draft acceptance
  (server timings draft_n_accepted/draft_n) is HIGH (>40%). Low acceptance = a convention is wrong
  (softcap/embed-scale/attn-scale) → iterate.
- corsair: buun@corsair.lan, ROCm 7.2.1 (PATH=/opt/rocm-7.2.1/bin), VMM default-on.

## Status — ✅ DONE (2026-07-09), validated on corsair (gfx1151)
- [x] branch + worktree + this spec
- [x] arch plumbing (enum, name, dot-form tensor enums + map + infos, class decls)
- [x] gemma4_dflash_draft.cpp graph (all 6 grafts)
- [x] llama-model.cpp dispatch (create/res=nullptr/rope-NEOX); drafter reg is arch-agnostic (block_size)
- [x] build on corsair + acceptance test → PASS

### Two fixes discovered during bring-up (both in-tree, uncommitted on branch)
1. **fc-dims**: `hparams.dflash_n_target_features` DEFAULTS to 25600 (a Qwen value), not 0 — so a
   `== 0` guard never fires. Derive it from the fc tensor's real ne[0] via
   `ml.get_tensor_meta("dflash.fc.weight")->ne[0]` (= 32256 = 6 target layers × 5376).
2. **no tokenizer**: the Lucebox draft GGUF ships NO tokenizer (shares the target's vocab; the Qwen
   dflash-draft embeds a full gpt2 tokenizer, which is why it never hit this). Fix in
   `src/llama-vocab.cpp`: for the two dflash-draft archs, make `tokenizer.ggml.model` optional and
   default missing→"none" → the existing `LLAMA_VOCAB_TYPE_NONE` path (reads vocab_size, dummy tokens).

### Acceptance results (gemma-4-31B-it-Q4_K_M target, temp 0, 200 tok, -c 4096)
- **Code**: coherent, **44.3% draft acceptance** (149/336), 12.2 t/s
- **Prose (chat template)**: coherent reasoning, **28.6% acceptance** (131/458), 8.8 t/s
- Load log confirms derived hparams: `block_size=16, mask_token=4, n_target_layers=6, n_embd=5376,
  target_ids=[1,12,23,35,46,57]`, gpu ring `6 layers x 512 slots x 5376 embd`.
- Grafts validated: near-zero acceptance would signal a wrong attn-scale/softcap/embed-scale; 28-44% ⇒ correct.
- Build dir: `/home/buun/bench/gemma4dflash_17ea3baff_20260709232458/repo` (test server killed after).
- NOTE: raw `/completion` prose greedy-loops (`***`) — a target-side instruct-model artifact on
  un-templated prompts, NOT a draft bug (spec decoding is lossless). Use the chat endpoint.
