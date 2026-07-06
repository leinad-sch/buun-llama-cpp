#pragma once

// TCQ decode alpha_V — the SINGLE source of truth for all three consumers:
//   fattn.cu            tcq_compute_alpha_v      (materialize/prefill path)
//   fattn-mma-turbo.cuh fused launcher constexpr (fused decode path)
//   set-rows.cu         alpha_v_tier             (ragged per-tier defaults)
// A desync between these silently runs prefill and decode at different alphas
// (cost 2h to find, once). Values are KLD-panel tuned:
//   t3/t2: flat optima for the coord-descent codebooks (K374/V500, K195/V208).
//   t1:    2026-07-05 native-path panel (q27, 3 depths x 2 books x alpha 1.06-1.30):
//          interior minimum at 1.26 in every completed series; the earlier
//          ragged-harness <=1.14 was instrument-not-deployment.
#define TURBO_TCQ_ALPHA_V_T1 1.26f
#define TURBO_TCQ_ALPHA_V_T2 1.06f
#define TURBO_TCQ_ALPHA_V_T3 1.02f
