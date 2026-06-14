#!/bin/bash
# EXP-15e per-head KV axis — does per-head d_eff predict which K heads tolerate
# aggressive quant? Matched-budget PERMUTATION test: every arm spends the identical
# per-global-layer K histogram (2 heads t8 + 2 heads t3), only the head->tier
# placement differs. K only; V + SWA-K stay fp16 (default=16). FULL KLD scoring.
#   anchor       : default=16, no bands           -> must be ~0 (known-answer gate)
#   protect_high : t8 on the 2 highest-d_eff heads -> wins iff d_eff predicts tolerance
#   protect_low  : t8 on the 2 lowest-d_eff heads  -> reverse
#   random       : t8 on 2 random heads            -> middle reference / floor
set -u
B=__BENCHDIR__
BIN=$B/build/bin/llama-perplexity
MODEL=/root/turbo-logits-kld/Qwen3.6-27B-Q6_K-BOXMODEL.gguf
CORPUS=/root/wikitext-2-raw/wiki.test.raw
BASE=/root/turbo-logits-kld/base_q6_f16kv_ctx2048_32ch.kld
SCHEDDIR=$B/sched
RES=$B/exp15e; mkdir -p $RES
TSV=$RES/results.tsv
echo -e "config\tKbudget_note\tmeanKLD\terr" > "$TSV"

run(){  # cfg schedstring note
  local cfg=$1 sched=$2 note=$3
  local LOG=$RES/${cfg}.log
  env TURBO_RAGGED_SCHEDULE="$sched" LD_LIBRARY_PATH=$B/build/bin timeout 5400 $BIN \
    -m $MODEL -f $CORPUS --kl-divergence-base $BASE --kl-divergence \
    -ctk f16 -ctv f16 -fa on -c 2048 -ngl 99 > $LOG 2>&1
  local LINE=$(grep -E "Mean +KLD" $LOG | tail -1)
  local K=$(echo "$LINE" | awk '{print $3}'); K=${K:--}
  local E=$(echo "$LINE" | awk '{print $5}'); E=${E:--}
  printf "%s\t%s\t%s\t%s\n" "$cfg" "$note" "$K" "$E" | tee -a "$TSV"
}

# Known-answer anchor FIRST (must be ~0); abort if not.
run anchor "default=16" "fp16"
AK=$(awk -F'\t' '$1=="anchor"{print $3}' "$TSV")
echo "anchor meanKLD=$AK (expect ~0)"

run protect_high "$(cat $SCHEDDIR/sched_protect_high.txt)" "2t8+2t3-Kglobal"
run protect_low  "$(cat $SCHEDDIR/sched_protect_low.txt)"  "2t8+2t3-Kglobal"
run random       "$(cat $SCHEDDIR/sched_random.txt)"       "2t8+2t3-Kglobal"
echo "=== EXP-15e per-head axis sweep complete ==="
column -t "$TSV"
