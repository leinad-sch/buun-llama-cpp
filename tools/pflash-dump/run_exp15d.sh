#!/bin/bash
# EXP-15d content axis — does the model's importance signal carry KLD beyond position?
# Matched-budget price sweep: every config = identical tier histogram (15 t8 / 20 t4 /
# 29 t3 per window, 0.597 B/val), only PERMUTED by priority = z_pos + weight*z_content.
#   anchor : default=fp16, NO mask                 -> must be ~0.000000 (known answer)
#   random : same histogram, random placement      -> floor
#   w0     : pure positional baseline (weight 0)    -> the bar to beat
#   w0.5..w8 : rising content price                 -> does any beat w0?
# FULL scoring is decisive (content lives in the basin; lastk64 is blind to it).
set -u
B=/root/bench/raggedkv-pflash_87c351d28_20260604205934
BIN=$B/build/bin/llama-perplexity
MODEL=/root/turbo-logits-kld/Qwen3.6-27B-Q6_K-BOXMODEL.gguf
CORPUS=/root/wikitext-2-raw/wiki.test.raw
BASE=/root/turbo-logits-kld/base_q6_f16kv_ctx8192_24ch.kld
CMASKDIR=/root/exp15d_cmasks
RES=$B/exp15d; mkdir -p $RES
TSV=$RES/results.tsv
echo -e "config\tscoring\tbytes_per_val\tmeanKLD\terr" > "$TSV"

run(){  # cfg scoring maskfile bpv
  local cfg=$1 sco=$2 mask=$3 bpv=$4
  local LOG=$RES/${cfg}_${sco}.log
  local SC="" CM=""
  [ "$sco" = "lastk64" ] && SC="TURBO_SCORE_LAST_K=64"
  [ -n "$mask" ] && CM="TURBO_RAGGED_CONTENT_MASK=$CMASKDIR/$mask"
  env $SC $CM TURBO_RAGGED_SCHEDULE="default=fp16" LD_LIBRARY_PATH=$B/build/bin timeout 5400 $BIN \
    -m $MODEL -f $CORPUS --kl-divergence-base $BASE --kl-divergence \
    -ctk f16 -ctv f16 -fa on -c 8192 -ngl 99 > $LOG 2>&1
  local LINE=$(grep -E "Mean +KLD" $LOG | tail -1)
  local K=$(echo "$LINE" | awk '{print $3}'); K=${K:--}
  local E=$(echo "$LINE" | awk '{print $5}'); E=${E:--}
  printf "%s\t%s\t%s\t%s\t%s\n" "$cfg" "$sco" "$bpv" "$K" "$E" | tee -a "$TSV"
}

# Known-answer anchor FIRST (must be ~0); abort if not.
run anchor full "" 16.0000
AK=$(awk -F'\t' '$1=="anchor"{print $4}' "$TSV")
echo "anchor full meanKLD=$AK (expect ~0)"

run random full cmask_random.bin 0.597
for wt in 0 0.5 1 2 4 8; do
  run w$wt full cmask_w$wt.bin 0.597
done
echo "=== EXP-15d content price sweep (full scoring) complete ==="
column -t "$TSV"
