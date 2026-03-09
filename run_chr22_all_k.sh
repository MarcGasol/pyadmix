#!/usr/bin/env bash
# Run TritonAdmix on chr22 for K=2..7 with profiling, then plot each Q matrix with population labels.
# Usage: from repo root, run: ./run_chr22_all_k.sh
# Log: output/run_chr22_all_k.log (also shown on terminal via tee)

set -e

VCF="data/1000G_chr22_pruned.vcf.gz"
OUTDIR="output"
LABELS="data/igsr_samples.tsv"
LOG="$OUTDIR/run_chr22_all_k.log"

mkdir -p "$OUTDIR"
exec > >(tee "$LOG") 2>&1
echo "Started at $(date). Log: $LOG"

for k in 2 3 4 5 6 7; do
  echo "========== K=$k =========="
  tritonadmix run --vcf "$VCF" -k "$k" -o "$OUTDIR/" --profile
  echo "Plotting K=$k ..."
  tritonadmix plot -q "$OUTDIR/1000G_chr22_pruned.${k}.Q" \
    --vcf "$VCF" \
    --labels "$LABELS"
done

echo "Done at $(date). All runs and plots for K=2..7 completed."
