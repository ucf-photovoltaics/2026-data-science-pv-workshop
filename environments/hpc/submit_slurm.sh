#!/usr/bin/env bash
# SLURM batch script for rdstemplate pipeline.
# v0.1.0: STUB — fill in all FIXME lines before submitting.
#
# Submit with:
#   sbatch environments/hpc/submit_slurm.sh
#
# Monitor with:
#   squeue -u $USER
#   tail -f rdstemplate-%j.out

# ── Job metadata ──────────────────────────────────────────────────────────────
#SBATCH --job-name=rdstemplate
#SBATCH --output=rdstemplate-%j.out
#SBATCH --error=rdstemplate-%j.err

# ── Resources ─────────────────────────────────────────────────────────────────
#SBATCH --partition=FIXME-site-specific    # e.g. "general" (UCF ARCC) or "shared" (Anvil)
#SBATCH --account=FIXME-site-specific     # e.g. your ACCESS allocation ID
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8                 # adjust to your job size
#SBATCH --mem=32G                         # adjust to your data size
#SBATCH --time=04:00:00                   # wall-time limit HH:MM:SS

# ── Environment ───────────────────────────────────────────────────────────────
# Option A: Conda environment (fill in env name and module if needed)
# module load FIXME-site-specific/conda   # e.g. "anaconda3/2023.09" on UCF ARCC
# conda activate rdstemplate

# Option B: Apptainer container (uncomment and fill in SIF path)
# SIF=/scratch/$USER/rdstemplate.sif
# RUNNER="apptainer exec $SIF"

RUNNER=""   # leave empty if using Conda; set to "apptainer exec $SIF" for containers

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO=/scratch/$USER/research-ds-template    # FIXME: path to your cloned repo
CONFIG=$REPO/configs/example.yaml           # FIXME: path to your experiment config
OUTDIR=/scratch/$USER/outputs/$SLURM_JOB_ID

mkdir -p "$OUTDIR"
cd "$REPO" || exit 1

# ── Run ───────────────────────────────────────────────────────────────────────
echo "Job $SLURM_JOB_ID started at $(date)"
echo "Config: $CONFIG"
echo "Output: $OUTDIR"

$RUNNER python -m rdstemplate run \
    --config "$CONFIG" \
    --out "$OUTDIR"

EXIT_CODE=$?
echo "Job finished at $(date) with exit code $EXIT_CODE"
exit $EXIT_CODE
