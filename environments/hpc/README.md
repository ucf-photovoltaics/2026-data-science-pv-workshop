# HPC Environment — v0.1.0 Stub

> **Status: documented stub.**
> The files in this directory are starting points, not a working HPC integration.
> Partition names, account strings, module names, and filesystem paths are
> **site-specific** and must be filled in against your cluster before use.
>
> Target clusters for this group: UCF ARCC and Purdue Anvil (NSF ACCESS).
> Both use SLURM but have different partition names, modules, and scratch paths.

---

## Intended approach

The `rdstemplate` package is **environment-agnostic** — it has no Colab-specific or
HPC-specific code. The only difference between a Colab run and an HPC run is the thin
shell around the same CLI command:

```
Colab cell:     !python -m rdstemplate run --config configs/example.yaml --out outputs/
SLURM script:   python -m rdstemplate run --config configs/example.yaml --out $SCRATCH/outputs/
```

### Data

Point `data_source.path` in your YAML config at your cluster's scratch or parallel
filesystem. For large datasets, pre-stage data to scratch before the job starts rather
than downloading at runtime — compute nodes often have **no internet access**.

```yaml
data_source:
  type: local
  path: /scratch/$USER/myexperiment/data
```

### Environment provisioning

Two options (pick one):

**Option A — Conda/mamba** (`environment.yml`):
```bash
mamba env create -f environments/hpc/environment.yml
conda activate rdstemplate
pip install -e .
```

**Option B — Apptainer/Singularity** (`container.def`):
```bash
apptainer build rdstemplate.sif environments/hpc/container.def
# Then in submit_slurm.sh, prepend: apptainer exec rdstemplate.sif
```

Dependencies must be provisioned **before** job submission, not inside the batch
script — most compute nodes run without internet.

### Submitting a job

```bash
sbatch environments/hpc/submit_slurm.sh
```

Check status:
```bash
squeue -u $USER
```

### Parallelism

Feature extraction is embarrassingly parallel — one pure function call per
`(sample_id, exposure_step)` with no shared mutable state. To scale to many samples:

1. **Single node, multi-core:** wrap `pipe.extract_features()` with `joblib.Parallel`
   (tier-c customisation — fork the repo and edit `pipeline.py`).
2. **Multi-node:** split the sample list across array jobs, run `extract` per subset,
   then merge the resulting Parquet files. The `extract` subcommand writes a tidy
   dataframe fragment; a final `run` step can load pre-extracted features.

This is left as a future enhancement; v0.1.0 runs single-process.
