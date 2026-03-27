# Reproducing the Numerical Experiments

This directory contains the Julia code for the paper:

> **Spectral conjugate gradient projection methods for large-scale monotone equations without Lipschitz continuity**
> Kabenge Hamiss, Mohammed Alshahrani, Mujahid N. Syed

Follow the steps below in order. Each step tells you what command to run and what output to expect.

---

## Step 0: Install Julia

1. Go to <https://julialang.org/downloads/> and download the **Current stable release** (1.11 or newer).
2. Run the installer. On Windows, check **"Add Julia to PATH"** when prompted.
3. Open a terminal (Command Prompt, PowerShell, or Bash) and verify:
   ```
   julia --version
   ```
   You should see something like `julia version 1.11.x`. Any version from 1.10 onward will work.

## Step 1: Install dependencies

Open a terminal and navigate to the `jcode/` folder:

```bash
cd jcode
```

Then install all required packages:

```bash
julia --project=. -e "using Pkg; Pkg.instantiate()"
```

This reads `Project.toml` and downloads everything Julia needs (CSV, DataFrames, Plots, etc.). It may take a few minutes the first time. You only need to do this once.

## Step 2: Smoke test (verify the setup)

```bash
julia --project=. scripts/s10_smoke_test.jl
```

This runs all 5 methods on all 18 test problems at a small size (n = 100) to confirm everything is working. It takes about a minute. Check the log in `results/logs/` for any errors.

If this step passes with no errors, your setup is correct.

## Step 3: Full benchmark (5,400 runs)

```bash
julia --project=. scripts/s45_benchmark.jl --all
```

This runs the main experiment: 18 problems, 6 dimensions (1K to 120K), 10 initial points, 5 methods. It produces 5,400 solver runs in total. Expect this to take several hours.

**Output:** `results/benchmark/raw.csv`

If the run is interrupted, you can resume from where it stopped:

```bash
julia --project=. scripts/s45_benchmark.jl --all --resume
```

To print a summary of the results after the run:

```bash
julia --project=. scripts/s45_benchmark.jl --summary
```

## Step 4: Compressed sensing experiment (1,200 runs)

```bash
julia --project=. scripts/s50_signal_restore.jl
```

This sweeps over 4 sparsity ratios, 3 measurement ratios, 4 noise levels, and 5 trials for each of the 5 methods. Signal length is n = 4,096.

**Output:** `results/signal_restore/cs_sweep.csv`

To do a quick test first (n = 512, reduced sweep):

```bash
julia --project=. scripts/s50_signal_restore.jl --quick
```

Supports `--resume` if interrupted.

## Step 5: Prepare the LIBSVM datasets

The logistic regression experiment uses 12 machine-learning datasets from the LIBSVM repository. The raw dataset files (`.t` format) are already included in `data/libsvm/`. Convert them to CSV:

```bash
julia --project=. scripts/s20_libsvm_to_csv.jl
```

Then verify the conversion:

```bash
julia --project=. scripts/s21_verify_libsvm_csv.jl
```

Both steps are fast (under a minute). You only need to do this once.

## Step 6: Logistic regression experiment (300 runs)

```bash
julia --project=. scripts/s55_logreg.jl
```

This runs all 5 methods on all 12 LIBSVM datasets with 5 trials each.

**Output:** `results/logreg/logreg_results.csv`

To do a quick test first (2 datasets, 1 trial):

```bash
julia --project=. scripts/s55_logreg.jl --quick
```

Supports `--resume`, `--datasets=a1a.t,colon-cancer`, `--methods=GCGPM,GMOPCGM`, and `--summary`.

## Step 7: Generate tables

```bash
julia --project=. scripts/s75_tables.jl
```

Produces LaTeX tables in `results/tables/`:

| File | Contents |
|------|----------|
| `table_A.tex` | Aggregate performance summary |
| `table_C.tex` | Pairwise wins/ties/losses |
| `table_D.tex` | Per-dimension breakdown |
| `table_E.tex` | Compressed sensing summary |
| `tables.tex` | Standalone document with detailed per-problem results |

## Step 8: Generate figures

```bash
julia --project=. scripts/s70_figures.jl
```

Produces PDF figures in `results/figures/`:

- **Performance profiles:** iterations, function evaluations, CPU time
- **Convergence trajectories:** representative problems (P5/n=50000, P8/n=10000)
- **Dimension scaling:** CPU time vs problem size
- **Compressed sensing:** reconstructed signals, residual convergence, MSE vs sparsity, MSE vs measurement ratio, iterations vs noise, phase transition heatmap

To generate only a subset:

```bash
julia --project=. scripts/s70_figures.jl --profiles      # performance profiles only
julia --project=. scripts/s70_figures.jl --convergence    # convergence plots only
julia --project=. scripts/s70_figures.jl --scaling        # scaling plot only
julia --project=. scripts/s70_figures.jl --signal         # CS figures only
```

---

## Directory structure

```
jcode/
├── Project.toml              # Dependencies
├── README.md                 # This file
├── data/
│   └── libsvm/               # 12 LIBSVM datasets (.t raw + .csv converted)
├── src/
│   ├── includes.jl           # Entry point (loads everything below)
│   ├── deps.jl               # Shared imports
│   ├── types.jl              # Method types and parameters
│   ├── problems.jl           # 18 test problems, initial points, dimensions
│   ├── projection.jl         # Projection operators
│   ├── solvers.jl            # All 5 solver implementations
│   ├── benchmark.jl          # Multi-solver benchmarking utilities
│   ├── logreg.jl             # Logistic regression problem builder
│   └── utils.jl              # Logging utilities
├── scripts/
│   ├── s10_smoke_test.jl     # Step 2: verify setup
│   ├── s20_libsvm_to_csv.jl  # Step 5: convert LIBSVM to CSV
│   ├── s21_verify_libsvm_csv.jl  # Step 5: verify conversion
│   ├── s45_benchmark.jl      # Step 3: full benchmark
│   ├── s50_signal_restore.jl # Step 4: compressed sensing
│   ├── s55_logreg.jl         # Step 6: logistic regression
│   ├── s70_figures.jl        # Step 8: generate figures
│   └── s75_tables.jl         # Step 7: generate tables
└── results/
    ├── benchmark/            # raw.csv + backups
    ├── signal_restore/       # cs_sweep.csv
    ├── logreg/               # logreg_results.csv
    ├── figures/              # Generated PDFs
    ├── tables/               # Generated LaTeX
    └── logs/                 # Timestamped logs
```

## Experiment setup

- **Methods:** GMOPCGM, GCGPM (proposed) and MOPCGM, CGPM, STTDFPM (competitors)
- **Benchmark:** 18 problems, 6 dimensions (1K--120K), 10 initial points = 5,400 runs
- **Compressed sensing:** 4 sparsity ratios, 3 measurement ratios, 4 noise levels, 5 trials = 1,200 runs
- **Logistic regression:** 12 LIBSVM datasets, 5 trials = 300 runs
- **Convergence tolerance:** 10^{-11} (benchmark), 10^{-5} (CS and logreg)
- **Maximum iterations:** 2,000 (benchmark), 5,000 (CS and logreg)
- **Constraint set:** R^n_+ for all problems except Problem 18 which uses [1, infinity)^n

Competitor methods use their originally published parameters. Our methods use the parameters reported in the paper. See `src/types.jl` for exact values.

## Troubleshooting

- **"Package X not found"**: Re-run `julia --project=. -e "using Pkg; Pkg.instantiate()"` from the `jcode/` folder.
- **Run interrupted**: Use `--resume` to continue from where it stopped (supported by Steps 3, 4, and 6).
- **Plots fail to save**: Make sure the `results/figures/` directory exists. Create it manually if needed.
- **Out of memory on large dimensions**: Close other programs. The largest runs (n = 120,000) need several GB of RAM.
