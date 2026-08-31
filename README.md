# Reproducing the Accepted Paper's Numerical Results

This repository contains the Julia code for the paper:

> **Derivative-Free Spectral Projection Methods for Large-Scale Monotone Equations**
> Kabenge Hamiss, Mohammed Alshahrani, Mujahid N. Syed

The paper was accepted by *Mathematics* (MDPI) as manuscript
mathematics-4520875 on 31 August 2026.

Follow the steps below in order. Each step tells you what command to run and what output to expect.

---

## Step 0: Install Julia

1. Go to <https://julialang.org/downloads/> and install Julia 1.12. The
   archived environment was generated with Julia 1.12.4.
2. Run the installer. On Windows, check **"Add Julia to PATH"** when prompted.
3. Open a terminal (Command Prompt, PowerShell, or Bash) and verify:
   ```
   julia --version
   ```
   You should see `julia version 1.12.x`.

## Step 1: Download and install dependencies

**Option A — Clone with Git** (recommended if you have Git installed):

```bash
git clone https://github.com/mmogib/SpectralCGPM.jl.git
cd SpectralCGPM.jl
```

**Option B — Direct download** (no Git required):

1. Go to <https://github.com/mmogib/SpectralCGPM.jl>
2. Click the green **Code** button, then **Download ZIP**
3. Extract the ZIP file
4. Open a terminal and navigate into the extracted folder:
   ```bash
   cd SpectralCGPM.jl-master
   ```

Then install all required Julia packages:

```bash
julia --project=. -e "using Pkg; Pkg.instantiate()"
```

This reads `Project.toml` and the archived `Manifest.toml` and downloads the
dependency versions used by this repository. It may take a few minutes the
first time. You only need to do this once.

**All commands below assume your terminal is inside this folder.**

## Archived accepted data

The complete run-level data behind the accepted paper are versioned in the
repository:

| Accepted output | Archived input | Rows |
|---|---|---:|
| Tables 4--6 and the benchmark-data figures | `results/benchmark/raw.csv` | 5,400 |
| Table 7 | `results/sensitivity/raw.csv` | 5,400 |
| Table 8 | `results/ablation/raw.csv` | 1,620 |
| Table 9 | `results/signal_restore/cs_sweep.csv` | 1,200 |
| Table 11 | `results/logreg/logreg_results.csv` | 300 |

The commands below regenerate tables and figures from these accepted data
without rerunning the experiments. Full experiment commands are also provided
for readers who want an independent rerun. A full rerun without `--resume`
replaces the corresponding CSV, so preserve the archived file if you need an
exact copy of the accepted data.

## Published algorithms and ablation switches

The no-argument constructors instantiate the algorithms published in the
paper:

- `SOPPMethod()` uses `tsel=:cond` and `srule=:max`.
- `SDLPMethod()` uses `srule=:max`.

Algorithms 1 and 2 in the paper describe only these defaults. The alternative
settings exist solely to reproduce the parameter-rule ablation in Table 8:

- SOPP: `tsel=:gap` replaces the condition-number Perry parameter by the
  eigenvalue-gap choice; `srule=:bb1`, `:bb2`, and `:abb` select the first
  ratio, second ratio, and alternating rule.
- SDLP: `srule=:r1`, `:r2`, and `:alt` select the first ratio, second ratio,
  and alternating rule.

The SDLP constructor and sensitivity CSV retain the code field name `eta` for
the initial line-search step. This is the parameter denoted by β in the
accepted paper; it is unrelated to `tsel` or `srule`.

## Step 2: Smoke test (verify the setup)

```bash
julia --project=. scripts/s10_smoke_test.jl
```

This runs all 5 methods on all 18 test problems at a small size (n = 100) to confirm everything is working. It takes about a minute. Check the log in `results/logs/` for any errors.

If this step passes with no errors, your setup is correct.

### Full automated test suite

Run all three test files before modifying or redistributing the code:

```bash
julia --project=. test/test_rebuilt_methods.jl
julia --project=. test/test_sensitivity.jl
julia --project=. test/test_ablation.jl
```

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

The logistic regression experiment uses 12 machine-learning datasets from the LIBSVM repository. The pre-converted CSV files are included in `data/libsvm/`. If you need to regenerate them from the raw `.t` files, run:

```bash
julia --project=. scripts/s20_libsvm_to_csv.jl
julia --project=. scripts/s21_verify_libsvm_csv.jl
```

Both steps are fast (under a minute). The CSV files are already provided, so **you can skip this step** unless you want to verify the conversion yourself.

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

Supports `--resume`, `--datasets=a1a.t,colon-cancer`, `--methods=SDLP,SOPP`, and `--summary`.

## Step 7: Parameter sensitivity (5,400 runs)

The one-at-a-time sensitivity study uses 30 configurations of SOPP and SDLP on
all 18 problems and 10 initial points at `n = 10,000`.

```bash
julia --threads=1 --project=. scripts/s60_sensitivity.jl
```

**Output:** `results/sensitivity/raw.csv`

Use `--resume` to continue an interrupted run. To validate the archived CSV
and print the per-family rate and median-function-evaluation bands used in
Table 7 without running any solver, use:

```bash
julia --threads=1 --project=. scripts/s60_sensitivity.jl --summary
```

In that summary, the SDLP family labelled `eta` is the β row of Table 7,
as explained above.

## Step 8: Parameter-choice ablation (1,620 runs)

The parameter-choice ablation compares the condition-number and eigenvalue-gap
Perry parameters for SOPP, as well as four spectral rules for SOPP and SDLP.
It runs nine configurations on all 18 problems and 10 initial points at
`n = 10,000`.

```bash
julia --threads=1 --project=. scripts/s65_ablation.jl
```

**Output:** `results/ablation/raw.csv`

The script writes a timestamped log under `results/logs/`. If a run is
interrupted, resume it without repeating completed instances:

```bash
julia --threads=1 --project=. scripts/s65_ablation.jl --resume
```

To validate the completed CSV and print per-configuration summaries, identity
checks, and failure-status counts without running solvers:

```bash
julia --threads=1 --project=. scripts/s65_ablation.jl --summary
```

The method constructors expose the same choices for focused experiments:

- `SOPPMethod(tsel=:cond, srule=:max)` uses the published defaults. Other
  choices are `tsel=:gap` and `srule=:bb1`, `:bb2`, or `:abb`.
- `SDLPMethod(srule=:max)` uses the published default. Other choices are
  `srule=:r1`, `:r2`, or `:alt`.

For `:abb` and `:alt`, the outer iteration index `k` is passed when the next
search direction is formed. Even `k` selects the first ratio and odd `k`
selects the second.

## Step 9: Generate the accepted-paper tables

```bash
julia --project=. scripts/s75_tables.jl
```

This reads the archived benchmark, compressed-sensing, and logistic-regression
CSVs and writes LaTeX tables under `results/tables/`. The accepted-paper
mapping is:

| Paper table | Generated file | Archived input |
|---|---|---|
| Table 4, aggregate performance | `results/tables/table_A.tex` | `results/benchmark/raw.csv` |
| Table 5, per-problem convergence | `results/tables/table_P.tex` | `results/benchmark/raw.csv` |
| Table 6, pairwise wins/ties/losses | `results/tables/table_C.tex` | `results/benchmark/raw.csv` |
| Table 9, compressed sensing | `results/tables/table_E.tex` | `results/signal_restore/cs_sweep.csv` |
| Table 11, logistic regression | `results/tables/table_F.tex` | `results/logreg/logreg_results.csv` |

The script also creates `table_D.tex` (per-dimension performance) and
`tables.tex` (detailed benchmark results). Tables 7 and 8 are printed by the
`--summary` commands in Steps 7 and 8; they are not emitted by
`s75_tables.jl`.

## Step 10: Generate the accepted-paper figures

```bash
julia --project=. scripts/s70_figures.jl
```

For only the figures used in the paper, the exact commands and outputs are:

| Command | Accepted-paper outputs |
|---|---|
| `julia --project=. scripts/s70_figures.jl --profiles` | `perf_iterations.pdf`, `perf_fevals.pdf`, `perf_time.pdf` |
| `julia --project=. scripts/s70_figures.jl --scaling` | `scaling_cpu_time.pdf` |
| `julia --project=. scripts/s70_figures.jl --convergence` | `convergence_P9_n50000.pdf` |
| `julia --project=. scripts/s70_figures.jl --signal` | `reconstructed_signals.pdf`, `cs_residual_convergence.pdf` |

All files are written to `results/figures/`. The performance profiles and
scaling plot read `results/benchmark/raw.csv`. The convergence command solves
the fixed Problem 9 instance at `n = 50,000`, and the signal command solves the
fixed seeded compressed-sensing instance described in the paper.

To generate only a subset:

```bash
julia --project=. scripts/s70_figures.jl --profiles      # performance profiles only
julia --project=. scripts/s70_figures.jl --convergence    # convergence plots only
julia --project=. scripts/s70_figures.jl --scaling        # scaling plot only
julia --project=. scripts/s70_figures.jl --signal         # CS figures only
```

Running `s70_figures.jl` with no flag also creates four exploratory
compressed-sensing sweep plots that are not used in the accepted paper.

---

## Directory structure

```
SpectralCGPM.jl/
├── Project.toml              # Dependencies
├── Manifest.toml             # Archived dependency versions
├── README.md                 # This file
├── data/
│   └── libsvm/               # 12 LIBSVM datasets (.csv)
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
│   ├── s60_sensitivity.jl    # Step 7: one-at-a-time sensitivity
│   ├── s65_ablation.jl       # Step 8: parameter-choice ablation
│   ├── s70_figures.jl        # Step 10: generate figures
│   └── s75_tables.jl         # Step 9: generate tables
├── test/                     # Full automated test suite
└── results/                  # Accepted CSVs plus generated outputs
    ├── benchmark/            # accepted raw.csv
    ├── sensitivity/          # accepted sensitivity raw.csv
    ├── ablation/             # accepted ablation raw.csv
    ├── signal_restore/       # cs_sweep.csv
    ├── logreg/               # logreg_results.csv
    ├── figures/              # generated PDFs (ignored by Git)
    ├── tables/               # generated LaTeX (ignored by Git)
    └── logs/                 # timestamped logs (ignored by Git)
```

## Experiment setup

- **Methods:** SOPP, SDLP (proposed) and MOPCGM, CGPM, STTDFPM (competitors)
- **Benchmark:** 18 problems, 6 dimensions (1K--120K), 10 initial points = 5,400 runs
- **Compressed sensing:** 4 sparsity ratios, 3 measurement ratios, 4 noise levels, 5 trials = 1,200 runs
- **Logistic regression:** 12 LIBSVM datasets, 5 trials = 300 runs
- **Sensitivity:** 30 configurations, 18 problems, 10 initial points = 5,400 runs
- **Parameter-choice ablation:** 9 configurations, 18 problems, 10 initial points = 1,620 runs
- **Convergence tolerance:** 10^{-11} (benchmark, sensitivity, ablation, and logistic regression), 10^{-5} (compressed sensing)
- **Maximum iterations:** 2,000 (benchmark, sensitivity, and ablation), 5,000 (compressed sensing and logistic regression)
- **Constraint set:** R^n_+ for all problems except Problem 18 which uses [1, infinity)^n

Competitor methods use their originally published parameters. Our methods use the parameters reported in the paper. See `src/types.jl` for exact values.

## Troubleshooting

- **"Package X not found"**: Re-run `julia --project=. -e "using Pkg; Pkg.instantiate()"` from inside the project folder.
- **Run interrupted**: Use `--resume` to continue from where it stopped (supported by the benchmark, compressed sensing, logistic regression, sensitivity, and ablation scripts).
- **Plots fail to save**: Make sure the `results/figures/` directory exists. Create it manually if needed.
- **Out of memory on large dimensions**: Close other programs. The largest runs (n = 120,000) need several GB of RAM.
