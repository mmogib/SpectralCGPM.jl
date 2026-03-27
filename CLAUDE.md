# TwoGenDFM — Julia Implementation

## Overview
Julia implementation of five derivative-free projection methods (GMOPCGM, GCGPM, MOPCGM, CGPM, STTDFPM) for solving large-scale nonlinear monotone equations with convex constraints. Style B (Flat Include).

## Structure
```
jcode/
├── Project.toml           # Dependencies
├── CLAUDE.md              # This file
├── README.md              # Reproducibility guide
├── data/
│   └── libsvm/            # 12 LIBSVM datasets (.csv, converted by s20)
├── src/
│   ├── includes.jl        # Entry point (include order matters)
│   ├── deps.jl            # Shared dependencies
│   ├── types.jl           # Method types, SolverResult, ProgressCallback
│   ├── problems.jl        # 18 test problems, 10 initial points, 6 dimensions
│   ├── projection.jl      # Projection operators (R^n_+, [1,∞)^n), spectral_proj
│   ├── solvers.jl         # All 5 solvers with safeguards (NaN, LS bail, stall)
│   ├── benchmark.jl       # Multi-solver benchmarking utilities
│   ├── utils.jl           # TeeIO logging, @tprintf macro
│   └── logreg.jl          # Logistic regression: LIBSVM CSV loader, G(x) builder
├── scripts/
│   ├── s10_smoke_test.jl   # Verify all solvers on all problems
│   ├── s20_libsvm_to_csv.jl  # Convert LIBSVM format to CSV
│   ├── s21_verify_libsvm_csv.jl  # Verify CSV conversions
│   ├── s45_benchmark.jl    # Full benchmark (--all, --resume, --summary, --methods)
│   ├── s50_signal_restore.jl  # CS parameter sweep (--quick, --resume)
│   ├── s55_logreg.jl       # Logistic regression (--quick, --resume, --summary, --datasets, --methods)
│   ├── s70_figures.jl      # 11 figure types (--profiles, --convergence, etc.)
│   └── s75_tables.jl       # 5 table types (A, C, D, E + detailed)
└── results/
    ├── benchmark/          # raw.csv + backup/
    ├── signal_restore/     # cs_sweep.csv
    ├── logreg/             # logreg_results.csv
    ├── figures/            # Generated PDFs
    ├── tables/             # Generated LaTeX
    └── logs/               # Timestamped logs
```

## Method Dispatch
```
AbstractMethod
├── GMOPCGMMethod   # Our method 1 (stall_limit kwarg)
├── GCGPMMethod     # Our method 2
├── MOPCGMMethod    # Sabi'u et al. 2023
├── CGPMMethod      # Zheng et al. 2020
└── STTDFPMMethod   # Ibrahim et al. 2023
```
Each has `solve(m::Method, prob, x0; eps, maxiter, cb, kwargs...)`.

## Safeguards in solvers
- NaN/Inf guard: terminates immediately if residual or direction blows up
- Line search failure: returns converged=false if no valid step found
- Stall detection (GMOPCGM only): stops after `stall_limit` consecutive non-improving iterations (default 10, set to typemax(Int) for CS and logreg)
- GCGPM gamma capped at 1.95 on non-improving branch (prevents exceeding 2.0)

## Key design decisions
- **No LazySets**: projections are simple (max.(x, 0), max.(x, 1), clamp.(x, -C, C))
- **Flat include (Style B)**: no module overhead, easy benchmarking
- **ProgressCallback**: created by scripts, passed into solve(), updates a single external progress bar with live iteration info
- **All outputs in results/**: never write to paper/ from scripts
- **logreg.jl**: precomputes bA = b.*A, preallocates work vectors, numerically stable sigmoid

## Rules
- **DO NOT run Julia scripts.** Mohammed runs scripts locally. Only create/edit scripts.
- Tests may be run to verify code changes.

## Status (updated 2026-03-19)
- All experiments complete: benchmark 5,400 + CS 1,200 + logreg 300
- All tables and figures generated
- GCGPM gamma fix applied, re-run complete
- Logistic regression: 12 LIBSVM datasets, 5 trials, all 5 methods
