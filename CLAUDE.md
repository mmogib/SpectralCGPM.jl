# TwoGenDFM — Julia Implementation

## Overview
Julia implementation of five derivative-free projection methods (SOPP, SDLP, MOPCGM, CGPM, STTDFPM) for solving large-scale nonlinear monotone equations with convex constraints. Style B (Flat Include).

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
├── SOPPMethod   # Our method 1 (stall_limit kwarg)
├── SDLPMethod     # Our method 2
├── MOPCGMMethod    # Sabi'u et al. 2023
├── CGPMMethod      # Zheng et al. 2020
└── STTDFPMMethod   # Ibrahim et al. 2023
```
Each has `solve(m::Method, prob, x0; eps, maxiter, cb, kwargs...)`.

## Safeguards in solvers (rebuilt 2026-08-06, exchange 05 + review 06)
- Honest termination: converged=true ONLY on a residual test (three sites per
  solver: initial/trial/projected); all other exits are distinct failure
  statuses in SolverResult.status
- Floating-point restart (SOPP/SDLP): failed exact-positivity or
  finiteness check restarts at -lambda*G (counted in SolverResult.restarts);
  NO denominator perturbation hacks (1e-30 guards remain in comparators only)
- Post-projection residual guard before forming any new direction component
- x0 projected onto Gamma on entry (proposed methods)
- Stall cap (SOPP): OPT-IN only, `stall_limit=typemax(Int)` by default
  (the analyzed algorithm does not require monotone residuals)
- Adaptive gamma_k: SOPP in [1.0,1.8] (start 1.1; x1.1 on improvement,
  hold otherwise); SDLP in [1.0,1.95] (start 1.8; x1.1 / x1.05); lambda_k
  updated EVERY iteration (old conditional thresholds removed)

## Key design decisions
- **No LazySets**: projections are simple (max.(x, 0), max.(x, 1), clamp.(x, -C, C))
- **Flat include (Style B)**: no module overhead, easy benchmarking
- **ProgressCallback**: created by scripts, passed into solve(), updates a single external progress bar with live iteration info
- **All outputs in results/**: never write to paper/ from scripts
- **logreg.jl**: precomputes bA = b.*A, preallocates work vectors, numerically stable sigmoid

## Rules (changed 2026-08-06, /join-revision adaptation)
- **Codex implements and runs all jcode changes**, per channel-message specs
  written by Claude (protocol: ../channels/README.md). Claude writes no code;
  Claude specifies (math-first) and reviews Codex's work.
- **Plots.jl exception**: Codex must NOT run scripts using Plots (terminal
  crash). Codex gives Mohammed the exact commands (cwd jcode/); Mohammed runs;
  Codex verifies the figures and reports with copy instructions.
- All outputs to results/ — never write to paper/ from scripts.

## Naming
Current method names throughout: SOPP, SDLP (+ competitors MOPCGM, CGPM,
STTDFPM). For the naming history / archive mapping, see the root CLAUDE.md —
it is deliberately recorded there and nowhere in jcode/.

## Status (updated 2026-08-06, post exchanges 08+09)
- Methods rebuilt (05/06), problems aligned with paper Table 2 (08: G8 −1
  restored; G2/G18 all-NaN domain sentinels — NaN trial = rejected step),
  renamed SOPP/SDLP everywhere (09; zero remnants outside results/).
  Tests 81/81; smoke: SOPP 18/18, SDLP 18/18, MOPCGM 18/18, CGPM 18/18,
  STTDFPM 17/18 (known honest P19 failure). s45 CSV now carries
  status/restarts columns; resume only valid against new-schema raw files.
- OLD RESULTS INVALID: all benchmarks/CS/logreg must be rerun (formulas,
  success criterion, fe accounting, names all changed).
- Rerun-phase TODOs: run s45 full benchmark (Mohammed schedules), then
  CS/logreg; sensitivity experiment; MSE metric check in CS scripts; logreg
  interior-root check; public repository prep.
