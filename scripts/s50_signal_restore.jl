# ============================================================================
# s50: Compressed Sensing — Deep Study
# ============================================================================
#
# Sweeps over sparsity ratio, measurement ratio, and relative noise ratio.
# For each configuration, runs all 5 methods and reports IT, FE, CPU, MSE.
#
# Usage:
#   cd jcode
#   julia --project=. scripts/s50_signal_restore.jl
#   julia --project=. scripts/s50_signal_restore.jl --quick     # small sweep
#   julia --project=. scripts/s50_signal_restore.jl --resume
# ============================================================================

include(joinpath(@__DIR__, "..", "src", "includes.jl"))
using CSV, DataFrames

# ── Configuration ────────────────────────────────────────────────────────────
N_SIGNAL   = 2^12              # signal length n = 4096 (overridden by --quick)
const EPS_CS     = 1e-5
const MAXITER_CS = 5000
const N_TRIALS   = 5           # independent trials per configuration
const RNG_SEED   = 42

# Sweep parameters
const SPARSITY_RATIOS  = [0.05, 0.10, 0.20, 0.30]   # k/n
const MEASUREMENT_RATIOS = [0.25, 0.50, 0.75]        # m/n
const NOISE_RATIOS     = [0.0, 0.001, 0.01, 0.1]

# Quick mode: smaller sweep for testing
quick_mode = "--quick" in ARGS
if quick_mode
    N_SIGNAL = 2^9             # 512 instead of 4096
    SPARSITY_RATIOS_RUN  = [0.10, 0.20]
    MEASUREMENT_RATIOS_RUN = [0.50]
    NOISE_RATIOS_RUN     = [0.0, 0.01]
    N_TRIALS_RUN = 1
else
    SPARSITY_RATIOS_RUN  = SPARSITY_RATIOS
    MEASUREMENT_RATIOS_RUN = MEASUREMENT_RATIOS
    NOISE_RATIOS_RUN     = NOISE_RATIOS
    N_TRIALS_RUN = N_TRIALS
end

do_resume = "--resume" in ARGS

# ── Methods ──────────────────────────────────────────────────────────────────
# Same default parameters as benchmark — oldcode used defaults for CS too
cs_methods = [
    ("SOPP",    SOPPMethod()),
    ("SDLP",    SDLPMethod()),
    ("MOPCGM",  MOPCGMMethod()),
    ("CGPM",    CGPMMethod()),
    ("STTDFPM", STTDFPMMethod()),
]

# ── Output setup ─────────────────────────────────────────────────────────────
results_dir = joinpath(JCODE_ROOT, "results", "signal_restore")
mkpath(results_dir)
raw_csv = joinpath(results_dir, "cs_sweep.csv")

logpath, tee, logfile = setup_logging("signal_restore")

println(tee, "=" ^ 75)
println(tee, "Compressed Sensing: Parameter Sweep")
@tprintf(tee, "  n=%d, trials=%d, quick=%s\n", N_SIGNAL, N_TRIALS_RUN, quick_mode)
@tprintf(tee, "  sparsity:    %s\n", join(SPARSITY_RATIOS_RUN, ", "))
@tprintf(tee, "  measurement: %s\n", join(MEASUREMENT_RATIOS_RUN, ", "))
@tprintf(tee, "  noise ratio: %s\n", join(NOISE_RATIOS_RUN, ", "))
println(tee, "=" ^ 75)

# ── CS problem builder (matching oldcode/functions.jl createCSData exactly) ───
# Key differences from naive setup:
#   1. Sensing matrix is QR-orthogonalized
#   2. Signal entries drawn from Normal(0, 0.001)
#   3. Regularization τ is adaptive: 0.01 * norm(x0, Inf)
#   4. c vector uses τ (not fixed η)

function make_cs_problem(rng, n, k, m, noise_ratio)
    # Sparse signal with small entries (Normal(0, 0.001) matching oldcode)
    x_orig = zeros(n)
    support = randperm(rng, n)[1:k]
    x_orig[support] = 0.001 * randn(rng, k)

    # Sensing matrix: random Normal(0,0.001), then QR-orthogonalized
    A_raw = 0.001 * randn(rng, m, n)
    A = Matrix(qr(A_raw').Q)'

    # Observation with noise
    noise = noise_ratio > 0 ? noise_ratio * 0.001 * randn(rng, m) : zeros(m)
    b = A * x_orig + noise

    # Initial point and adaptive regularization (matching oldcode)
    x0_cs = A' * b
    tau = 0.01 * norm(x0_cs, Inf)

    # Complementarity reformulation
    ATA = A' * A
    c = tau * ones(2n) + vcat(-x0_cs, x0_cs)
    z0 = vcat(max.(x0_cs, 0), max.(-x0_cs, 0))

    # G(z) = min{z, Q*z + c} where Q = [ATA -ATA; -ATA ATA]
    function G_cs(z)
        u = z[1:n]
        v = z[n+1:2n]
        Bu = ATA * (u - v)
        Qz = vcat(Bu, -Bu)
        return min.(z, Qz + c)
    end

    proj_nn(z) = max.(z, 0.0)
    prob = TestProblem(0, "CS", G_cs, proj_nn, "cs")

    return prob, z0, x_orig
end

recover_signal(z, n) = z[1:n] - z[n+1:2n]

# ── Resume detection ─────────────────────────────────────────────────────────
header = ["method", "sparsity_ratio", "measurement_ratio", "noise_ratio",
          "trial", "converged", "iterations", "f_evals", "cpu_time", "mse",
          "rel_err"]

completed = Set{Tuple{String,Float64,Float64,Float64,Int}}()
if do_resume && isfile(raw_csv)
    first_line = open(readline, raw_csv)
    first_line == join(header, ",") ||
        error("Resume schema mismatch in $raw_csv; expected $(join(header, ','))")
    for line in eachline(raw_csv)
        startswith(line, "method") && continue
        parts = split(line, ",")
        length(parts) == length(header) || error("Malformed resume row in $raw_csv: $line")
        push!(completed, (parts[1], parse(Float64, parts[2]), parse(Float64, parts[3]),
                          parse(Float64, parts[4]), parse(Int, parts[5])))
    end
    @tprintf(tee, "Resume: %d rows already done\n", length(completed))
end

# ── CSV header ───────────────────────────────────────────────────────────────
raw_io = open(raw_csv, do_resume && isfile(raw_csv) ? "a" : "w")
if !(do_resume && isfile(raw_csv) && filesize(raw_csv) > 0)
    println(raw_io, join(header, ","))
    flush(raw_io)
end

# ── Main sweep ───────────────────────────────────────────────────────────────
n = N_SIGNAL
total_configs = length(SPARSITY_RATIOS_RUN) * length(MEASUREMENT_RATIOS_RUN) *
                length(NOISE_RATIOS_RUN) * N_TRIALS_RUN * length(cs_methods)

prog = Progress(total_configs; barlen=40, showspeed=true, desc="  CS sweep: ")
n_done = Ref(0)
n_conv = Ref(0)
n_fail = Ref(0)

for sr in SPARSITY_RATIOS_RUN
    k = round(Int, sr * n)
    for mr in MEASUREMENT_RATIOS_RUN
        m = round(Int, mr * n)
        for noise_ratio in NOISE_RATIOS_RUN
            for trial in 1:N_TRIALS_RUN
                rng = Random.Xoshiro(RNG_SEED + trial - 1)
                prob, z0, x_orig = make_cs_problem(rng, n, k, m, noise_ratio)

                for (mname, method) in cs_methods
                    key = (mname, Float64(sr), Float64(mr), Float64(noise_ratio), Int(trial))
                    if key in completed
                        n_done[] += 1
                        ProgressMeter.update!(prog, n_done[];
                            showvalues=[(:done, "$(n_done[])/$total_configs"),
                                        (:converged, n_conv[]), (:failed, n_fail[]),
                                        (:current, "SKIP $mname k/n=$sr m/n=$mr σ=$noise_ratio t=$trial")])
                        continue
                    end

                    result = try
                        solve(method, prob, z0; eps=EPS_CS, maxiter=MAXITER_CS,
                              stall_limit=typemax(Int))
                    catch e
                        @printf(tee.log,
                                "  ERROR: %s k/n=%.2f m/n=%.2f noise_ratio=%.4f trial=%d: %s\n",
                                mname, sr, mr, noise_ratio, trial, sprint(showerror, e))
                        SolverResult(false, 0, 0, NaN, 0.0, z0)
                    end

                    n_done[] += 1
                    result.converged ? (n_conv[] += 1) : (n_fail[] += 1)
                    ProgressMeter.update!(prog, n_done[];
                        showvalues=[(:done, "$(n_done[])/$total_configs"),
                                    (:converged, n_conv[]), (:failed, n_fail[]),
                                    (:current, "$mname k/n=$sr m/n=$mr σ=$noise_ratio t=$trial")])

                    x_rec = recover_signal(result.x, n)
                    mse = norm(x_orig - x_rec)^2 / n
                    rel_err = norm(x_orig - x_rec) / norm(x_orig)

                    @printf(raw_io, "%s,%.2f,%.2f,%.4f,%d,%s,%d,%d,%.9f,%.10e,%.10e\n",
                            mname, sr, mr, noise_ratio, trial,
                            result.converged, result.iterations, result.f_evals,
                            result.cpu_time, mse, rel_err)
                    flush(raw_io)

                end
            end
        end
    end
end
finish!(prog)
close(raw_io)

# ── Summary ──────────────────────────────────────────────────────────────────
println(tee, "\n" * "=" ^ 75)
println(tee, "Summary by method (aggregated over all configurations):")
println(tee, "-" ^ 75)

df = CSV.read(raw_csv, DataFrame)
@tprintf(tee, "  %-10s %8s %8s %10s %12s\n", "Method", "med_IT", "med_FE", "med_CPU", "med_MSE")
println(tee, "-" ^ 75)
for (mname, _) in cs_methods
    sub = filter(r -> r.method == mname && r.converged, df)
    if nrow(sub) > 0
        @tprintf(tee, "  %-10s %8.0f %8.0f %10.4f %12.2e\n",
                 mname, median(sub.iterations), median(sub.f_evals),
                 median(sub.cpu_time), median(sub.mse))
    else
        @tprintf(tee, "  %-10s %8s %8s %10s %12s\n", mname, "---", "---", "---", "---")
    end
end

println(tee, "\nResults: $raw_csv")
teardown_logging(tee, logpath)
