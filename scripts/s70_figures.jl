# ============================================================================
# s70: Figure Generation
# ============================================================================
#
# Figures:
#   1. Performance profiles (iterations, function evals, CPU time)
#   2. Convergence trajectories (||G|| vs iteration for a representative problem)
#   3. Dimension scaling (CPU time vs n)
#   4. Signal restoration: reconstructed signals
#   5. Signal restoration: MSE vs iteration
#
# Usage:
#   cd jcode
#   julia --project=. scripts/s70_figures.jl
#   julia --project=. scripts/s70_figures.jl --profiles
#   julia --project=. scripts/s70_figures.jl --convergence
#   julia --project=. scripts/s70_figures.jl --scaling
#   julia --project=. scripts/s70_figures.jl --signal
# ============================================================================

include(joinpath(@__DIR__, "..", "src", "includes.jl"))
using CSV, DataFrames
using Plots, LaTeXStrings, BenchmarkProfiles
pgfplotsx()

const RESULTS_DIR = joinpath(JCODE_ROOT, "results")
const IMGS_DIR = joinpath(RESULTS_DIR, "figures")
mkpath(IMGS_DIR)

# ── Consistent styling ────────────────────────────────────────────────────────
const METHOD_ORDER  = ["SOPP", "SDLP", "MOPCGM", "CGPM", "STTDFPM"]
const METHOD_COLORS = [:blue, :red, :green, :darkorange1, :purple]
const METHOD_LSTYLE = [:solid, :solid, :dash, :dash, :dashdot]
const METHOD_LW     = [2.5, 2.5, 2.0, 2.0, 2.0]
const METHOD_MARKER = [:circle, :diamond, :utriangle, :square, :star5]

# Fixed compressed-sensing instance used by both manuscript CS figures.
const CS_FIG_SEED = 42
const CS_FIG_N = 2^12
const CS_FIG_SPARSITY_RATIO = 0.10
const CS_FIG_MEASUREMENT_RATIO = 0.50
const CS_FIG_NOISE_RATIO = 0.01

# Fixed paper-problem identity for the benchmark convergence showcase.
const CONVERGENCE_PAPER_PROBLEM = 9
const CONVERGENCE_INTERNAL_PROBLEM = 11
@assert PROBLEM_IDS[CONVERGENCE_PAPER_PROBLEM] == CONVERGENCE_INTERNAL_PROBLEM

_midx(m) = findfirst(==(m), METHOD_ORDER)

function _load_benchmark_data()
    raw_csv = joinpath(RESULTS_DIR, "benchmark", "raw.csv")
    if !isfile(raw_csv)
        @warn "Benchmark data not found: $raw_csv"
        return nothing
    end

    df = CSV.read(raw_csv, DataFrame)
    all(t -> isfinite(t) && t > 0.0, df.cpu_time) ||
        error("Benchmark cpu_time values must be finite and strictly positive: $raw_csv")
    return df
end

# ═══════════════════════════════════════════════════════════════════════════════
# 1. Performance profiles
# ═══════════════════════════════════════════════════════════════════════════════

function make_performance_profiles()
    df = _load_benchmark_data()
    isnothing(df) && return
    println("  Loaded $(nrow(df)) rows")

    available = intersect(METHOD_ORDER, unique(df.method))
    aidx = [_midx(m) for m in available]
    instances = unique(select(df, :problem, :n, :x0_label))
    n_inst = nrow(instances)

    for (metric, title, filename) in [
            (:iterations, "Iterations",          "perf_iterations.pdf"),
            (:f_evals,    "Function evaluations", "perf_fevals.pdf"),
            (:cpu_time,   "CPU time",             "perf_time.pdf"),
        ]

        T = fill(Inf, n_inst, length(available))
        for (i, inst) in enumerate(eachrow(instances))
            for (j, m) in enumerate(available)
                rows = filter(r -> r.problem == inst.problem &&
                                   r.n == inst.n &&
                                   r.x0_label == inst.x0_label &&
                                   r.method == m, df)
                if nrow(rows) == 1 && rows[1, :converged]
                    v = Float64(rows[1, metric])
                    isfinite(v) && v > 0.0 ||
                        error("Performance-profile metric $metric must be finite and strictly positive")
                    T[i, j] = v
                end
            end
        end

        p = performance_profile(PlotsBackend(), T, available;
                title=title, legend=:bottomright,
                xlabel=L"\tau", ylabel=L"\rho(\tau)",
                logscale=true,
                linewidth=METHOD_LW[aidx],
                linestyle=METHOD_LSTYLE[aidx],
                palette=METHOD_COLORS[aidx],
                legendfontsize=9, minorgrid=true, size=(600, 400))

        savefig(p, joinpath(IMGS_DIR, filename))
        println("  Saved: $filename")
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# 2. Convergence trajectories
# ═══════════════════════════════════════════════════════════════════════════════
# Solve a representative problem and record ||G|| at each iteration.

function make_convergence_plot(; dim=50_000, x0_val=1.0)
    paper_prob_id = CONVERGENCE_PAPER_PROBLEM
    internal_prob_id = PROBLEM_IDS[paper_prob_id]
    prob = get_problem(internal_prob_id, dim)
    x0 = x0_val * ones(dim)

    methods_list = [
        ("SOPP",    SOPPMethod()),
        ("SDLP",    SDLPMethod()),
        ("MOPCGM",  MOPCGMMethod()),
        ("CGPM",    CGPMMethod()),
        ("STTDFPM", STTDFPMMethod()),
    ]

    p = plot(xlabel="Iteration", ylabel=L"\|G(x_k)\|",
             yscale=:log10, legend=:topright,
             size=(600, 400), minorgrid=true,
             title="Convergence on P$(paper_prob_id), n=$dim")

    for (mname, method) in methods_list
        result, ks, residuals =
            _solve_with_history(method, prob, x0; eps=1e-11, maxiter=2000)

        idx = _midx(mname)
        plot!(p, ks, residuals;
              label=mname, color=METHOD_COLORS[idx],
              linestyle=METHOD_LSTYLE[idx], linewidth=METHOD_LW[idx])

        @printf("  %s: %d iterations, final ||G||=%.2e\n",
                mname, result.iterations, residuals[end])
    end

    filename = "convergence_P$(paper_prob_id)_n$(dim).pdf"
    savefig(p, joinpath(IMGS_DIR, filename))
    println("  Saved: $filename")
end

# Run the production solver once and collect its authoritative residual history.
function _solve_with_history(method, prob, x0; eps=1e-11, maxiter=2000)
    ks = Int[]
    residuals = Float64[]
    on_iter = (k, residual) -> begin
        push!(ks, k)
        push!(residuals, Float64(residual))
    end
    result = solve(method, prob, x0; eps=eps, maxiter=maxiter, on_iter=on_iter)
    return result, ks, residuals
end

# ══════════════════════════════════════════════════════════════════════════════
# 3. Dimension scaling
# ═══════════════════════════════════════════════════════════════════════════════
# For a fixed problem, plot median CPU time vs dimension from benchmark data.

function make_scaling_plot()
    df = _load_benchmark_data()
    isnothing(df) && return
    dims = sort(unique(df.n))
    available = intersect(METHOD_ORDER, unique(df.method))

    p = plot(xlabel="Dimension (n)", ylabel="Median CPU time (s)",
             xscale=:log10, yscale=:log10,
             legend=:topleft, size=(600, 400), minorgrid=true,
             title="Scaling behavior (median over all problems and initial points)")

    for m in available
        idx = _midx(m)
        med_times = Float64[]
        valid_dims = Int[]
        for d in dims
            sub = filter(r -> r.method == m && r.n == d && r.converged, df)
            if nrow(sub) > 0
                push!(med_times, median(sub.cpu_time))
                push!(valid_dims, d)
            end
        end
        if !isempty(valid_dims)
            plot!(p, valid_dims, med_times;
                  label=m, color=METHOD_COLORS[idx],
                  linestyle=METHOD_LSTYLE[idx], linewidth=METHOD_LW[idx],
                  marker=METHOD_MARKER[idx], markersize=4)
        end
    end

    filename = "scaling_cpu_time.pdf"
    savefig(p, joinpath(IMGS_DIR, filename))
    println("  Saved: $filename")
end

# ═══════════════════════════════════════════════════════════════════════════════
# 4. Signal restoration: reconstructed signals
# ═══════════════════════════════════════════════════════════════════════════════

function _make_cs_problem(rng)
    n = CS_FIG_N
    k = round(Int, CS_FIG_SPARSITY_RATIO * n)
    m = round(Int, CS_FIG_MEASUREMENT_RATIO * n)
    x_orig = zeros(n)
    support = randperm(rng, n)[1:k]
    x_orig[support] = 0.001 * randn(rng, k)
    A_raw = 0.001 * randn(rng, m, n)
    A = Matrix(qr(A_raw').Q)'
    noise_ratio = CS_FIG_NOISE_RATIO
    noise = noise_ratio * 0.001 * randn(rng, m)
    b = A * x_orig + noise
    x0_cs = A' * b
    tau = 0.01 * norm(x0_cs, Inf)
    ATA = A' * A
    c = tau * ones(2n) + vcat(-x0_cs, x0_cs)
    z0 = vcat(max.(x0_cs, 0), max.(-x0_cs, 0))
    function G_cs(z)
        u = z[1:n]; v = z[n+1:2n]
        Bu = ATA * (u - v)
        return min.(z, vcat(Bu, -Bu) + c)
    end
    proj_nn(z) = max.(z, 0.0)
    prob = TestProblem(0, "CS", G_cs, proj_nn, "cs")
    return prob, z0, x_orig, n
end

function make_signal_reconstruction_plot()
    rng = Random.Xoshiro(CS_FIG_SEED)
    prob, z0, x_orig, n = _make_cs_problem(rng)

    methods_cs = [
        ("SOPP",    SOPPMethod()),
        ("SDLP",    SDLPMethod()),
        ("MOPCGM",  MOPCGMMethod()),
        ("CGPM",    CGPMMethod()),
        ("STTDFPM", STTDFPMMethod()),
    ]

    p = plot(layout=(3, 2), size=(900, 800), margin=3Plots.mm)
    plot!(p[1], x_orig, label="Original", color=:black, lw=0.5,
          title="Original signal", xlabel="", ylabel="Amplitude")

    for (idx, (mname, method)) in enumerate(methods_cs)
        midx = _midx(mname)
        result = solve(method, prob, z0; eps=1e-5, maxiter=5000)
        x_rec = result.x[1:n] - result.x[n+1:2n]
        mse = norm(x_orig - x_rec)^2 / n
        plot!(p[idx+1], x_rec, label="", color=METHOD_COLORS[midx], lw=0.5,
              title=@sprintf("%s (MSE=%.2e)", mname, mse), xlabel="", ylabel="Amplitude")
        @printf("  %s: IT=%d FE=%d MSE=%.2e time=%.2fs\n",
                mname, result.iterations, result.f_evals, mse, result.cpu_time)
    end

    savefig(p, joinpath(IMGS_DIR, "reconstructed_signals.pdf"))
    println("  Saved: reconstructed_signals.pdf")
end

# ═══════════════════════════════════════════════════════════════════════════════
# 5. Signal restoration: MSE vs iteration
# ═══════════════════════════════════════════════════════════════════════════════

function make_mse_convergence_plot()
    rng = Random.Xoshiro(CS_FIG_SEED)
    prob, z0, x_orig, n = _make_cs_problem(rng)

    methods_cs = [
        ("SOPP",    SOPPMethod()),
        ("SDLP",    SDLPMethod()),
        ("MOPCGM",  MOPCGMMethod()),
        ("CGPM",    CGPMMethod()),
        ("STTDFPM", STTDFPMMethod()),
    ]

    p = plot(xlabel="Iteration", ylabel=L"\|G(z_k)\|",
             yscale=:log10, legend=:topright,
             size=(600, 400), minorgrid=true,
             title="Signal recovery: residual convergence")

    for (mname, method) in methods_cs
        midx = _midx(mname)
        result, ks, residuals =
            _solve_with_history(method, prob, z0; eps=1e-5, maxiter=5000)
        plot!(p, ks, residuals;
              label=mname, color=METHOD_COLORS[midx],
              linestyle=METHOD_LSTYLE[midx], linewidth=METHOD_LW[midx])
        @printf("  %s: %d iterations, final ||G||=%.2e\n",
                mname, result.iterations, residuals[end])
    end

    savefig(p, joinpath(IMGS_DIR, "cs_residual_convergence.pdf"))
    println("  Saved: cs_residual_convergence.pdf")
end

# ═══════════════════════════════════════════════════════════════════════════════
# 6. CS sweep: MSE vs sparsity ratio (grouped by method, one curve per noise level)
# ═══════════════════════════════════════════════════════════════════════════════

function make_cs_sweep_plots()
    cs_csv = joinpath(RESULTS_DIR, "signal_restore", "cs_sweep.csv")
    if !isfile(cs_csv)
        @warn "CS sweep data not found: $cs_csv. Run s50_signal_restore.jl first."
        return
    end

    df = CSV.read(cs_csv, DataFrame)
    conv = filter(r -> r.converged, df)
    available = intersect(METHOD_ORDER, unique(df.method))
    println("  Loaded $(nrow(df)) rows ($(nrow(conv)) converged)")

    # --- Plot A: MSE vs sparsity ratio, fixed m/n=0.5, one subplot per method ---
    mr_fixed = 0.5
    sub = filter(r -> r.measurement_ratio == mr_fixed, conv)
    noise_ratios = sort(unique(sub.noise_ratio))
    srs = sort(unique(sub.sparsity_ratio))

    p = plot(layout=(1, length(available)), size=(250*length(available), 350),
             margin=4Plots.mm, link=:y)

    for (j, m) in enumerate(available)
        midx = _midx(m)
        for noise_ratio in noise_ratios
            ss = filter(r -> r.method == m && r.noise_ratio == noise_ratio, sub)
            med_mse = [let s = filter(r -> r.sparsity_ratio == sr, ss)
                        nrow(s) > 0 ? median(s.mse) : NaN
                       end for sr in srs]
            plot!(p[j], srs, med_mse;
                  label=(j==1 ? "σ=$noise_ratio" : ""),
                  xlabel="k/n", ylabel=(j==1 ? "Median MSE" : ""),
                  yscale=:log10, title=m,
                  marker=:circle, markersize=3, linewidth=1.5)
        end
    end

    savefig(p, joinpath(IMGS_DIR, "cs_mse_vs_sparsity.pdf"))
    println("  Saved: cs_mse_vs_sparsity.pdf")

    # --- Plot B: MSE vs measurement ratio, fixed k/n=0.1, one curve per method ---
    sr_fixed = 0.1
    noise_ratio_fixed = 0.01
    sub2 = filter(r -> r.sparsity_ratio == sr_fixed &&
                       r.noise_ratio == noise_ratio_fixed, conv)
    mrs = sort(unique(sub2.measurement_ratio))

    p2 = plot(xlabel="m/n (measurement ratio)", ylabel="Median MSE",
              yscale=:log10, legend=:topright,
              size=(600, 400), minorgrid=true,
              title="Recovery quality: k/n=$sr_fixed, σ=$noise_ratio_fixed")

    for m in available
        midx = _midx(m)
        ss = filter(r -> r.method == m, sub2)
        med_mse = [let s = filter(r -> r.measurement_ratio == mr, ss)
                    nrow(s) > 0 ? median(s.mse) : NaN
                   end for mr in mrs]
        plot!(p2, mrs, med_mse;
              label=m, color=METHOD_COLORS[midx],
              linestyle=METHOD_LSTYLE[midx], linewidth=METHOD_LW[midx],
              marker=METHOD_MARKER[midx], markersize=4)
    end

    savefig(p2, joinpath(IMGS_DIR, "cs_mse_vs_measurement.pdf"))
    println("  Saved: cs_mse_vs_measurement.pdf")

    # --- Plot C: Iterations vs noise level, fixed k/n=0.1, m/n=0.5 ---
    sub3 = filter(r -> r.sparsity_ratio == sr_fixed && r.measurement_ratio == mr_fixed, conv)
    noise_ratios = sort(unique(sub3.noise_ratio))

    p3 = plot(xlabel="Noise ratio σ", ylabel="Median iterations",
              legend=:topleft, size=(600, 400), minorgrid=true,
              title="Solver effort: k/n=$sr_fixed, m/n=$mr_fixed")

    for m in available
        midx = _midx(m)
        ss = filter(r -> r.method == m, sub3)
        med_it = [let s = filter(r -> r.noise_ratio == ratio, ss)
                   nrow(s) > 0 ? median(s.iterations) : NaN
                  end for ratio in noise_ratios]
        plot!(p3, noise_ratios, med_it;
              label=m, color=METHOD_COLORS[midx],
              linestyle=METHOD_LSTYLE[midx], linewidth=METHOD_LW[midx],
              marker=METHOD_MARKER[midx], markersize=4)
    end

    savefig(p3, joinpath(IMGS_DIR, "cs_iters_vs_noise.pdf"))
    println("  Saved: cs_iters_vs_noise.pdf")

    # --- Plot D: Phase transition — convergence rate (%) vs (sparsity, measurement) ---
    # One heatmap per method showing % of trials converged
    noise_ratio_fixed = 0.01
    sub4 = filter(r -> r.noise_ratio == noise_ratio_fixed, df)
    all_srs = sort(unique(sub4.sparsity_ratio))
    all_mrs = sort(unique(sub4.measurement_ratio))

    p4 = plot(layout=(1, length(available)), size=(250*length(available), 350),
              margin=4Plots.mm)

    for (j, m) in enumerate(available)
        midx = _midx(m)
        rate_matrix = zeros(length(all_srs), length(all_mrs))
        for (si, sr) in enumerate(all_srs)
            for (mi, mr) in enumerate(all_mrs)
                ss = filter(r -> r.method == m && r.sparsity_ratio == sr &&
                                 r.measurement_ratio == mr, sub4)
                rate_matrix[si, mi] = nrow(ss) > 0 ? 100 * count(ss.converged) / nrow(ss) : 0
            end
        end
        heatmap!(p4[j], all_mrs, all_srs, rate_matrix;
                 xlabel="m/n", ylabel=(j==1 ? "k/n" : ""),
                 title=m, clims=(0, 100), color=:viridis,
                 colorbar=(j==length(available)))
    end

    savefig(p4, joinpath(IMGS_DIR, "cs_phase_transition.pdf"))
    println("  Saved: cs_phase_transition.pdf")
end

# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

do_all = isempty(ARGS)

println("Generating figures...")

if do_all || "--profiles" in ARGS
    println("\n1. Performance profiles:")
    make_performance_profiles()
end

if do_all || "--convergence" in ARGS
    println("\n2. Convergence trajectories:")
    make_convergence_plot()
end

if do_all || "--scaling" in ARGS
    println("\n3. Dimension scaling:")
    make_scaling_plot()
end

if do_all || "--signal" in ARGS
    println("\n4. Reconstructed signals:")
    make_signal_reconstruction_plot()
    println("\n5. MSE convergence:")
    make_mse_convergence_plot()
end

if do_all || "--cs-sweep" in ARGS
    println("\n6. CS sweep analysis:")
    make_cs_sweep_plots()
end

println("\nDone. Figures saved to: $IMGS_DIR")
