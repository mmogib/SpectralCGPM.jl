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
const METHOD_ORDER  = ["GMOPCGM", "GCGPM", "MOPCGM", "CGPM", "STTDFPM"]
const METHOD_COLORS = [:blue, :red, :green, :darkorange1, :purple]
const METHOD_LSTYLE = [:solid, :solid, :dash, :dash, :dashdot]
const METHOD_LW     = [2.5, 2.5, 2.0, 2.0, 2.0]
const METHOD_MARKER = [:circle, :diamond, :utriangle, :square, :star5]

_midx(m) = findfirst(==(m), METHOD_ORDER)

# ═══════════════════════════════════════════════════════════════════════════════
# 1. Performance profiles
# ═══════════════════════════════════════════════════════════════════════════════

function make_performance_profiles()
    raw_csv = joinpath(RESULTS_DIR, "benchmark", "raw.csv")
    if !isfile(raw_csv)
        @warn "Benchmark data not found: $raw_csv"
        return
    end

    df = CSV.read(raw_csv, DataFrame)
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
                    T[i, j] = v > 0 ? v : 1e-15
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

function make_convergence_plot(; prob_id=5, dim=50_000, x0_val=1.0)
    prob = get_problem(prob_id, dim)
    x0 = x0_val * ones(dim)

    methods_list = [
        ("GMOPCGM", GMOPCGMMethod()),
        ("GCGPM",   GCGPMMethod()),
        ("MOPCGM",  MOPCGMMethod()),
        ("CGPM",    CGPMMethod()),
        ("STTDFPM", STTDFPMMethod()),
    ]

    p = plot(xlabel="Iteration", ylabel=L"\|G(x_k)\|",
             yscale=:log10, legend=:topright,
             size=(600, 400), minorgrid=true,
             title="Convergence on $(prob.name), n=$dim")

    for (mname, method) in methods_list
        # Collect residual history by solving with a tracking wrapper
        history = Float64[]
        fe = Ref(0)
        G_tracked(x) = (fe[] += 1; prob.G(x))

        # Run solve and track via a simple re-implementation that logs
        residuals = _collect_residuals(method, prob, x0; eps=1e-11, maxiter=2000)

        idx = _midx(mname)
        plot!(p, 1:length(residuals), residuals;
              label=mname, color=METHOD_COLORS[idx],
              linestyle=METHOD_LSTYLE[idx], linewidth=METHOD_LW[idx])

        @printf("  %s: %d iterations, final ||G||=%.2e\n",
                mname, length(residuals), residuals[end])
    end

    filename = "convergence_P$(prob_id)_n$(dim).pdf"
    savefig(p, joinpath(IMGS_DIR, filename))
    println("  Saved: $filename")
end

# Helper: run solver and collect ||G|| at each iteration
function _collect_residuals(method, prob, x0; eps=1e-11, maxiter=2000)
    residuals = Float64[]
    G(x) = prob.G(x)
    x = copy(x0)
    Gx = G(x)
    push!(residuals, norm(Gx))

    # Simple re-solve tracking residuals via the projection loop
    # We call solve() but we need the history. Easiest: solve and trust
    # the ProgressCallback mechanism to NOT interfere, then just run
    # a lightweight version here.
    result = solve(method, prob, x0; eps=eps, maxiter=maxiter)

    # Since we can't easily extract per-iteration history from solve(),
    # approximate by running again with manual tracking.
    # For now, just record initial and final.
    # TODO: add history support to solve() if needed.

    # Actually, let's do a proper tracking solve for this figure.
    # We duplicate the core loop for GMOPCGM-like methods generically.
    # This is figure-only code, not the benchmark solver.

    residuals = _track_solve(method, prob, x0, eps, maxiter)
    return residuals
end

# Generic tracking solver — runs the same algorithm but records ||G|| each iteration
function _track_solve(m::GMOPCGMMethod, prob, x0, eps, maxiter)
    G(x) = prob.G(x)
    x = copy(x0); Gx = G(x); normGx = norm(Gx)
    hist = [normGx]
    normGx < eps && return hist
    p = -Gx; phi = m.lambda0; gamma_val = m.gamma
    for k in 1:maxiter
        alpha, z, Gz = _backtrack_ours(G, x, p, m.rho, m.beta, m.zeta, m.zeta1, m.zeta2, Ref(0))
        normGz = norm(Gz)
        normGz < eps && (push!(hist, normGz); return hist)
        mu_k = dot(Gz, x .- z) / normGz^2
        x_new = prob.proj(x .- gamma_val .* mu_k .* Gz)
        Gx_new = G(x_new); normGx_new = norm(Gx_new)
        gamma_val = normGx_new < normGx ? min(gamma_val*1.1,1.8) : max(gamma_val*1.0,1.0)
        s = z .- x; y = Gx_new .- Gx; v = y .+ m.tau .* s
        norms2 = dot(s,s); sv = dot(s,v); normv2 = dot(v,v)
        if !(normGx_new < 0.75*normGx) && norms2 > 1e-30 && abs(sv) > 1e-30
            phi = spectral_proj(max(norms2/sv, sv/normv2), m.alpha_min, m.alpha_max)
        end
        if norms2 > 1e-30 && abs(sv) > 1e-30
            ts = phi*sv/norms2; dv = dot(p,v)
            bm = abs(dv) > 1e-30 ? dot(v .- ts.*s, Gx_new)/dv : 0.0
            Gn2 = dot(Gx_new,Gx_new)
            rc = Gn2 > 1e-30 ? dot(Gx_new,p)/Gn2 : 0.0
            p = -(phi+bm*rc).*Gx_new .+ bm.*p
        else; p = -phi.*Gx_new; end
        x = x_new; Gx = Gx_new; normGx = normGx_new
        push!(hist, normGx)
        normGx < eps && return hist
        norm(p)*10 < eps && return hist
    end
    return hist
end

function _track_solve(m::GCGPMMethod, prob, x0, eps, maxiter)
    G(x) = prob.G(x)
    x = copy(x0); Gx = G(x); normGx = norm(Gx)
    hist = [normGx]; normGx < eps && return hist
    d = -Gx; phi = m.lambda0; gv = m.gamma
    for k in 1:maxiter
        alpha,z,Gz = _backtrack_ours(G,x,d,m.rho,m.eta,m.zeta,m.zeta1,m.zeta2,Ref(0))
        normGz = norm(Gz); normGz < eps && (push!(hist,normGz); return hist)
        tk = dot(Gz,x.-z)/normGz^2; x_new = prob.proj(x .- gv.*tk.*Gz)
        Gx_new = G(x_new); nGn = norm(Gx_new)
        gv = nGn < normGx ? min(gv*1.1,1.70) : max(gv*1.05,1.05)
        y = Gx_new.-Gx; Gp=dot(Gx_new,d); yp=dot(y,d)
        rk = 1.0+max(0.0,-Gp/(yp+sign(yp)*1e-30)); w = y.+rk.*d; wd=dot(w,d)
        if abs(wd) > 1e-30
            ck=Gp/wd; ww=dot(w,w); dd=dot(d,d)
            if !(nGn<0.6*normGx) && dd>1e-30; phi=spectral_proj(max(ww/wd,wd/dd),m.alpha_min,m.alpha_max); end
            bk=(dot(Gx_new,w)-phi*ck*ww)/wd; d = -phi.*Gx_new.+bk.*d.+m.tau.*ck.*w
        else; d = -phi.*Gx_new; end
        x=x_new; Gx=Gx_new; normGx=nGn; push!(hist,normGx)
        normGx < eps && return hist; norm(d)*10 < eps && return hist
    end; return hist
end

function _track_solve(m::MOPCGMMethod, prob, x0, eps, maxiter)
    F(x)=prob.G(x); x=copy(x0); Fx=F(x); nF=norm(Fx); hist=[nF]; nF<eps && return hist
    d=-Fx
    for k in 1:maxiter
        dd=dot(d,d); alpha=m.rho
        for j in 0:50; alpha=m.rho*m.eta^j; z=x.+alpha.*d; Fz=F(z)
            (-dot(Fz,d)>=m.zeta*alpha*dd||alpha<=1e-5) && break; end
        z=x.+alpha.*d; Fz=F(z); nFz=norm(Fz); nFz<eps && (push!(hist,nFz); return hist)
        bk=dot(Fz,x.-z)/nFz^2; xn=prob.proj(x.-bk.*Fz); Fn=F(xn)
        s=z.-x; y=Fn.-Fx.+m.lambda.*s; ss=dot(s,s); sy=dot(s,y)
        ts=ss>1e-30 ? sy/ss : 0.0; dy=dot(d,y)
        bm=abs(dy)>1e-30 ? dot(y.-ts.*s,Fn)/dy : 0.0
        Fn2=dot(Fn,Fn); rc=Fn2>1e-30 ? dot(Fn,d)/Fn2 : 0.0
        d=-(1.0+bm*rc).*Fn.+bm.*d; x=xn; Fx=Fn; nF=norm(Fx); push!(hist,nF)
        nF<eps && return hist; norm(d)*10<eps && return hist
    end; return hist
end

function _track_solve(m::CGPMMethod, prob, x0, eps, maxiter)
    F(x)=prob.G(x); x=copy(x0); Fx=F(x); nF=norm(Fx); hist=[nF]; nF<eps && return hist
    d=-Fx
    for k in 1:maxiter
        dn2=dot(d,d); alpha=m.beta_ls
        for j in 0:50; alpha=m.beta_ls*m.rho^j; z=x.+alpha.*d; Fz=F(z)
            (-dot(Fz,d)>=m.sigma*alpha*norm(Fz)*dn2||alpha<=1e-5) && break; end
        z=x.+alpha.*d; Fz=F(z); nFz=norm(Fz); nFz<eps && (push!(hist,nFz); return hist)
        tk=dot(Fz,x.-z)/nFz^2; xn=prob.proj(x.-tk.*Fz); Fn=F(xn)
        y=Fn.-Fx; dd=dot(d,d); lc=1.0+max(0.0,-dot(y,d)/max(dd,1e-30))
        w=y.+lc.*d; wd=dot(w,d)
        if abs(wd)>1e-30; ck=dot(Fn,d)/wd; bk=(dot(Fn,w)-2.0*ck*dot(w,w))/wd
            d=-m.sigma1.*Fn.+bk.*d.+m.sigma2.*ck.*w
        else; d=-m.sigma1.*Fn; end
        x=xn; Fx=Fn; nF=norm(Fx); push!(hist,nF)
        nF<eps && return hist; norm(d)*10<eps && return hist
    end; return hist
end

function _track_solve(m::STTDFPMMethod, prob, x0, eps, maxiter)
    F(x)=prob.G(x); x=copy(x0); Fx=F(x); nF=norm(Fx); hist=[nF]; nF<eps && return hist
    d=-Fx
    for k in 1:maxiter
        norm(d)*10<eps && return hist
        dd=dot(d,d); tau=1.0
        for j in 0:50; tau=m.beta_ls^j; z=x.+tau.*d; Fz=F(z)
            pv=clamp_scalar(norm(Fz),m.eta1,m.eta2)
            (-dot(Fz,d)>=m.sigma*tau*pv*dd||tau<=1e-5) && break; end
        z=x.+tau.*d; Fz=F(z); nFz=norm(Fz); nFz<eps && (push!(hist,nFz); return hist)
        mk=dot(Fz,x.-z)/nFz^2; xn=prob.proj(x.-m.gamma.*mk.*Fz); Fn=F(xn)
        y=Fn.-Fx; ss=(xn.-x).+m.r_param.*y; yy=dot(y,y); sy=dot(ss,y)
        aI=yy>1e-30 ? spectral_proj(sy/yy,m.alpha_min,m.alpha_max) : 1.0
        vk=max(m.psi*norm(d)*norm(y),nF^2)
        if abs(vk)>1e-30; bk=dot(Fn,y)/vk; aII=dot(Fn,d)/vk
        else; bk=0.0; aII=0.0; end
        d=-aI.*Fn.+bk.*d.-aII.*y; x=xn; Fx=Fn; nF=norm(Fx); push!(hist,nF)
        nF<eps && return hist
    end; return hist
end

# ═══════════════════════════════════════════════════════════════════════════════
# 3. Dimension scaling
# ═══════════════════════════════════════════════════════════════════════════════
# For a fixed problem, plot median CPU time vs dimension from benchmark data.

function make_scaling_plot()
    raw_csv = joinpath(RESULTS_DIR, "benchmark", "raw.csv")
    if !isfile(raw_csv)
        @warn "Benchmark data not found: $raw_csv"
        return
    end

    df = CSV.read(raw_csv, DataFrame)
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
    n = 2^12; k = n ÷ 8; m = n ÷ 4
    x_orig = zeros(n)
    support = randperm(rng, n)[1:k]
    x_orig[support] = 0.001 * randn(rng, k)
    A_raw = 0.001 * randn(rng, m, n)
    A = Matrix(qr(A_raw').Q)'
    noise = 0.0001 * 0.001 * randn(rng, m)
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
    rng = Random.Xoshiro(42)
    prob, z0, x_orig, n = _make_cs_problem(rng)

    methods_cs = [
        ("GMOPCGM", GMOPCGMMethod()),
        ("GCGPM",   GCGPMMethod()),
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
        mse = norm(x_orig - x_rec) / n
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
    rng = Random.Xoshiro(42)
    prob, z0, x_orig, n = _make_cs_problem(rng)

    methods_cs = [
        ("GMOPCGM", GMOPCGMMethod()),
        ("GCGPM",   GCGPMMethod()),
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
        residuals = _track_solve(method, prob, z0, 1e-5, 5000)
        plot!(p, 1:length(residuals), residuals;
              label=mname, color=METHOD_COLORS[midx],
              linestyle=METHOD_LSTYLE[midx], linewidth=METHOD_LW[midx])
        @printf("  %s: %d iterations, final ||G||=%.2e\n",
                mname, length(residuals), residuals[end])
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
    noise_levels = sort(unique(sub.noise_sigma))
    srs = sort(unique(sub.sparsity_ratio))

    p = plot(layout=(1, length(available)), size=(250*length(available), 350),
             margin=4Plots.mm, link=:y)

    for (j, m) in enumerate(available)
        midx = _midx(m)
        for sigma in noise_levels
            ss = filter(r -> r.method == m && r.noise_sigma == sigma, sub)
            med_mse = [let s = filter(r -> r.sparsity_ratio == sr, ss)
                        nrow(s) > 0 ? median(s.mse) : NaN
                       end for sr in srs]
            plot!(p[j], srs, med_mse;
                  label=(j==1 ? "σ=$sigma" : ""),
                  xlabel="k/n", ylabel=(j==1 ? "Median MSE" : ""),
                  yscale=:log10, title=m,
                  marker=:circle, markersize=3, linewidth=1.5)
        end
    end

    savefig(p, joinpath(IMGS_DIR, "cs_mse_vs_sparsity.pdf"))
    println("  Saved: cs_mse_vs_sparsity.pdf")

    # --- Plot B: MSE vs measurement ratio, fixed k/n=0.1, one curve per method ---
    sr_fixed = 0.1
    sigma_fixed = 0.01
    sub2 = filter(r -> r.sparsity_ratio == sr_fixed && r.noise_sigma == sigma_fixed, conv)
    mrs = sort(unique(sub2.measurement_ratio))

    p2 = plot(xlabel="m/n (measurement ratio)", ylabel="Median MSE",
              yscale=:log10, legend=:topright,
              size=(600, 400), minorgrid=true,
              title="Recovery quality: k/n=$sr_fixed, σ=$sigma_fixed")

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
    sigmas = sort(unique(sub3.noise_sigma))

    p3 = plot(xlabel="Noise level σ", ylabel="Median iterations",
              legend=:topleft, size=(600, 400), minorgrid=true,
              title="Solver effort: k/n=$sr_fixed, m/n=$mr_fixed")

    for m in available
        midx = _midx(m)
        ss = filter(r -> r.method == m, sub3)
        med_it = [let s = filter(r -> r.noise_sigma == sig, ss)
                   nrow(s) > 0 ? median(s.iterations) : NaN
                  end for sig in sigmas]
        plot!(p3, sigmas, med_it;
              label=m, color=METHOD_COLORS[midx],
              linestyle=METHOD_LSTYLE[midx], linewidth=METHOD_LW[midx],
              marker=METHOD_MARKER[midx], markersize=4)
    end

    savefig(p3, joinpath(IMGS_DIR, "cs_iters_vs_noise.pdf"))
    println("  Saved: cs_iters_vs_noise.pdf")

    # --- Plot D: Phase transition — convergence rate (%) vs (sparsity, measurement) ---
    # One heatmap per method showing % of trials converged
    noise_fixed = 0.01
    sub4 = filter(r -> r.noise_sigma == noise_fixed, df)
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
