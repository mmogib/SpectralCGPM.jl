using Test

include(joinpath(@__DIR__, "..", "src", "includes.jl"))

@testset "SOPP and SDLP public names" begin
    @test isdefined(Main, :SOPPMethod)
    @test isdefined(Main, :SDLPMethod)
    @test isdefined(Main, :_sopp_next_direction)
    @test isdefined(Main, :_sdlp_next_direction)
    if isdefined(Main, :SOPPMethod) && isdefined(Main, :SDLPMethod)
        @test method_name(SOPPMethod()) == "SOPP"
        @test method_name(SDLPMethod()) == "SDLP"
    end
end

@testset "Rebuilt method parameter contracts" begin
    gm = SOPPMethod()
    @test gm.c == gm.alpha_max + 1.0
    @test_throws AssertionError SOPPMethod(alpha_min=0.0)
    @test_throws AssertionError SOPPMethod(alpha_min=1.1)
    @test_throws AssertionError SOPPMethod(alpha_max=0.9)
    @test_throws AssertionError SOPPMethod(c=2.0, alpha_max=2.0)

    @test_throws AssertionError SDLPMethod(tau=0.0, alpha_min=0.5)
    @test SDLPMethod(tau=0.0, alpha_min=0.5001).alpha_min == 0.5001
end

@testset "SOPP rebuilt direction" begin
    m = SOPPMethod(tau=1.0, alpha_min=0.1, alpha_max=2.0, c=3.0)
    Gx = [1.0, 2.0]
    Gz = [1.5, 2.2]
    Gnext = [0.4, -0.3]
    s = [0.5, -0.25]
    p = [1.0, -0.5]

    q = Gz .- Gx .+ m.tau .* s
    sq = dot(s, q)
    s2 = dot(s, s)
    q2 = dot(q, q)
    expected_lambda = clamp(max(s2 / sq, sq / q2), m.alpha_min, m.alpha_max)
    expected_t = expected_lambda / 2 * (sq / s2 + q2 / sq)
    expected_theta = dot(q .- expected_t .* s, Gnext) / dot(p, q)
    pperp = p .- (dot(Gnext, p) / dot(Gnext, Gnext)) .* Gnext
    clip_limit = sqrt(m.c^2 - expected_lambda^2) * norm(Gnext) / norm(pperp)
    expected_bar_theta = sign(expected_theta) * min(abs(expected_theta), clip_limit)
    expected_direction = -expected_lambda .* Gnext .+ expected_bar_theta .* pperp

    actual = _sopp_next_direction(m, Gx, Gz, Gnext, s, p, 1.0)
    @test !actual.restarted
    @test actual.lambda ≈ expected_lambda
    @test actual.direction ≈ expected_direction
    @test abs(dot(Gnext, actual.direction) + expected_lambda * dot(Gnext, Gnext)) ≤ 1e-12
    @test norm(actual.direction) ≤ m.c * norm(Gnext) + 1e-12

    bad = _sopp_next_direction(m, zeros(2), [-2.0, 0.0], [1.0, -1.0], [1.0, 0.0], [1.0, 0.0], 1.2)
    @test bad.restarted
    @test bad.lambda == 1.2
    @test bad.direction == -1.2 .* [1.0, -1.0]
end

@testset "SDLP restored correction and direction" begin
    m = SDLPMethod(tau=0.1, alpha_min=0.6, alpha_max=2.0)
    Gx = [1.0, 2.0]
    Gnext = [0.4, -0.3]
    p = [-1.0, 0.5]

    y = Gnext .- Gx
    p2 = dot(p, p)
    r = 1.0 + max(0.0, -dot(y, p) / p2)
    w = y .+ r .* p
    pw = dot(p, w)
    w2 = dot(w, w)
    expected_lambda = clamp(max(w2 / pw, pw / p2), m.alpha_min, m.alpha_max)
    a = dot(Gnext, p) / pw
    theta = dot(Gnext, w) / pw - expected_lambda * (w2 / pw) * (dot(Gnext, p) / pw)
    expected_direction = -expected_lambda .* Gnext .+ theta .* p .+ m.tau .* a .* w

    actual = _sdlp_next_direction(m, Gx, Gnext, p, 1.0)
    @test !actual.restarted
    @test actual.lambda ≈ expected_lambda
    @test actual.direction ≈ expected_direction

    bad = _sdlp_next_direction(m, Gx, Gnext, zeros(2), 1.3)
    @test bad.restarted
    @test bad.lambda == 1.3
    @test bad.direction == -1.3 .* Gnext
end

@testset "Projection, residual guards, gamma bounds, and diagnostics" begin
    projected_root = TestProblem(901, "projected-root", x -> copy(x), x -> max.(x, 0.0), "test")
    for method in (SOPPMethod(), SDLPMethod())
        result = solve(method, projected_root, [-1.0, -2.0]; eps=1e-12, maxiter=2)
        @test result.converged
        @test result.status == :converged_initial
        @test result.x == zeros(2)
        @test result.restarts == 0
    end

    identity_problem = TestProblem(902, "projection-root", x -> copy(x), x -> copy(x), "test")
    gm = SOPPMethod(beta=2 / 3, rho=0.5, zeta=0.1, gamma=1.5, zeta1=1.0, zeta2=1.0)
    gc = SDLPMethod(tau=0.0, alpha_min=0.6, eta=2 / 3, rho=0.5,
                     zeta=0.1, gamma=1.5, zeta1=1.0, zeta2=1.0)
    for method in (gm, gc)
        result = solve(method, identity_problem, [1.0, 0.0]; eps=1e-12, maxiter=2)
        @test result.converged
        @test result.status == :converged_projected
        @test result.iterations == 1
        @test result.x == zeros(2)
        @test result.restarts == 0
    end

    @test _update_gamma(gm, 1.7, true) == 1.8
    @test _update_gamma(gm, 1.2, false) == 1.2
    @test _update_gamma(gc, 1.9, true) == 1.95
    @test _update_gamma(gc, 1.9, false) == 1.95

    infeasible = TestProblem(903, "infeasible", x -> copy(x), x -> max.(x, 0.0), "test")
    @test !_is_feasible(infeasible, [-1.0, 0.0])
    @test _is_feasible(infeasible, [1.0, 0.0])

    stalled = SolverResult(false, 3, 4, 1.0, 0.1, ones(2); status=:stalled, restarts=2)
    @test stalled.status == :stalled
    @test stalled.restarts == 2
end

@testset "Backtracking evaluation accounting" begin
    fe = Ref(0)
    counted_identity(x) = (fe[] += 1; copy(x))
    alpha, _, _ = _backtrack_ours(
        counted_identity, [1.0], [-1.0], 0.5, 0.5, 0.1, 1.0, 1.0, fe)
    @test alpha == 0.5
    @test fe[] == 1

    fe[] = 0
    counted_failure(x) = (fe[] += 1; -ones(length(x)))
    alpha, _, Gz = _backtrack_ours(
        counted_failure, [1.0], [-1.0], 0.5, 1e-16, 0.1, 1.0, 1.0, fe)
    @test alpha === nothing
    @test Gz === nothing
    @test fe[] == 1
end

@testset "Rerun problem definitions and domain handling" begin
    p7_at_ones = G8(ones(4))
    @test p7_at_ones[end] == 1.0

    @test all(isnan, G2([-1.0, 0.0], 2))
    @test all(isnan, G2([-1.1, 0.0], 2))
    @test G2(zeros(2), 2) == zeros(2)
    @test all(isnan, G18([0.0, 1.0]))
    @test all(isnan, G18([-0.1, 1.0]))
    @test all(isfinite, G18(ones(2)))

    fe = Ref(0)
    nan_then_finite(x) = begin
        fe[] += 1
        x[1] <= -1.0 ? [NaN] : [1.0]
    end
    alpha, _, Gz = _backtrack_ours(
        nan_then_finite, [0.0], [-1.0], 0.25, 2.0, 0.1, 1.0, 1.0, fe)
    @test alpha == 0.5
    @test Gz == [1.0]
    @test fe[] == 2

    p2 = get_problem(2, 2)
    p2_method = SOPPMethod(beta=100.0, rho=0.5)
    p2_result = @test_nowarn solve(
        p2_method, p2, ones(2); eps=1e-6, maxiter=2, cb=ProgressCallback())
    @test p2_result isa SolverResult

    p16 = get_problem(18, 2)
    p16_method = SDLPMethod(eta=10.0, rho=0.5)
    p16_result = @test_nowarn solve(
        p16_method, p16, fill(2.0, 2); eps=1e-6, maxiter=2,
        cb=ProgressCallback())
    @test p16_result isa SolverResult
end

@testset "Benchmark diagnostic schemas" begin
    rows = run_benchmark(
        AbstractMethod[SOPPMethod()], [1], [2],
        n -> [(ones(n), "ones")]; eps=1e-6, maxiter=1)
    @test length(rows) == 1
    @test hasproperty(rows[1], :status)
    @test hasproperty(rows[1], :restarts)

    s45_source = read(
        joinpath(@__DIR__, "..", "scripts", "s45_benchmark.jl"), String)
    @test occursin("\"status\"", s45_source)
    @test occursin("\"restarts\"", s45_source)
    @test occursin("result.status", s45_source)
    @test occursin("result.restarts", s45_source)
end

@testset "Honest convergence-site audit" begin
    solver_source = read(joinpath(@__DIR__, "..", "src", "solvers.jl"), String)
    @test !occursin("SolverResult(true", solver_source)
    @test occursin("stall_limit::Int=typemax(Int)", solver_source)
end

@testset "Production iteration hook contract" begin
    identity_problem = TestProblem(
        905, "hook-identity", x -> copy(x), x -> copy(x), "test")
    methods = AbstractMethod[
        SOPPMethod(beta=2 / 3, rho=0.5, zeta=0.1,
                   gamma=1.5, zeta1=1.0, zeta2=1.0),
        SDLPMethod(tau=0.0, alpha_min=0.6, eta=2 / 3, rho=0.5,
                   zeta=0.1, gamma=1.5, zeta1=1.0, zeta2=1.0),
        MOPCGMMethod(),
        CGPMMethod(),
        STTDFPMMethod(),
    ]

    solver_source = read(joinpath(@__DIR__, "..", "src", "solvers.jl"), String)
    hook_available = occursin(
        "on_iter::Union{Nothing,Function}=nothing", solver_source)
    @test hook_available

    if hook_available
      for method in methods
        trace = Tuple{Int,Float64}[]
        hooked = solve(
            method, identity_problem, [1.0]; eps=1e-8, maxiter=1000,
            on_iter=(k, residual) -> push!(trace, (k, Float64(residual))))
        plain = solve(method, identity_problem, [1.0]; eps=1e-8, maxiter=1000)

        @test hooked.converged
        @test !isempty(trace)
        @test first(trace)[1] == 0
        @test getindex.(trace, 1) == collect(0:hooked.iterations)
        @test length(unique(getindex.(trace, 1))) == length(trace)
        @test all(isfinite(last(entry)) for entry in trace)
        @test last(trace)[2] == hooked.residual
        @test norm(identity_problem.G(hooked.x)) == hooked.residual

        @test plain.converged == hooked.converged
        @test plain.status == hooked.status
        @test plain.restarts == hooked.restarts
        @test plain.iterations == hooked.iterations
        @test plain.f_evals == hooked.f_evals
        @test plain.residual == hooked.residual
        @test plain.x == hooked.x

        initial_trace = Tuple{Int,Float64}[]
        initial = solve(
            method, identity_problem, zeros(1); eps=1e-12, maxiter=2,
            on_iter=(k, residual) ->
                push!(initial_trace, (k, Float64(residual))))
        @test initial.status == :converged_initial
        @test initial.iterations == 0
        @test initial_trace == [(0, 0.0)]
      end

      flip_problem = TestProblem(
        906, "hook-preupdate-failure",
        x -> x[1] < 0.0 ? [-1.0] : [1.0], x -> copy(x), "test")
    failure_trace = Tuple{Int,Float64}[]
    failure = solve(
        SOPPMethod(beta=1.0, rho=0.5, zeta=0.1,
                   zeta1=1.0, zeta2=1.0),
        flip_problem, [0.0]; eps=1e-12, maxiter=2,
        on_iter=(k, residual) ->
            push!(failure_trace, (k, Float64(residual))))
    @test !failure.converged
    @test failure.status == :line_search_failed
    @test failure.iterations == 1
      @test failure_trace == [(0, 1.0)]
    end
end

@testset "Monotonic elapsed-time conversion" begin
    @test isdefined(Main, :_elapsed_seconds)
    if isdefined(Main, :_elapsed_seconds)
        @test _elapsed_seconds(UInt64(10), UInt64(1_000_000_010)) == 1.0
        wrapped = _elapsed_seconds(
            typemax(UInt64) - UInt64(499), UInt64(500))
        @test wrapped ≈ 1e-6 atol=eps(Float64)
    end
end

@testset "Metrics script contracts" begin
    scripts_dir = joinpath(@__DIR__, "..", "scripts")
    s45 = read(joinpath(scripts_dir, "s45_benchmark.jl"), String)
    s50 = read(joinpath(scripts_dir, "s50_signal_restore.jl"), String)
    s55 = read(joinpath(scripts_dir, "s55_logreg.jl"), String)
    s70 = read(joinpath(scripts_dir, "s70_figures.jl"), String)
    solver_source = read(joinpath(@__DIR__, "..", "src", "solvers.jl"), String)

    @test occursin("mse = norm(x_orig - x_rec)^2 / n", s50)
    @test occursin(
        "rel_err = norm(x_orig - x_rec) / norm(x_orig)", s50)
    @test occursin("\"rel_err\"", s50)
    @test occursin(
        "Set{Tuple{String,Float64,Float64,Float64,Int}}", s50)
    @test occursin("Resume schema mismatch", s50)
    @test occursin("ERROR:", s50)

    @test occursin("mse = norm(x_orig - x_rec)^2 / n", s70)
    @test occursin("function _solve_with_history", s70)
    @test !occursin("function _collect_residuals", s70)
    @test !occursin("function _track_solve", s70)

    @test occursin("yyyymmdd_HHMMSS", s45)
    @test occursin("%.9f", s45)
    @test occursin("%.9f", s50)
    @test occursin("%.9f", s55)
    @test length(findall("time_ns()", solver_source)) >= 6
    @test !occursin("t0 = time()", solver_source)
end

@testset "Compressed-sensing relative-noise naming contracts" begin
    scripts_dir = joinpath(@__DIR__, "..", "scripts")
    s50 = read(joinpath(scripts_dir, "s50_signal_restore.jl"), String)
    s70 = read(joinpath(scripts_dir, "s70_figures.jl"), String)

    @test !occursin("NOISE_AMPLITUDE", s50)
    @test occursin("NOISE_RATIOS", s50)
    @test occursin("NOISE_RATIOS_RUN", s50)
    @test occursin(
        "noise_ratio > 0 ? noise_ratio * 0.001 * randn(rng, m) : zeros(m)",
        s50)
    @test occursin("\"noise_ratio\"", s50)
    @test !occursin("noise_sigma", s50)
    @test !occursin(r"\bsigma\b", s50)

    @test occursin("noise_ratio = CS_FIG_NOISE_RATIO", s70)
    @test occursin(
        "noise = noise_ratio * 0.001 * randn(rng, m)", s70)
    @test !occursin("noise_sigma", s70)
end

@testset "Figure and table generation contracts" begin
    scripts_dir = joinpath(@__DIR__, "..", "scripts")
    s70 = read(joinpath(scripts_dir, "s70_figures.jl"), String)
    s75 = read(joinpath(scripts_dir, "s75_tables.jl"), String)
    contains_parse_error(x) = x isa Expr &&
        (x.head === :error || any(contains_parse_error, x.args))

    @test !contains_parse_error(Meta.parseall(s70))
    @test !contains_parse_error(Meta.parseall(s75))

    @test !occursin("1e-15", s70)
    @test occursin(
        "all(t -> isfinite(t) && t > 0.0, df.cpu_time)", s70)
    @test occursin(
        "joinpath(RESULTS_DIR, \"benchmark\", \"raw.csv\")", s70)
    @test occursin(
        "joinpath(RESULTS_DIR, \"signal_restore\", \"cs_sweep.csv\")", s70)
    @test occursin("const CS_FIG_SEED = 42", s70)
    @test occursin("const CS_FIG_N = 2^12", s70)
    @test occursin("const CS_FIG_SPARSITY_RATIO = 0.10", s70)
    @test occursin("const CS_FIG_MEASUREMENT_RATIO = 0.50", s70)
    @test occursin("const CS_FIG_NOISE_RATIO = 0.01", s70)
    @test occursin(
        "k = round(Int, CS_FIG_SPARSITY_RATIO * n)", s70)
    @test occursin(
        "m = round(Int, CS_FIG_MEASUREMENT_RATIO * n)", s70)
    @test occursin("Random.Xoshiro(CS_FIG_SEED)", s70)
    @test length(findall("_solve_with_history(", s70)) >= 3

    @test !occursin("\\\\documentclass", s75)
    @test !occursin("\\\\newcommand", s75)
    @test !occursin("\\\\begin{document}", s75)
    @test !occursin("round(median(conv.cpu_time)", s75)
    @test occursin("format_median", s75)
    @test occursin("table_P.tex", s75)
    @test occursin("\\\\label{tab:perproblem}", s75)
    @test occursin(
        "for (paper_problem_number, internal_problem_id) in enumerate(internal_problem_ids)",
        s75)
    @test occursin("problem = string(\"P\", internal_problem_id)", s75)
    @test occursin("logreg_results.csv", s75)
    @test occursin("table_F.tex", s75)
    @test occursin("\\\\label{tab:logreg}", s75)
end

@testset "Accepted-paper Problem 9 convergence contract" begin
    s70 = read(joinpath(@__DIR__, "..", "scripts", "s70_figures.jl"), String)

    @test PROBLEM_IDS[9] == 11
    @test occursin("const CONVERGENCE_PAPER_PROBLEM = 9", s70)
    @test occursin("const CONVERGENCE_INTERNAL_PROBLEM = 11", s70)
    @test occursin(
        "PROBLEM_IDS[CONVERGENCE_PAPER_PROBLEM] == CONVERGENCE_INTERNAL_PROBLEM",
        s70)
    @test occursin("paper_prob_id = CONVERGENCE_PAPER_PROBLEM", s70)
    @test occursin("internal_prob_id = PROBLEM_IDS[paper_prob_id]", s70)
    @test occursin("prob = get_problem(internal_prob_id, dim)", s70)
    @test occursin(
        "title=\"Convergence on P\$(paper_prob_id), n=\$dim\"", s70)
    @test occursin(
        "filename = \"convergence_P\$(paper_prob_id)_n\$(dim).pdf\"", s70)
    @test !occursin("prob_id=5", s70)
    @test !occursin("title=\"Convergence on \$(prob.name)", s70)
end
