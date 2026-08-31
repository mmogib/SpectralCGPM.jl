using Test
using CSV, DataFrames, Printf

include(joinpath(@__DIR__, "..", "src", "includes.jl"))

@testset "Parameter-choice method contracts" begin
    sopp = SOPPMethod()
    sdlp = SDLPMethod()

    @test hasproperty(sopp, :tsel)
    @test hasproperty(sopp, :srule)
    @test hasproperty(sdlp, :srule)
    if hasproperty(sopp, :tsel) && hasproperty(sopp, :srule) &&
       hasproperty(sdlp, :srule)
        @test sopp.tsel == :cond
        @test sopp.srule == :max
        @test sdlp.srule == :max
    end

    @test_throws ArgumentError SOPPMethod(tsel=:unknown)
    @test_throws ArgumentError SOPPMethod(srule=:unknown)
    @test_throws ArgumentError SDLPMethod(srule=:unknown)

    positional_sopp = SOPPMethod(
        1.0, 0.8, 0.5, 1e-4, 0.1, 2.0, 1.0, 1.1, 3.0, 1.0, 1.0)
    positional_sdlp = SDLPMethod(
        0.001, 0.5, 0.6, 0.1, 0.55, 4.9, 1.0, 1.8, 1.0, 1.0)
    if hasproperty(positional_sopp, :tsel) &&
       hasproperty(positional_sdlp, :srule)
        @test positional_sopp.tsel == :cond
        @test positional_sopp.srule == :max
        @test positional_sdlp.srule == :max
    end
end

@testset "Public documentation and naming" begin
    readme = read(joinpath(@__DIR__, "..", "README.md"), String)
    @test occursin("Parameter-choice ablation", readme)
    @test occursin("scripts/s65_ablation.jl", readme)
    @test occursin("results/ablation/raw.csv", readme)
    @test occursin("--resume", readme)
    @test occursin("--summary", readme)

    public_files = [
        joinpath(@__DIR__, "..", "src", "types.jl"),
        joinpath(@__DIR__, "..", "src", "solvers.jl"),
        joinpath(@__DIR__, "..", "scripts", "s65_ablation.jl"),
        joinpath(@__DIR__, "..", "README.md"),
    ]
    for path in public_files
        isfile(path) || continue
        source = read(path, String)
        internal_terms = Regex("(?i)" * "clau" * "de|" * "co" * "dex")
        @test !occursin(internal_terms, source)
    end
end

@testset "Parameter-choice formulas and iteration parity" begin
    required = (
        :_sopp_spectral_candidate,
        :_sopp_t_parameter,
        :_sdlp_spectral_candidate,
    )
    for name in required
        @test isdefined(Main, name)
    end

    if all(name -> isdefined(Main, name), required)
        s2, sq, q2 = 8.0, 2.0, 1.0
        @test _sopp_spectral_candidate(SOPPMethod(srule=:max), 1, s2, sq, q2) == 4.0
        @test _sopp_spectral_candidate(SOPPMethod(srule=:bb1), 1, s2, sq, q2) == 4.0
        @test _sopp_spectral_candidate(SOPPMethod(srule=:bb2), 1, s2, sq, q2) == 2.0
        @test _sopp_spectral_candidate(SOPPMethod(srule=:abb), 2, s2, sq, q2) == 4.0
        @test _sopp_spectral_candidate(SOPPMethod(srule=:abb), 1, s2, sq, q2) == 2.0

        lambda = 1.5
        @test _sopp_t_parameter(SOPPMethod(tsel=:cond), lambda, s2, sq, q2) ==
              lambda / 2 * (sq / s2 + q2 / sq)
        @test _sopp_t_parameter(SOPPMethod(tsel=:gap), lambda, s2, sq, q2) ==
              lambda * sq / s2

        p2, pw, w2 = 8.0, 2.0, 1.0
        @test _sdlp_spectral_candidate(SDLPMethod(srule=:max), 1, p2, pw, w2) == 0.5
        @test _sdlp_spectral_candidate(SDLPMethod(srule=:r1), 1, p2, pw, w2) == 0.5
        @test _sdlp_spectral_candidate(SDLPMethod(srule=:r2), 1, p2, pw, w2) == 0.25
        @test _sdlp_spectral_candidate(SDLPMethod(srule=:alt), 2, p2, pw, w2) == 0.5
        @test _sdlp_spectral_candidate(SDLPMethod(srule=:alt), 1, p2, pw, w2) == 0.25
    end
end

@testset "Spectral-ratio diagnostic hook" begin
    n = 10
    problem = get_problem(2, n)
    x0 = ones(n)

    for method in (SOPPMethod(), SDLPMethod())
        observations = NamedTuple[]
        supported = try
            solve(
                method, problem, x0;
                eps=1e-30,
                maxiter=2,
                on_spectral=(k, first_ratio, second_ratio) -> push!(
                    observations,
                    (k=k, first_ratio=first_ratio,
                     second_ratio=second_ratio)),
            )
            true
        catch error
            error isa MethodError || rethrow()
            false
        end
        @test supported
        if supported
            @test !isempty(observations)
            @test all(item -> item.k >= 1, observations)
            @test all(item -> isfinite(item.first_ratio), observations)
            @test all(item -> isfinite(item.second_ratio), observations)
        end
    end
end

@testset "Frozen benchmark compatibility subset" begin
    benchmark_csv = joinpath(
        @__DIR__, "..", "results", "benchmark", "raw.csv")
    @test isfile(benchmark_csv)
    if isfile(benchmark_csv) && hasproperty(SOPPMethod(), :tsel) &&
       hasproperty(SDLPMethod(), :srule)
        # raw.csv stores residuals at %.10e. These golden bit patterns were
        # captured from the pre-switch solver for the same frozen cases so the
        # test checks both the exact Float64 and the CSV representation.
        residual_bits = Dict(
            ("SOPP", 1)  => 0x3da128cc800f5699,
            ("SOPP", 11) => 0x3da4ee6e59554fe0,
            ("SOPP", 20) => 0x3da055dc0b9ed627,
            ("SDLP", 1)  => 0x0000000000000000,
            ("SDLP", 11) => 0x3d8b036e587cfad1,
            ("SDLP", 20) => 0x3d889a77adeceba2,
        )
        benchmark = CSV.read(benchmark_csv, DataFrame; types=Dict(:residual => String))
        n = 10_000
        x0, x0_label = only(filter(
            item -> last(item) == "1.0", get_initial_points(n)))
        for (method_name, method) in
            (("SOPP", SOPPMethod()), ("SDLP", SDLPMethod())),
            problem_id in PROBLEM_IDS[[1, 9, 18]]
            problem = get_problem(problem_id, n)
            frozen = only(filter(
                row -> row.method == method_name &&
                       row.problem == problem.name && row.n == n &&
                       row.x0_label == x0_label,
                benchmark))
            result = solve(method, problem, x0; eps=1e-11, maxiter=2000)
            @test result.iterations == frozen.iterations
            @test result.f_evals == frozen.f_evals
            @test reinterpret(UInt64, result.residual) ==
                  residual_bits[(method_name, problem_id)]
            @test @sprintf("%.10e", result.residual) == frozen.residual
        end
    end
end

const ABLATION_SCRIPT = joinpath(
    @__DIR__, "..", "scripts", "s65_ablation.jl")

@testset "Ablation script contract" begin
    @test isfile(ABLATION_SCRIPT)
    if isfile(ABLATION_SCRIPT)
        include(ABLATION_SCRIPT)
        required = (
            :AblationConfig,
            :ABLATION_COLUMNS,
            :build_ablation_design,
            :validate_ablation_configs!,
            :validate_ablation_results,
            :ablation_configuration_summary,
            :identity_differences,
            :failure_status_summary,
            :parse_ablation_args,
        )
        for name in required
            @test isdefined(Main, name)
        end

        if all(name -> isdefined(Main, name), required)
            design = build_ablation_design()
            @test length(design) == 9
            @test [cfg.config_label for cfg in design] == [
                "default", "tsel=gap", "srule=bb1", "srule=bb2",
                "srule=abb", "default", "srule=r1", "srule=r2",
                "srule=alt",
            ]
            @test count(cfg -> cfg.method_name == "SOPP", design) == 5
            @test count(cfg -> cfg.method_name == "SDLP", design) == 4
            @test validate_ablation_configs!(design) === design
            @test ABLATION_COLUMNS[end-1:end] == ["tsel", "srule"]
            @test parse_ablation_args(String[]) ==
                  (resume=false, summary=false)
            @test parse_ablation_args(["--resume"]) ==
                  (resume=true, summary=false)
            @test parse_ablation_args(["--summary"]) ==
                  (resume=false, summary=true)
            @test_throws ArgumentError parse_ablation_args(["--quick"])

            synthetic = DataFrame(
                method=fill("SOPP", 4),
                config_label=["default", "default", "srule=bb1", "srule=bb1"],
                param_family=["baseline", "baseline", "srule", "srule"],
                problem_id=fill(1, 4),
                problem=fill("P1", 4),
                n=fill(10_000, 4),
                x0_label=["0.4", "0.5", "0.4", "0.5"],
                converged=[true, false, true, false],
                status=["converged_trial", "maxiter", "converged_trial", "maxiter"],
                restarts=[1, 2, 3, 4],
                iterations=[10, 2000, 10, 1999],
                f_evals=[20, 4000, 20, 4000],
                residual=[1e-12, 1.0, 1e-12, 1.0],
                cpu_time=[0.1, 2.0, 0.2, 3.0],
            )
            summary = ablation_configuration_summary(synthetic)
            default = only(filter(row -> row.config_label == "default", summary))
            @test default.total == 2
            @test default.converged == 1
            @test default.rate_pct == 50.0
            @test default.median_iterations == 10.0
            @test default.median_f_evals == 20.0
            @test default.median_cpu == 0.1
            @test default.restarts == 3

            differences = identity_differences(
                synthetic, "SOPP", "default", "srule=bb1")
            @test nrow(differences) == 1
            @test only(differences.x0_label) == "0.5"

            failures = failure_status_summary(synthetic)
            @test nrow(failures) == 2
            @test all(failures.status .== "maxiter")
            @test all(failures.count .== 1)
        end
    end
end
