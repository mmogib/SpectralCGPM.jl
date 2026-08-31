using Test

const SENSITIVITY_SCRIPT = joinpath(
    @__DIR__, "..", "scripts", "s60_sensitivity.jl")

@testset "Sensitivity script exists" begin
    @test isfile(SENSITIVITY_SCRIPT)
end

include(SENSITIVITY_SCRIPT)

const REQUIRED_SENSITIVITY_API = [
    :SensitivityConfig,
    :SENSITIVITY_COLUMNS,
    :build_candidate_configs,
    :deduplicate_configs,
    :build_sensitivity_design,
    :canonical_config_key,
    :validate_config!,
    :validate_configs!,
    :run_key,
    :expected_instance_keys,
    :validate_results,
    :configuration_summary,
    :family_band_summary,
    :parse_sensitivity_args,
]

@testset "Sensitivity script public contract" begin
    for name in REQUIRED_SENSITIVITY_API
        @test isdefined(Main, name)
    end
end

if all(name -> isdefined(Main, name), REQUIRED_SENSITIVITY_API)
    @testset "Sensitivity CSV schema" begin
        @test SENSITIVITY_COLUMNS == [
            "method", "config_label", "param_family", "problem_id",
            "problem", "n", "x0_label", "tau", "rho", "beta", "eta",
            "zeta", "alpha_min", "alpha_max", "lambda0", "gamma", "c",
            "zeta1", "zeta2", "eps", "maxiter", "converged", "status",
            "restarts", "iterations", "f_evals", "residual", "cpu_time",
        ]
    end

    @testset "Sensitivity grid legality and deduplication" begin
        candidates, memberships, ranges = build_candidate_configs()
        @test count(c -> c.method_name == "SOPP", candidates) == 19
        @test count(c -> c.method_name == "SDLP", candidates) == 19
        @test length(memberships) == 10
        @test length(ranges) == 10

        configs = deduplicate_configs(candidates)
        @test length(configs) == 30
        @test count(c -> c.method_name == "SOPP", configs) == 15
        @test count(c -> c.method_name == "SDLP", configs) == 15
        @test count(c -> c.config_label == "default", configs) == 2
        @test all(c -> !occursin(',', c.config_label), configs)
        @test length(unique(canonical_config_key.(configs))) == 30
        @test length(unique((c.method_name, c.config_label) for c in configs)) == 30

        sopp_alpha = filter(
            c -> c.method_name == "SOPP" && c.param_family == "alpha",
            configs)
        @test any(c -> c.method.alpha_max == 2.5, sopp_alpha)
        @test all(c -> c.method.alpha_max != 5.0, sopp_alpha)
        @test all(c -> c.method.c == 3.0, sopp_alpha)

        design = build_sensitivity_design()
        @test design.configs == configs
        @test all(length(labels) >= 3 for labels in values(design.family_members))
        @test all("default" in labels for labels in values(design.family_members))
        @test sum(length, values(design.family_members)) == 38
        @test validate_configs!(design.configs) === design.configs
    end

    @testset "Sensitivity legality gate rejects invalid constructors" begin
        bad_sopp_method = SOPPMethod(
            1.0, 0.8, 0.5, 1e-4, 0.1, 2.0, 1.0, 1.1, 2.0, 1.0, 1.0)
        illegal_sopp = SensitivityConfig(
            "SOPP", "alpha", "illegal-alpha", bad_sopp_method)
        @test_throws AssertionError validate_config!(illegal_sopp)

        bad_sdlp_method = SDLPMethod(
            0.1, 0.5, 0.6, 0.1, 0.55, 4.9, 1.0, 1.8, 1.0, 1.0)
        illegal_sdlp = SensitivityConfig(
            "SDLP", "tau", "illegal-tau", bad_sdlp_method)
        @test_throws AssertionError validate_config!(illegal_sdlp)
    end

    @testset "Sensitivity key and coverage contracts" begin
        design = build_sensitivity_design()
        cfg = first(design.configs)
        @test run_key(cfg, 1, 10_000, "0.4") ==
              (cfg.method_name, cfg.config_label, 1, 10_000, "0.4")
        instances = expected_instance_keys()
        @test length(instances) == 180
        @test length(unique(instances)) == 180
        @test Set(first.(instances)) == Set(PROBLEM_IDS)
        @test Set(last.(instances)) == Set(last.(get_initial_points(10_000)))
    end

    @testset "Sensitivity argument parsing" begin
        @test parse_sensitivity_args(String[]) ==
              (resume=false, summary=false)
        @test parse_sensitivity_args(["--resume"]) ==
              (resume=true, summary=false)
        @test parse_sensitivity_args(["--summary"]) ==
              (resume=false, summary=true)
        @test_throws ArgumentError parse_sensitivity_args(["--quick"])
    end

    @testset "Sensitivity summaries use converged rows and shared defaults" begin
        raw = DataFrame(
            method=["SOPP", "SOPP", "SOPP", "SOPP"],
            config_label=["default", "default", "tau=0.1", "tau=0.1"],
            converged=[true, false, true, true],
            iterations=[10, 999, 20, 40],
            f_evals=[30, 999, 60, 100],
            cpu_time=[0.1, 9.9, 0.2, 0.4],
        )
        summary = configuration_summary(raw)
        base = only(filter(r -> r.config_label == "default", summary))
        tau = only(filter(r -> r.config_label == "tau=0.1", summary))
        @test base.total == 2
        @test base.converged == 1
        @test base.rate_pct == 50.0
        @test base.median_iterations == 10.0
        @test base.median_f_evals == 30.0
        @test base.median_cpu == 0.1
        @test tau.rate_pct == 100.0
        @test tau.median_f_evals == 80.0

        design = build_sensitivity_design()
        synthetic = DataFrame(
            method=[c.method_name for c in design.configs],
            param_family=[c.param_family for c in design.configs],
            config_label=[c.config_label for c in design.configs],
            total=fill(180, length(design.configs)),
            converged=collect(151:180),
            rate_pct=collect(151:180) ./ 1.8,
            median_iterations=collect(1.0:30.0),
            median_f_evals=collect(101.0:130.0),
            median_cpu=collect(0.01:0.01:0.30),
        )
        bands = family_band_summary(synthetic, design)
        @test nrow(bands) == 10
        @test all(bands.n_configs .== [
            length(design.family_members[(r.method, r.param_family)])
            for r in eachrow(bands)
        ])
    end
end
