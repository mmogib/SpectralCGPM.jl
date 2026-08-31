# ============================================================================
# s65: SOPP/SDLP Parameter-Choice Ablation Study
# ============================================================================
#
# Protocol:
#   9 configurations x 18 problems x 10 initial points at n=10,000.
#
# Usage (from jcode/):
#   julia --threads=1 --project=. scripts/s65_ablation.jl
#   julia --threads=1 --project=. scripts/s65_ablation.jl --resume
#   julia --threads=1 --project=. scripts/s65_ablation.jl --summary
#
# Output:
#   results/ablation/raw.csv
#   results/logs/ablation_TIMESTAMP.log
# ============================================================================

if !isdefined(Main, :SOPPMethod)
    include(joinpath(@__DIR__, "..", "src", "includes.jl"))
end
using CSV, DataFrames, Dates, LinearAlgebra, Printf, ProgressMeter, Statistics

const ABLATION_N = 10_000
const ABLATION_EPS = 1e-11
const ABLATION_MAXITER = 2000
const ABLATION_COLUMNS = [
    "method", "config_label", "param_family", "problem_id", "problem", "n",
    "x0_label", "tau", "rho", "beta", "eta", "zeta", "alpha_min",
    "alpha_max", "lambda0", "gamma", "c", "zeta1", "zeta2", "eps",
    "maxiter", "converged", "status", "restarts", "iterations", "f_evals",
    "residual", "cpu_time", "tsel", "srule",
]
const ABLATION_NONFINITE_STATUSES = Set([
    "nonfinite", "invalid_trial_residual",
])

# These are the Table 3 values used by the sensitivity study.
const SOPP_ABLATION_DEFAULTS = (
    tau=1.0, rho=0.8, beta=0.5, zeta=1e-4,
    alpha_min=0.1, alpha_max=2.0, lambda0=1.0, gamma=1.1,
    c=3.0, zeta1=1.0, zeta2=1.0,
)
const SDLP_ABLATION_DEFAULTS = (
    tau=0.001, rho=0.5, eta=0.6, zeta=0.1,
    alpha_min=0.55, alpha_max=4.9, lambda0=1.0, gamma=1.8,
    zeta1=1.0, zeta2=1.0,
)

struct AblationConfig
    method_name::String
    param_family::String
    config_label::String
    method::AbstractMethod
end

_ablation_sopp(; kwargs...) = SOPPMethod(; merge(
    SOPP_ABLATION_DEFAULTS, (; kwargs...))...)
_ablation_sdlp(; kwargs...) = SDLPMethod(; merge(
    SDLP_ABLATION_DEFAULTS, (; kwargs...))...)

function build_ablation_design()
    return AblationConfig[
        AblationConfig("SOPP", "baseline", "default",
                       _ablation_sopp(tsel=:cond, srule=:max)),
        AblationConfig("SOPP", "tsel", "tsel=gap",
                       _ablation_sopp(tsel=:gap, srule=:max)),
        AblationConfig("SOPP", "srule", "srule=bb1",
                       _ablation_sopp(tsel=:cond, srule=:bb1)),
        AblationConfig("SOPP", "srule", "srule=bb2",
                       _ablation_sopp(tsel=:cond, srule=:bb2)),
        AblationConfig("SOPP", "srule", "srule=abb",
                       _ablation_sopp(tsel=:cond, srule=:abb)),
        AblationConfig("SDLP", "baseline", "default",
                       _ablation_sdlp(srule=:max)),
        AblationConfig("SDLP", "srule", "srule=r1",
                       _ablation_sdlp(srule=:r1)),
        AblationConfig("SDLP", "srule", "srule=r2",
                       _ablation_sdlp(srule=:r2)),
        AblationConfig("SDLP", "srule", "srule=alt",
                       _ablation_sdlp(srule=:alt)),
    ]
end

function _ablation_config_key(cfg::AblationConfig)
    m = cfg.method
    if m isa SOPPMethod
        return (
            cfg.method_name, cfg.config_label,
            m.tau, m.rho, m.beta, m.zeta, m.alpha_min, m.alpha_max,
            m.lambda0, m.gamma, m.c, m.zeta1, m.zeta2, m.tsel, m.srule,
        )
    elseif m isa SDLPMethod
        return (
            cfg.method_name, cfg.config_label,
            m.tau, m.rho, m.eta, m.zeta, m.alpha_min, m.alpha_max,
            m.lambda0, m.gamma, m.zeta1, m.zeta2, m.srule,
        )
    end
    throw(ArgumentError("Unsupported ablation method: $(typeof(m))"))
end

function validate_ablation_config!(cfg::AblationConfig)
    isempty(cfg.config_label) && throw(ArgumentError("Empty configuration label"))
    (occursin(',', cfg.config_label) || occursin('\n', cfg.config_label)) &&
        throw(ArgumentError("Configuration labels must be CSV-safe"))

    m = cfg.method
    if m isa SOPPMethod
        cfg.method_name == "SOPP" || throw(ArgumentError(
            "SOPP configuration has method label $(cfg.method_name)"))
        all(isfinite, (
            m.tau, m.rho, m.beta, m.zeta, m.alpha_min, m.alpha_max,
            m.lambda0, m.gamma, m.c, m.zeta1, m.zeta2,
        )) || throw(ArgumentError("SOPP configuration contains a nonfinite value"))
        SOPPMethod(
            tau=m.tau, rho=m.rho, beta=m.beta, zeta=m.zeta,
            alpha_min=m.alpha_min, alpha_max=m.alpha_max,
            lambda0=m.lambda0, gamma=m.gamma, c=m.c,
            zeta1=m.zeta1, zeta2=m.zeta2,
            tsel=m.tsel, srule=m.srule,
        )
    elseif m isa SDLPMethod
        cfg.method_name == "SDLP" || throw(ArgumentError(
            "SDLP configuration has method label $(cfg.method_name)"))
        all(isfinite, (
            m.tau, m.rho, m.eta, m.zeta, m.alpha_min, m.alpha_max,
            m.lambda0, m.gamma, m.zeta1, m.zeta2,
        )) || throw(ArgumentError("SDLP configuration contains a nonfinite value"))
        SDLPMethod(
            tau=m.tau, rho=m.rho, eta=m.eta, zeta=m.zeta,
            alpha_min=m.alpha_min, alpha_max=m.alpha_max,
            lambda0=m.lambda0, gamma=m.gamma,
            zeta1=m.zeta1, zeta2=m.zeta2, srule=m.srule,
        )
    else
        throw(ArgumentError("Unsupported ablation method: $(typeof(m))"))
    end
    return cfg
end

function validate_ablation_configs!(configs::Vector{AblationConfig})
    foreach(validate_ablation_config!, configs)
    length(configs) == 9 || throw(ArgumentError(
        "Expected 9 configurations, found $(length(configs))"))
    count(cfg -> cfg.method_name == "SOPP", configs) == 5 ||
        throw(ArgumentError("Expected 5 SOPP configurations"))
    count(cfg -> cfg.method_name == "SDLP", configs) == 4 ||
        throw(ArgumentError("Expected 4 SDLP configurations"))
    labels = [(cfg.method_name, cfg.config_label) for cfg in configs]
    length(unique(labels)) == length(labels) ||
        throw(ArgumentError("Configuration labels are not unique per method"))
    keys = _ablation_config_key.(configs)
    length(unique(keys)) == length(keys) ||
        throw(ArgumentError("Duplicate ablation configurations"))
    return configs
end

ablation_run_key(cfg::AblationConfig, problem_id::Integer, n::Integer,
                 x0_label::AbstractString) =
    (cfg.method_name, cfg.config_label, Int(problem_id), Int(n), String(x0_label))

function _ablation_expected_instances(n::Int=ABLATION_N)
    labels = last.(get_initial_points(n))
    return [(pid, label) for pid in PROBLEM_IDS for label in labels]
end

function _ablation_parameters(cfg::AblationConfig)
    m = cfg.method
    if m isa SOPPMethod
        return (
            tau=m.tau, rho=m.rho, beta=m.beta, eta=missing, zeta=m.zeta,
            alpha_min=m.alpha_min, alpha_max=m.alpha_max,
            lambda0=m.lambda0, gamma=m.gamma, c=m.c,
            zeta1=m.zeta1, zeta2=m.zeta2,
            tsel=string(m.tsel), srule=string(m.srule),
        )
    end
    return (
        tau=m.tau, rho=m.rho, beta=missing, eta=m.eta, zeta=m.zeta,
        alpha_min=m.alpha_min, alpha_max=m.alpha_max,
        lambda0=m.lambda0, gamma=m.gamma, c=missing,
        zeta1=m.zeta1, zeta2=m.zeta2,
        tsel=missing, srule=string(m.srule),
    )
end

function _ablation_result_row(cfg::AblationConfig, problem::TestProblem, n::Int,
                              x0_label::String, result::SolverResult)
    p = _ablation_parameters(cfg)
    return (
        method=cfg.method_name,
        config_label=cfg.config_label,
        param_family=cfg.param_family,
        problem_id=problem.id,
        problem=problem.name,
        n=n,
        x0_label=x0_label,
        tau=p.tau,
        rho=p.rho,
        beta=p.beta,
        eta=p.eta,
        zeta=p.zeta,
        alpha_min=p.alpha_min,
        alpha_max=p.alpha_max,
        lambda0=p.lambda0,
        gamma=p.gamma,
        c=p.c,
        zeta1=p.zeta1,
        zeta2=p.zeta2,
        eps=ABLATION_EPS,
        maxiter=ABLATION_MAXITER,
        converged=result.converged,
        status=string(result.status),
        restarts=result.restarts,
        iterations=result.iterations,
        f_evals=result.f_evals,
        residual=result.residual,
        cpu_time=result.cpu_time,
        tsel=p.tsel,
        srule=p.srule,
    )
end

_ablation_csv_text(::Missing) = ""
_ablation_csv_text(x::Bool) = lowercase(string(x))
_ablation_csv_text(x::Integer) = string(x)
_ablation_csv_text(x::AbstractFloat) = @sprintf("%.17g", x)
function _ablation_csv_text(x)
    s = string(x)
    (occursin(',', s) || occursin('\n', s) || occursin('\r', s)) &&
        throw(ArgumentError("Unescaped CSV text: $s"))
    return s
end

function _write_ablation_result(io::IO, row::NamedTuple)
    println(io, join(
        (_ablation_csv_text(getproperty(row, Symbol(name)))
         for name in ABLATION_COLUMNS), ','))
    flush(io)
    return nothing
end

function _ablation_require(condition::Bool, message::AbstractString)
    condition || throw(ArgumentError(message))
    return nothing
end

function _same_ablation_parameter(actual, expected)
    if ismissing(expected)
        return ismissing(actual)
    end
    return !ismissing(actual) && isfinite(Float64(actual)) &&
           Float64(actual) == expected
end

function validate_ablation_results(df::DataFrame, configs;
                                   require_complete::Bool=true)
    _ablation_require(names(df) == ABLATION_COLUMNS,
                      "Ablation CSV schema mismatch")
    config_map = Dict((cfg.method_name, cfg.config_label) => cfg
                      for cfg in configs)
    expected_instances = Set(_ablation_expected_instances())
    seen = Set{Tuple{String,String,Int,Int,String}}()

    for row in eachrow(df)
        method_name = String(row.method)
        label = String(row.config_label)
        config_key = (method_name, label)
        _ablation_require(haskey(config_map, config_key),
                          "Unknown configuration: $config_key")
        cfg = config_map[config_key]
        _ablation_require(String(row.param_family) == cfg.param_family,
                          "Parameter-family mismatch for $config_key")
        _ablation_require(Int(row.n) == ABLATION_N,
                          "Unexpected dimension for $config_key")

        problem_id = Int(row.problem_id)
        x0_label = String(row.x0_label)
        _ablation_require((problem_id, x0_label) in expected_instances,
                          "Unexpected instance: ($problem_id, $x0_label)")
        _ablation_require(
            String(row.problem) == get_problem(problem_id, ABLATION_N).name,
            "Problem label mismatch for internal id $problem_id")

        key = (method_name, label, problem_id, Int(row.n), x0_label)
        _ablation_require(!(key in seen), "Duplicate ablation key: $key")
        push!(seen, key)

        parameters = _ablation_parameters(cfg)
        for name in (
            :tau, :rho, :beta, :eta, :zeta, :alpha_min, :alpha_max,
            :lambda0, :gamma, :c, :zeta1, :zeta2,
        )
            _ablation_require(
                _same_ablation_parameter(getproperty(row, name),
                                         getproperty(parameters, name)),
                "Parameter mismatch in $name for $config_key")
        end
        expected_tsel = parameters.tsel
        actual_tsel = ismissing(row.tsel) ? missing : String(row.tsel)
        _ablation_require(isequal(actual_tsel, expected_tsel),
                          "tsel mismatch for $config_key")
        _ablation_require(String(row.srule) == parameters.srule,
                          "srule mismatch for $config_key")
        _ablation_require(Float64(row.eps) == ABLATION_EPS,
                          "Tolerance mismatch for $key")
        _ablation_require(Int(row.maxiter) == ABLATION_MAXITER,
                          "Iteration-cap mismatch for $key")

        restarts = Int(row.restarts)
        iterations = Int(row.iterations)
        f_evals = Int(row.f_evals)
        _ablation_require(restarts >= 0 && iterations >= 0 && f_evals >= 0,
                          "Negative counter for $key")
        cpu_time = Float64(row.cpu_time)
        _ablation_require(isfinite(cpu_time) && cpu_time > 0.0,
                          "Nonpositive/nonfinite elapsed time for $key")

        status = String(row.status)
        residual = Float64(row.residual)
        converged = Bool(row.converged)
        if isfinite(residual)
            _ablation_require(residual >= 0.0, "Negative residual for $key")
        else
            _ablation_require(status in ABLATION_NONFINITE_STATUSES,
                              "Unexpected nonfinite residual for $key")
        end
        if converged
            _ablation_require(startswith(status, "converged"),
                              "Converged flag/status mismatch for $key")
            _ablation_require(isfinite(residual) && residual <= ABLATION_EPS,
                              "Converged residual exceeds tolerance for $key")
        else
            _ablation_require(!startswith(status, "converged"),
                              "Failure flag/status mismatch for $key")
        end
    end

    expected_keys = Set(
        ablation_run_key(cfg, problem_id, ABLATION_N, x0_label)
        for cfg in configs
        for (problem_id, x0_label) in expected_instances)
    _ablation_require(issubset(seen, expected_keys),
                      "Ablation CSV contains keys outside the protocol")
    if require_complete
        _ablation_require(length(seen) == 1_620,
                          "Expected 1,620 rows, found $(length(seen))")
        _ablation_require(seen == expected_keys,
                          "Ablation CSV does not have exact protocol coverage")
        for cfg in configs
            count_cfg = count(
                key -> key[1] == cfg.method_name &&
                       key[2] == cfg.config_label,
                seen)
            _ablation_require(count_cfg == 180,
                "$(cfg.method_name) $(cfg.config_label) has $count_cfg rows, expected 180")
        end
    end
    return seen
end

function ablation_configuration_summary(df::DataFrame)
    rows = NamedTuple[]
    for sub in groupby(df, [:method, :config_label])
        converged_rows = filter(:converged => identity, sub)
        total = nrow(sub)
        converged = nrow(converged_rows)
        push!(rows, (
            method=String(sub.method[1]),
            config_label=String(sub.config_label[1]),
            total=total,
            converged=converged,
            rate_pct=100.0 * converged / total,
            median_iterations=converged > 0 ?
                median(converged_rows.iterations) : missing,
            median_f_evals=converged > 0 ?
                median(converged_rows.f_evals) : missing,
            median_cpu=converged > 0 ?
                median(converged_rows.cpu_time) : missing,
            restarts=sum(Int.(sub.restarts)),
        ))
    end
    return DataFrame(rows)
end

_same_float_bits(a, b) =
    reinterpret(UInt64, Float64(a)) == reinterpret(UInt64, Float64(b))

function identity_differences(df::DataFrame, method_name::String,
                              baseline_label::String, variant_label::String)
    baseline = filter(
        row -> row.method == method_name && row.config_label == baseline_label,
        df)
    variant = filter(
        row -> row.method == method_name && row.config_label == variant_label,
        df)
    baseline_map = Dict(
        (Int(row.problem_id), Int(row.n), String(row.x0_label)) => row
        for row in eachrow(baseline))
    variant_map = Dict(
        (Int(row.problem_id), Int(row.n), String(row.x0_label)) => row
        for row in eachrow(variant))
    _ablation_require(keys(baseline_map) == keys(variant_map),
                      "Identity configurations do not cover the same instances")

    rows = NamedTuple[]
    for key in sort!(collect(keys(baseline_map)))
        base = baseline_map[key]
        other = variant_map[key]
        differs = Int(base.iterations) != Int(other.iterations) ||
                  Int(base.f_evals) != Int(other.f_evals) ||
                  !_same_float_bits(base.residual, other.residual)
        differs || continue
        push!(rows, (
            problem_id=key[1],
            problem=String(base.problem),
            n=key[2],
            x0_label=key[3],
            baseline_iterations=Int(base.iterations),
            variant_iterations=Int(other.iterations),
            baseline_f_evals=Int(base.f_evals),
            variant_f_evals=Int(other.f_evals),
            baseline_residual=Float64(base.residual),
            variant_residual=Float64(other.residual),
        ))
    end
    return DataFrame(rows)
end

function failure_status_summary(df::DataFrame)
    failures = filter(:converged => !, df)
    if nrow(failures) == 0
        return DataFrame(
            method=String[], config_label=String[], status=String[], count=Int[])
    end
    summary = combine(
        groupby(failures, [:method, :config_label, :status]), nrow => :count)
    sort!(summary, [:method, :config_label, :status])
    return summary
end

_ablation_metric_text(x; digits::Int=1) =
    ismissing(x) ? "---" : @sprintf("%.*f", digits, Float64(x))

function print_ablation_summary(io, df::DataFrame)
    summary = ablation_configuration_summary(df)
    config_order = Dict(
        (cfg.method_name, cfg.config_label) => index
        for (index, cfg) in enumerate(build_ablation_design()))
    summary.order = [config_order[(String(row.method), String(row.config_label))]
                     for row in eachrow(summary)]
    sort!(summary, :order)

    println(io, "=" ^ 103)
    println(io, "Parameter-choice ablation summary (medians over converged rows)")
    println(io, "-" ^ 103)
    @tprintf(io, "%-6s | %-12s | %5s | %5s | %7s | %8s | %8s | %10s | %8s\n",
             "method", "config", "total", "conv", "rate%", "med_IT",
             "med_FE", "med_CPU", "restarts")
    println(io, "-" ^ 103)
    for row in eachrow(summary)
        @tprintf(io, "%-6s | %-12s | %5d | %5d | %7.2f | %8s | %8s | %10s | %8d\n",
                 row.method, row.config_label, row.total, row.converged,
                 row.rate_pct, _ablation_metric_text(row.median_iterations),
                 _ablation_metric_text(row.median_f_evals),
                 _ablation_metric_text(row.median_cpu; digits=6), row.restarts)
    end
    select!(summary, Not(:order))
    println(io, "=" ^ 103)

    for (method_name, baseline, variant) in (
        ("SOPP", "default", "srule=bb1"),
        ("SDLP", "default", "srule=r1"),
    )
        differences = identity_differences(
            df, method_name, baseline, variant)
        @tprintf(io, "%s identity check (%s vs %s): %d/180 differing instances\n",
                 method_name, variant, baseline, nrow(differences))
        if nrow(differences) > 0
            for row in eachrow(differences)
                @tprintf(io,
                    "  %s x0=%s: IT %d/%d, FE %d/%d, residual %.17g/%.17g\n",
                    row.problem, row.x0_label,
                    row.baseline_iterations, row.variant_iterations,
                    row.baseline_f_evals, row.variant_f_evals,
                    row.baseline_residual, row.variant_residual)
            end
        end
    end

    failures = failure_status_summary(df)
    println(io, "Failure statuses")
    if nrow(failures) == 0
        println(io, "  none")
    else
        for row in eachrow(failures)
            @tprintf(io, "  %-6s %-12s %-24s %d\n",
                     row.method, row.config_label, row.status, row.count)
        end
    end
    return summary, failures
end

function parse_ablation_args(args)
    resume = false
    summary = false
    for arg in args
        if arg == "--resume"
            resume = true
        elseif arg == "--summary"
            summary = true
        else
            throw(ArgumentError("Unknown argument: $arg"))
        end
    end
    resume && summary && throw(ArgumentError(
        "--resume and --summary cannot be used together"))
    return (resume=resume, summary=summary)
end

function _open_ablation_results(raw_csv::String, resume::Bool, configs, tee)
    completed = Set{Tuple{String,String,Int,Int,String}}()
    append_mode = resume && isfile(raw_csv)
    if append_mode
        df = CSV.read(raw_csv, DataFrame)
        completed = validate_ablation_results(
            df, configs; require_complete=false)
        @tprintf(tee, "Resume: %d valid rows already complete\n", length(completed))
    elseif isfile(raw_csv)
        backup_dir = joinpath(dirname(raw_csv), "backup")
        mkpath(backup_dir)
        timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
        backup_path = joinpath(backup_dir, "raw_$timestamp.csv")
        cp(raw_csv, backup_path)
        println(tee, "Backup: $backup_path")
    end

    io = open(raw_csv, append_mode ? "a" : "w")
    if !append_mode || filesize(raw_csv) == 0
        println(io, join(ABLATION_COLUMNS, ','))
        flush(io)
    end
    return io, completed
end

function run_ablation!(raw_csv::String, configs, resume::Bool, tee)
    Threads.nthreads() == 1 || error(
        "Ablation study requires exactly one Julia thread; use --threads=1")
    BLAS.set_num_threads(1)

    io, completed = _open_ablation_results(raw_csv, resume, configs, tee)
    initial_points = get_initial_points(ABLATION_N)
    total = length(configs) * length(PROBLEM_IDS) * length(initial_points)
    remaining = total - length(completed)
    _ablation_require(remaining >= 0,
                      "Completed-key count exceeds protocol size")

    println(tee, "=" ^ 80)
    println(tee, "SOPP/SDLP parameter-choice ablation study")
    @tprintf(tee, "  configurations: %d\n", length(configs))
    @tprintf(tee, "  instances/configuration: %d problems x %d starts = 180\n",
             length(PROBLEM_IDS), length(initial_points))
    @tprintf(tee, "  n=%d, eps=%.0e, maxiter=%d\n",
             ABLATION_N, ABLATION_EPS, ABLATION_MAXITER)
    @tprintf(tee, "  rows: total=%d, complete=%d, remaining=%d\n",
             total, length(completed), remaining)
    @tprintf(tee, "  threads: Julia=%d, BLAS=%d\n",
             Threads.nthreads(), BLAS.get_num_threads())
    println(tee, "=" ^ 80)

    progress = Progress(remaining; barlen=40, showspeed=true,
                        desc="  Ablation: ")
    done = 0
    t0 = time_ns()
    try
        for cfg in configs
            for problem_id in PROBLEM_IDS
                problem = get_problem(problem_id, ABLATION_N)
                for (x0, x0_label) in initial_points
                    key = ablation_run_key(
                        cfg, problem_id, ABLATION_N, x0_label)
                    key in completed && continue

                    result = solve(
                        cfg.method, problem, x0;
                        eps=ABLATION_EPS,
                        maxiter=ABLATION_MAXITER,
                    )
                    _write_ablation_result(io, _ablation_result_row(
                        cfg, problem, ABLATION_N, x0_label, result))

                    done += 1
                    ProgressMeter.update!(progress, done;
                        showvalues=[
                            (:done, "$done/$remaining"),
                            (:config, "$(cfg.method_name) $(cfg.config_label)"),
                            (:instance, "$(problem.name) x0=$x0_label"),
                        ])
                end
            end
        end
    finally
        close(io)
        remaining > 0 && finish!(progress)
    end

    elapsed = _elapsed_seconds(t0, time_ns())
    @tprintf(tee, "Run wall time: %.6f s (%.3f min)\n", elapsed, elapsed / 60)
    df = CSV.read(raw_csv, DataFrame)
    validate_ablation_results(df, configs; require_complete=true)
    println(tee, "Validated exact coverage: 1,620 unique rows; 180/configuration")
    print_ablation_summary(tee, df)
    println(tee, "Results: $raw_csv")
    return df, elapsed
end

function main()
    args = parse_ablation_args(ARGS)
    configs = build_ablation_design()
    validate_ablation_configs!(configs)

    results_dir = joinpath(JCODE_ROOT, "results", "ablation")
    mkpath(results_dir)
    raw_csv = joinpath(results_dir, "raw.csv")
    logpath, tee, logfile = setup_logging("ablation")
    try
        if args.summary
            isfile(raw_csv) || error("Ablation CSV not found: $raw_csv")
            df = CSV.read(raw_csv, DataFrame)
            validate_ablation_results(df, configs; require_complete=true)
            println(tee, "Validated exact coverage: 1,620 unique rows; 180/configuration")
            print_ablation_summary(tee, df)
            println(tee, "Results: $raw_csv")
        else
            run_ablation!(raw_csv, configs, args.resume, tee)
        end
    finally
        teardown_logging(tee, logpath)
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
