# ============================================================================
# s60: SOPP/SDLP One-at-a-Time Parameter Sensitivity Study
# ============================================================================
#
# Protocol:
#   30 unique configurations x 18 problems x 10 initial points at n=10,000.
#   Each method's default is solved once and reused in all five family bands.
#
# Usage (from jcode/):
#   julia --threads=1 --project=. scripts/s60_sensitivity.jl
#   julia --threads=1 --project=. scripts/s60_sensitivity.jl --resume
#   julia --threads=1 --project=. scripts/s60_sensitivity.jl --summary
#
# Output:
#   results/sensitivity/raw.csv
#   results/logs/sensitivity_TIMESTAMP.log
# ============================================================================

if !isdefined(Main, :SOPPMethod)
    include(joinpath(@__DIR__, "..", "src", "includes.jl"))
end
using CSV, DataFrames

const SENSITIVITY_N = 10_000
const SENSITIVITY_EPS = 1e-11
const SENSITIVITY_MAXITER = 2000
const SENSITIVITY_METHOD_ORDER = ["SOPP", "SDLP"]
const SENSITIVITY_FAMILY_ORDER = Dict(
    "SOPP" => ["tau", "c", "alpha", "rho", "zeta"],
    "SDLP" => ["tau", "alpha", "rho", "eta", "zeta"],
)
const SENSITIVITY_COLUMNS = [
    "method", "config_label", "param_family", "problem_id", "problem", "n",
    "x0_label", "tau", "rho", "beta", "eta", "zeta", "alpha_min",
    "alpha_max", "lambda0", "gamma", "c", "zeta1", "zeta2", "eps",
    "maxiter", "converged", "status", "restarts", "iterations", "f_evals",
    "residual", "cpu_time",
]
const SENSITIVITY_NONFINITE_STATUSES = Set([
    "nonfinite", "invalid_trial_residual",
])

const SOPP_SENSITIVITY_DEFAULTS = (
    tau=1.0, rho=0.8, beta=0.5, zeta=1e-4,
    alpha_min=0.1, alpha_max=2.0, lambda0=1.0, gamma=1.1,
    c=3.0, zeta1=1.0, zeta2=1.0,
)
const SDLP_SENSITIVITY_DEFAULTS = (
    tau=0.001, rho=0.5, eta=0.6, zeta=0.1,
    alpha_min=0.55, alpha_max=4.9, lambda0=1.0, gamma=1.8,
    zeta1=1.0, zeta2=1.0,
)

struct SensitivityConfig
    method_name::String
    param_family::String
    config_label::String
    method::AbstractMethod
end

Base.:(==)(a::SensitivityConfig, b::SensitivityConfig) =
    a.method_name == b.method_name &&
    a.param_family == b.param_family &&
    a.config_label == b.config_label &&
    canonical_config_key(a) == canonical_config_key(b)

_sopp(; kwargs...) = SOPPMethod(; merge(
    SOPP_SENSITIVITY_DEFAULTS, (; kwargs...))...)
_sdlp(; kwargs...) = SDLPMethod(; merge(
    SDLP_SENSITIVITY_DEFAULTS, (; kwargs...))...)

function _parameter_text(x::Real)
    s = @sprintf("%.10g", Float64(x))
    s = replace(s, "e-0" => "e-", "e+0" => "e+")
    return s
end

_value_label(name::String, x::Real) = "$(name)=$(_parameter_text(x))"
_alpha_label(amin::Real, amax::Real) =
    "alpha_min=$(_parameter_text(amin));alpha_max=$(_parameter_text(amax))"

function canonical_config_key(cfg::SensitivityConfig)
    m = cfg.method
    if m isa SOPPMethod
        return (
            "SOPP", m.tau, m.rho, m.beta, m.zeta, m.alpha_min, m.alpha_max,
            m.lambda0, m.gamma, m.c, m.zeta1, m.zeta2,
        )
    elseif m isa SDLPMethod
        return (
            "SDLP", m.tau, m.rho, m.eta, m.zeta, m.alpha_min, m.alpha_max,
            m.lambda0, m.gamma, m.zeta1, m.zeta2,
        )
    end
    throw(ArgumentError("Unsupported sensitivity method: $(typeof(m))"))
end

function _add_family!(
        candidates::Vector{SensitivityConfig},
        memberships::Dict{Tuple{String,String},Vector{String}},
        ranges::Dict{Tuple{String,String},String},
        method_name::String, family::String, settings, range_text::String,
        default_key)
    labels = String[]
    for (label, method) in settings
        candidate = SensitivityConfig(method_name, family, label, method)
        if canonical_config_key(candidate) == default_key
            candidate = SensitivityConfig(
                method_name, "shared_default", "default", method)
        end
        push!(candidates, candidate)
        push!(labels, candidate.config_label)
    end
    memberships[(method_name, family)] = unique(labels)
    ranges[(method_name, family)] = range_text
    return nothing
end

function build_candidate_configs()
    candidates = SensitivityConfig[]
    memberships = Dict{Tuple{String,String},Vector{String}}()
    ranges = Dict{Tuple{String,String},String}()

    sopp_default = SensitivityConfig(
        "SOPP", "shared_default", "default", _sopp())
    sopp_default_key = canonical_config_key(sopp_default)

    sopp_tau = [0.1, 0.5, 1.0, 2.0, 10.0]
    _add_family!(candidates, memberships, ranges, "SOPP", "tau",
        [(_value_label("tau", v), _sopp(tau=v)) for v in sopp_tau],
        join(_parameter_text.(sopp_tau), ", "), sopp_default_key)

    sopp_c = [2.5, 3.0, 5.0, 10.0]
    _add_family!(candidates, memberships, ranges, "SOPP", "c",
        [(_value_label("c", v), _sopp(c=v)) for v in sopp_c],
        join(_parameter_text.(sopp_c), ", "), sopp_default_key)

    # The originally proposed (0.1, 5.0) point is illegal with fixed c=3.0.
    # Using alpha_max=2.5 preserves both c > alpha_max and OAT semantics.
    sopp_alpha = [(0.01, 2.0), (0.1, 2.0), (0.1, 2.5), (0.5, 2.0)]
    _add_family!(candidates, memberships, ranges, "SOPP", "alpha",
        [(_alpha_label(amin, amax),
          _sopp(alpha_min=amin, alpha_max=amax, c=3.0))
         for (amin, amax) in sopp_alpha],
        join(["($(_parameter_text(a)),$(_parameter_text(b)))"
              for (a, b) in sopp_alpha], ", "), sopp_default_key)

    sopp_rho = [0.5, 0.8, 0.9]
    _add_family!(candidates, memberships, ranges, "SOPP", "rho",
        [(_value_label("rho", v), _sopp(rho=v)) for v in sopp_rho],
        join(_parameter_text.(sopp_rho), ", "), sopp_default_key)

    sopp_zeta = [1e-5, 1e-4, 1e-2]
    _add_family!(candidates, memberships, ranges, "SOPP", "zeta",
        [(_value_label("zeta", v), _sopp(zeta=v)) for v in sopp_zeta],
        join(_parameter_text.(sopp_zeta), ", "), sopp_default_key)

    sdlp_default = SensitivityConfig(
        "SDLP", "shared_default", "default", _sdlp())
    sdlp_default_key = canonical_config_key(sdlp_default)

    sdlp_tau = [0.0, 0.001, 0.01, 0.05, 0.09]
    _add_family!(candidates, memberships, ranges, "SDLP", "tau",
        [(_value_label("tau", v), _sdlp(tau=v)) for v in sdlp_tau],
        join(_parameter_text.(sdlp_tau), ", "), sdlp_default_key)

    sdlp_alpha = [
        (0.51, 4.9), (0.55, 4.9), (1.0, 4.9), (0.55, 2.0), (0.55, 10.0),
    ]
    _add_family!(candidates, memberships, ranges, "SDLP", "alpha",
        [(_alpha_label(amin, amax),
          _sdlp(alpha_min=amin, alpha_max=amax))
         for (amin, amax) in sdlp_alpha],
        join(["($(_parameter_text(a)),$(_parameter_text(b)))"
              for (a, b) in sdlp_alpha], ", "), sdlp_default_key)

    sdlp_rho = [0.3, 0.5, 0.8]
    _add_family!(candidates, memberships, ranges, "SDLP", "rho",
        [(_value_label("rho", v), _sdlp(rho=v)) for v in sdlp_rho],
        join(_parameter_text.(sdlp_rho), ", "), sdlp_default_key)

    sdlp_eta = [0.3, 0.6, 1.0]
    _add_family!(candidates, memberships, ranges, "SDLP", "eta",
        [(_value_label("eta", v), _sdlp(eta=v)) for v in sdlp_eta],
        join(_parameter_text.(sdlp_eta), ", "), sdlp_default_key)

    sdlp_zeta = [0.01, 0.1, 0.5]
    _add_family!(candidates, memberships, ranges, "SDLP", "zeta",
        [(_value_label("zeta", v), _sdlp(zeta=v)) for v in sdlp_zeta],
        join(_parameter_text.(sdlp_zeta), ", "), sdlp_default_key)

    return candidates, memberships, ranges
end

function deduplicate_configs(candidates::Vector{SensitivityConfig})
    configs = SensitivityConfig[]
    seen = Dict{Any,SensitivityConfig}()
    for cfg in candidates
        key = canonical_config_key(cfg)
        if haskey(seen, key)
            prior = seen[key]
            prior.config_label == cfg.config_label || throw(ArgumentError(
                "One parameter tuple has conflicting labels: " *
                "$(prior.config_label) and $(cfg.config_label)"))
            prior.param_family == cfg.param_family || throw(ArgumentError(
                "A duplicate tuple is not marked as a shared default"))
        else
            seen[key] = cfg
            push!(configs, cfg)
        end
    end
    return configs
end

function build_sensitivity_design()
    candidates, memberships, ranges = build_candidate_configs()
    configs = deduplicate_configs(candidates)
    count(c -> c.method_name == "SOPP", candidates) == 19 ||
        error("SOPP must have 19 family memberships")
    count(c -> c.method_name == "SDLP", candidates) == 19 ||
        error("SDLP must have 19 family memberships")
    count(c -> c.method_name == "SOPP", configs) == 15 ||
        error("SOPP must have 15 unique configurations")
    count(c -> c.method_name == "SDLP", configs) == 15 ||
        error("SDLP must have 15 unique configurations")
    sum(length, values(memberships)) == 38 ||
        error("Sensitivity family membership count must be 38")
    return (configs=configs, family_members=memberships, ranges=ranges)
end

function validate_config!(cfg::SensitivityConfig)
    isempty(cfg.config_label) && throw(ArgumentError("Empty configuration label"))
    (occursin(',', cfg.config_label) || occursin('\n', cfg.config_label)) &&
        throw(ArgumentError("Configuration labels must be CSV-safe"))
    m = cfg.method
    if m isa SOPPMethod
        @assert cfg.method_name == "SOPP"
        @assert all(isfinite, (
            m.tau, m.rho, m.beta, m.zeta, m.alpha_min, m.alpha_max,
            m.lambda0, m.gamma, m.c, m.zeta1, m.zeta2,
        ))
        SOPPMethod(
            tau=m.tau, rho=m.rho, beta=m.beta, zeta=m.zeta,
            alpha_min=m.alpha_min, alpha_max=m.alpha_max,
            lambda0=m.lambda0, gamma=m.gamma, c=m.c,
            zeta1=m.zeta1, zeta2=m.zeta2,
        )
    elseif m isa SDLPMethod
        @assert cfg.method_name == "SDLP"
        @assert all(isfinite, (
            m.tau, m.rho, m.eta, m.zeta, m.alpha_min, m.alpha_max,
            m.lambda0, m.gamma, m.zeta1, m.zeta2,
        ))
        SDLPMethod(
            tau=m.tau, rho=m.rho, eta=m.eta, zeta=m.zeta,
            alpha_min=m.alpha_min, alpha_max=m.alpha_max,
            lambda0=m.lambda0, gamma=m.gamma,
            zeta1=m.zeta1, zeta2=m.zeta2,
        )
    else
        throw(ArgumentError("Unsupported sensitivity method: $(typeof(m))"))
    end
    return cfg
end

function validate_configs!(configs::Vector{SensitivityConfig})
    foreach(validate_config!, configs)
    length(configs) == 30 || throw(ArgumentError(
        "Expected 30 unique configurations, found $(length(configs))"))
    keys = canonical_config_key.(configs)
    length(unique(keys)) == length(keys) ||
        throw(ArgumentError("Duplicate constructor tuples remain"))
    labels = [(c.method_name, c.config_label) for c in configs]
    length(unique(labels)) == length(labels) ||
        throw(ArgumentError("Configuration labels are not unique per method"))
    count(c -> c.config_label == "default", configs) == 2 ||
        throw(ArgumentError("Each method must store its default exactly once"))
    return configs
end

run_key(cfg::SensitivityConfig, problem_id::Integer, n::Integer,
        x0_label::AbstractString) =
    (cfg.method_name, cfg.config_label, Int(problem_id), Int(n), String(x0_label))

function expected_instance_keys(n::Int=SENSITIVITY_N)
    labels = last.(get_initial_points(n))
    return [(pid, label) for pid in PROBLEM_IDS for label in labels]
end

function _config_parameters(cfg::SensitivityConfig)
    m = cfg.method
    if m isa SOPPMethod
        return (
            tau=m.tau, rho=m.rho, beta=m.beta, eta=missing, zeta=m.zeta,
            alpha_min=m.alpha_min, alpha_max=m.alpha_max,
            lambda0=m.lambda0, gamma=m.gamma, c=m.c,
            zeta1=m.zeta1, zeta2=m.zeta2,
        )
    end
    return (
        tau=m.tau, rho=m.rho, beta=missing, eta=m.eta, zeta=m.zeta,
        alpha_min=m.alpha_min, alpha_max=m.alpha_max,
        lambda0=m.lambda0, gamma=m.gamma, c=missing,
        zeta1=m.zeta1, zeta2=m.zeta2,
    )
end

function _result_row(cfg::SensitivityConfig, prob::TestProblem, n::Int,
                     x0_label::String, result::SolverResult)
    p = _config_parameters(cfg)
    return (
        method=cfg.method_name,
        config_label=cfg.config_label,
        param_family=cfg.param_family,
        problem_id=prob.id,
        problem=prob.name,
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
        eps=SENSITIVITY_EPS,
        maxiter=SENSITIVITY_MAXITER,
        converged=result.converged,
        status=string(result.status),
        restarts=result.restarts,
        iterations=result.iterations,
        f_evals=result.f_evals,
        residual=result.residual,
        cpu_time=result.cpu_time,
    )
end

_csv_text(::Missing) = ""
_csv_text(x::Bool) = lowercase(string(x))
_csv_text(x::Integer) = string(x)
_csv_text(x::AbstractFloat) = @sprintf("%.17g", x)
function _csv_text(x)
    s = string(x)
    (occursin(',', s) || occursin('\n', s) || occursin('\r', s)) &&
        throw(ArgumentError("Unescaped CSV text: $s"))
    return s
end

function _write_result(io::IO, row::NamedTuple)
    println(io, join(
        (_csv_text(getproperty(row, Symbol(name)))
         for name in SENSITIVITY_COLUMNS), ','))
    flush(io)
    return nothing
end

function _require(condition::Bool, message::AbstractString)
    condition || throw(ArgumentError(message))
    return nothing
end

function _same_parameter(actual, expected)
    if ismissing(expected)
        return ismissing(actual)
    end
    return !ismissing(actual) && isfinite(Float64(actual)) &&
           Float64(actual) == expected
end

function validate_results(df::DataFrame, design; require_complete::Bool=true)
    _require(names(df) == SENSITIVITY_COLUMNS,
             "Sensitivity CSV schema mismatch")

    configs = Dict((c.method_name, c.config_label) => c
                   for c in design.configs)
    expected_instances = Set(expected_instance_keys())
    seen = Set{Tuple{String,String,Int,Int,String}}()

    for row in eachrow(df)
        method = String(row.method)
        label = String(row.config_label)
        key2 = (method, label)
        _require(haskey(configs, key2),
                 "Unknown configuration label: $key2")
        cfg = configs[key2]
        _require(String(row.param_family) == cfg.param_family,
                 "Parameter-family mismatch for $key2")
        _require(Int(row.n) == SENSITIVITY_N,
                 "Unexpected dimension for $key2")
        pid = Int(row.problem_id)
        x0_label = String(row.x0_label)
        _require((pid, x0_label) in expected_instances,
                 "Unexpected instance ($pid, $x0_label)")
        _require(String(row.problem) == get_problem(pid, SENSITIVITY_N).name,
                 "Problem label mismatch for internal id $pid")

        key = (method, label, pid, Int(row.n), x0_label)
        _require(!(key in seen), "Duplicate sensitivity key: $key")
        push!(seen, key)

        params = _config_parameters(cfg)
        for name in propertynames(params)
            _require(_same_parameter(getproperty(row, name),
                                     getproperty(params, name)),
                     "Parameter mismatch in $name for $key2")
        end
        _require(isfinite(Float64(row.eps)) &&
                 Float64(row.eps) == SENSITIVITY_EPS,
                 "Tolerance mismatch for $key")
        _require(Int(row.maxiter) == SENSITIVITY_MAXITER,
                 "Iteration-cap mismatch for $key")

        restarts = Int(row.restarts)
        iterations = Int(row.iterations)
        f_evals = Int(row.f_evals)
        _require(restarts >= 0 && iterations >= 0 && f_evals >= 0,
                 "Negative counter for $key")
        cpu = Float64(row.cpu_time)
        _require(isfinite(cpu) && cpu > 0.0,
                 "Nonpositive/nonfinite elapsed time for $key")

        status = String(row.status)
        residual = Float64(row.residual)
        converged = Bool(row.converged)
        if isfinite(residual)
            _require(residual >= 0.0, "Negative residual for $key")
        else
            _require(status in SENSITIVITY_NONFINITE_STATUSES,
                     "Unexpected nonfinite residual for $key with status=$status")
        end
        if converged
            _require(startswith(status, "converged"),
                     "Converged flag/status mismatch for $key")
            _require(isfinite(residual) && residual <= SENSITIVITY_EPS,
                     "Converged residual exceeds tolerance for $key")
        else
            _require(!startswith(status, "converged"),
                     "Failure flag/status mismatch for $key")
        end
    end

    expected_keys = Set(
        run_key(cfg, pid, SENSITIVITY_N, label)
        for cfg in design.configs for (pid, label) in expected_instances)
    _require(issubset(seen, expected_keys),
             "Sensitivity CSV contains keys outside the protocol")
    if require_complete
        _require(length(seen) == 5_400,
                 "Expected 5,400 rows, found $(length(seen))")
        _require(seen == expected_keys,
                 "Sensitivity CSV does not have exact protocol coverage")
        for cfg in design.configs
            count_cfg = count(k -> k[1] == cfg.method_name &&
                                  k[2] == cfg.config_label, seen)
            _require(count_cfg == 180,
                     "$(cfg.method_name) $(cfg.config_label) has " *
                     "$count_cfg rows, expected 180")
        end
    end
    return seen
end

function configuration_summary(df::DataFrame)
    rows = NamedTuple[]
    for sub in groupby(df, [:method, :config_label])
        conv = filter(:converged => identity, sub)
        n_total = nrow(sub)
        n_conv = nrow(conv)
        family = "param_family" in names(sub) ?
            only(unique(String.(sub.param_family))) : ""
        push!(rows, (
            method=String(sub.method[1]),
            param_family=family,
            config_label=String(sub.config_label[1]),
            total=n_total,
            converged=n_conv,
            rate_pct=100.0 * n_conv / n_total,
            median_iterations=n_conv > 0 ? median(conv.iterations) : missing,
            median_f_evals=n_conv > 0 ? median(conv.f_evals) : missing,
            median_cpu=n_conv > 0 ? median(conv.cpu_time) : missing,
        ))
    end
    return DataFrame(rows)
end

function family_band_summary(config_summary::DataFrame, design)
    rows = NamedTuple[]
    for method in SENSITIVITY_METHOD_ORDER
        for family in SENSITIVITY_FAMILY_ORDER[method]
            labels = design.family_members[(method, family)]
            sub = filter(
                r -> r.method == method && r.config_label in labels,
                config_summary)
            _require(nrow(sub) == length(labels),
                     "Incomplete summary membership for $method/$family")
            med_fe = sub.median_f_evals
            all_finite_fe = all(x -> !ismissing(x) && isfinite(Float64(x)), med_fe)
            push!(rows, (
                method=method,
                param_family=family,
                tested_range=design.ranges[(method, family)],
                n_configs=length(labels),
                rate_min_pct=minimum(sub.rate_pct),
                rate_max_pct=maximum(sub.rate_pct),
                median_f_evals_min=all_finite_fe ?
                    minimum(Float64.(med_fe)) : missing,
                median_f_evals_max=all_finite_fe ?
                    maximum(Float64.(med_fe)) : missing,
            ))
        end
    end
    return DataFrame(rows)
end

function _metric_text(x; digits::Int=1)
    ismissing(x) && return "---"
    return @sprintf("%.*f", digits, Float64(x))
end

function print_sensitivity_summary(io, df::DataFrame, design)
    config_summary = configuration_summary(df)
    bands = family_band_summary(config_summary, design)

    println(io, "=" ^ 118)
    println(io, "Per-configuration sensitivity summary (medians over converged rows)")
    println(io, "-" ^ 118)
    @tprintf(io, "%-6s | %-14s | %-39s | %5s | %5s | %7s | %8s | %8s | %9s\n",
             "method", "family", "config", "total", "conv", "rate%",
             "med_IT", "med_FE", "med_CPU")
    println(io, "-" ^ 118)
    for row in eachrow(config_summary)
        @tprintf(io, "%-6s | %-14s | %-39s | %5d | %5d | %7.2f | %8s | %8s | %9s\n",
                 row.method, row.param_family, row.config_label,
                 row.total, row.converged, row.rate_pct,
                 _metric_text(row.median_iterations),
                 _metric_text(row.median_f_evals),
                 _metric_text(row.median_cpu; digits=6))
    end

    println(io, "=" ^ 118)
    println(io, "Per-family sensitivity bands (default included in every family)")
    println(io, "-" ^ 118)
    @tprintf(io, "%-6s | %-7s | %-43s | %4s | %-17s | %-17s\n",
             "method", "family", "tested range", "n", "rate band (%)",
             "median-FE band")
    println(io, "-" ^ 118)
    for row in eachrow(bands)
        rate_band = @sprintf("%.2f--%.2f", row.rate_min_pct, row.rate_max_pct)
        fe_band = ismissing(row.median_f_evals_min) ? "---" :
            @sprintf("%.1f--%.1f", row.median_f_evals_min,
                     row.median_f_evals_max)
        @tprintf(io, "%-6s | %-7s | %-43s | %4d | %-17s | %-17s\n",
                 row.method, row.param_family, row.tested_range,
                 row.n_configs, rate_band, fe_band)
    end
    println(io, "=" ^ 118)
    return config_summary, bands
end

function parse_sensitivity_args(args)
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

function _open_results(raw_csv::String, resume::Bool, design, tee)
    completed = Set{Tuple{String,String,Int,Int,String}}()
    append_mode = resume && isfile(raw_csv)
    if append_mode
        df = CSV.read(raw_csv, DataFrame)
        completed = validate_results(df, design; require_complete=false)
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
        println(io, join(SENSITIVITY_COLUMNS, ','))
        flush(io)
    end
    return io, completed
end

function run_sensitivity!(raw_csv::String, design, resume::Bool, tee)
    Threads.nthreads() == 1 || error(
        "Sensitivity study requires exactly one Julia thread; use --threads=1")
    BLAS.set_num_threads(1)

    io, completed = _open_results(raw_csv, resume, design, tee)
    initial_points = get_initial_points(SENSITIVITY_N)
    total = length(design.configs) * length(PROBLEM_IDS) * length(initial_points)
    remaining = total - length(completed)
    _require(remaining >= 0, "Completed-key count exceeds protocol size")

    println(tee, "=" ^ 80)
    println(tee, "SOPP/SDLP OAT sensitivity study")
    @tprintf(tee, "  configurations: %d unique (19 memberships per method)\n",
             length(design.configs))
    @tprintf(tee, "  instances/configuration: %d problems x %d starts = 180\n",
             length(PROBLEM_IDS), length(initial_points))
    @tprintf(tee, "  n=%d, eps=%.0e, maxiter=%d\n",
             SENSITIVITY_N, SENSITIVITY_EPS, SENSITIVITY_MAXITER)
    @tprintf(tee, "  rows: total=%d, complete=%d, remaining=%d\n",
             total, length(completed), remaining)
    @tprintf(tee, "  threads: Julia=%d, BLAS=%d\n",
             Threads.nthreads(), BLAS.get_num_threads())
    println(tee, "=" ^ 80)

    progress = Progress(remaining; barlen=40, showspeed=true,
                        desc="  Sensitivity: ")
    done = 0
    t0 = time_ns()
    try
        for cfg in design.configs
            for pid in PROBLEM_IDS
                prob = get_problem(pid, SENSITIVITY_N)
                for (x0, x0_label) in initial_points
                    key = run_key(cfg, pid, SENSITIVITY_N, x0_label)
                    key in completed && continue

                    result = solve(
                        cfg.method, prob, x0;
                        eps=SENSITIVITY_EPS,
                        maxiter=SENSITIVITY_MAXITER,
                    )
                    _write_result(io, _result_row(
                        cfg, prob, SENSITIVITY_N, x0_label, result))

                    done += 1
                    ProgressMeter.update!(progress, done;
                        showvalues=[
                            (:done, "$done/$remaining"),
                            (:config, "$(cfg.method_name) $(cfg.config_label)"),
                            (:instance, "$(prob.name) x0=$x0_label"),
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
    validate_results(df, design; require_complete=true)
    println(tee, "Validated exact coverage: 5,400 unique rows; 180/configuration")
    print_sensitivity_summary(tee, df, design)
    println(tee, "Results: $raw_csv")
    return df, elapsed
end

function main()
    args = parse_sensitivity_args(ARGS)
    design = build_sensitivity_design()

    # Legality gate: validate the entire deduplicated grid before logging,
    # opening the CSV, or calling any solver.
    validate_configs!(design.configs)

    results_dir = joinpath(JCODE_ROOT, "results", "sensitivity")
    mkpath(results_dir)
    raw_csv = joinpath(results_dir, "raw.csv")
    logpath, tee, logfile = setup_logging("sensitivity")
    try
        if args.summary
            isfile(raw_csv) || error("Sensitivity CSV not found: $raw_csv")
            df = CSV.read(raw_csv, DataFrame)
            validate_results(df, design; require_complete=true)
            println(tee, "Validated exact coverage: 5,400 unique rows; 180/configuration")
            print_sensitivity_summary(tee, df, design)
            println(tee, "Results: $raw_csv")
        else
            run_sensitivity!(raw_csv, design, args.resume, tee)
        end
    finally
        teardown_logging(tee, logpath)
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
