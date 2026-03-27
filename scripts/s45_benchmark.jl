# ============================================================================
# s45: Full Benchmark
# ============================================================================
#
# Usage:
#   cd jcode
#   julia --project=. scripts/s45_benchmark.jl --all
#   julia --project=. scripts/s45_benchmark.jl --all --resume
#   julia --project=. scripts/s45_benchmark.jl --problems=1-5,10 --dims=1000,10000
#   julia --project=. scripts/s45_benchmark.jl --methods=GMOPCGM,GCGPM
#   julia --project=. scripts/s45_benchmark.jl --inits=0,1.0,8.2
#   julia --project=. scripts/s45_benchmark.jl --summary
#   julia --project=. scripts/s45_benchmark.jl --problems=1 --verbose
# ============================================================================

include(joinpath(@__DIR__, "..", "src", "includes.jl"))
using CSV, DataFrames

# ── Constants ────────────────────────────────────────────────────────────────
const EPS     = 1e-11
const MAXITER = 2000

const ALL_METHODS = Dict(
    "GMOPCGM"  => GMOPCGMMethod(),
    "GCGPM"    => GCGPMMethod(),
    "MOPCGM"   => MOPCGMMethod(),
    "CGPM"     => CGPMMethod(),
    "STTDFPM"  => STTDFPMMethod(),
)
const METHOD_ORDER = ["GMOPCGM", "GCGPM", "MOPCGM", "CGPM", "STTDFPM"]

const RAW_COLS = ["method", "problem", "n", "x0_label", "converged",
                  "iterations", "f_evals", "residual", "cpu_time"]

# ── ARGS parsing ─────────────────────────────────────────────────────────────

function parse_int_list(s::AbstractString)
    ids = Int[]
    for part in split(s, ",")
        part = strip(part)
        if contains(part, "-")
            ab = split(part, "-")
            append!(ids, parse(Int, ab[1]):parse(Int, ab[2]))
        else
            push!(ids, parse(Int, part))
        end
    end
    return ids
end

function parse_args(args)
    cfg = Dict{Symbol,Any}(
        :problems  => copy(PROBLEM_IDS),
        :dims      => DIMENSIONS,
        :methods   => METHOD_ORDER,
        :inits     => nothing,
        :resume    => false,
        :summary   => false,
        :verbose   => false,
        :all       => false,
    )
    for a in args
        if a == "--all"
            cfg[:all] = true
        elseif a == "--resume"
            cfg[:resume] = true
        elseif a == "--summary"
            cfg[:summary] = true
        elseif a == "--verbose"
            cfg[:verbose] = true
        elseif startswith(a, "--problems=")
            cfg[:problems] = parse_int_list(split(a, "="; limit=2)[2])
        elseif startswith(a, "--dims=")
            cfg[:dims] = [parse(Int, strip(s)) for s in split(split(a, "="; limit=2)[2], ",")]
        elseif startswith(a, "--methods=")
            cfg[:methods] = [strip(s) for s in split(split(a, "="; limit=2)[2], ",")]
        elseif startswith(a, "--inits=")
            cfg[:inits] = [strip(s) for s in split(split(a, "="; limit=2)[2], ",")]
        end
    end
    if cfg[:all]
        cfg[:problems] = copy(PROBLEM_IDS)
        cfg[:dims] = DIMENSIONS
        cfg[:methods] = METHOD_ORDER
        cfg[:inits] = nothing
    end
    return cfg
end

cfg = parse_args(ARGS)

# ── Output setup ─────────────────────────────────────────────────────────────
results_dir = joinpath(JCODE_ROOT, "results", "benchmark")
mkpath(results_dir)
raw_csv = joinpath(results_dir, "raw.csv")

logpath, tee, logfile = setup_logging("benchmark")

function get_filtered_inits(n, filter_labels)
    all_inits = get_initial_points(n)
    filter_labels === nothing ? all_inits : [(x0, l) for (x0, l) in all_inits if l in filter_labels]
end

# ── Backup before writing ────────────────────────────────────────────────────
if !cfg[:summary] && isfile(raw_csv)
    backup_dir = joinpath(results_dir, "backup")
    mkpath(backup_dir)
    ts = Dates.format(now(), "yyyymmdd_HHMMss")
    backup_path = joinpath(backup_dir, "raw_$ts.csv")
    cp(raw_csv, backup_path)
    println(tee, "Backup: $backup_path")
end

# ── Summary mode ─────────────────────────────────────────────────────────────
if cfg[:summary]
    if !isfile(raw_csv)
        println(tee, "ERROR: raw CSV not found: $raw_csv")
        teardown_logging(tee, logpath)
        exit(1)
    end
    df = CSV.read(raw_csv, DataFrame)

    println(tee, "=" ^ 75)
    @tprintf(tee, "Benchmark Summary  (%d rows)\n", nrow(df))
    println(tee, "-" ^ 75)
    @tprintf(tee, "  %-10s %6s %6s %7s %8s %8s %10s\n",
             "Method", "Total", "Conv", "Rate", "med_IT", "med_FE", "med_CPU")
    println(tee, "-" ^ 75)
    for m in METHOD_ORDER
        sub = filter(r -> r.method == m, df)
        nrow(sub) == 0 && continue
        conv = filter(r -> r.converged, sub)
        n_t = nrow(sub); n_c = nrow(conv)
        @tprintf(tee, "  %-10s %6d %6d %6.1f%% %8.0f %8.0f %10.4f\n",
                 m, n_t, n_c, 100n_c/n_t,
                 n_c > 0 ? median(conv.iterations) : NaN,
                 n_c > 0 ? median(conv.f_evals) : NaN,
                 n_c > 0 ? median(conv.cpu_time) : NaN)
    end

    # Win counts
    println(tee, "-" ^ 75)
    println(tee, "Wins (fewest iterations among converged):")
    wins = Dict(m => 0 for m in METHOD_ORDER)
    for gdf in groupby(df, [:problem, :n, :x0_label])
        conv = filter(r -> r.converged, gdf)
        nrow(conv) == 0 && continue
        best = minimum(conv.iterations)
        for r in eachrow(conv)
            r.iterations == best && (wins[r.method] += 1)
        end
    end
    for m in METHOD_ORDER
        @tprintf(tee, "  %-10s %6d\n", m, wins[m])
    end
    println(tee, "=" ^ 75)
    teardown_logging(tee, logpath)
    exit(0)
end

# ── Build work list ──────────────────────────────────────────────────────────
selected_methods = [(name, ALL_METHODS[name]) for name in cfg[:methods] if haskey(ALL_METHODS, name)]

struct WorkItem
    mname::String
    method::AbstractMethod
    prob_id::Int
    dim::Int
    x0::Vector{Float64}
    x0_label::String
end

work = WorkItem[]
for dim in cfg[:dims]
    inits = get_filtered_inits(dim, cfg[:inits])
    for pid in cfg[:problems]
        for (x0, x0_label) in inits
            for (mname, method) in selected_methods
                push!(work, WorkItem(mname, method, pid, dim, x0, x0_label))
            end
        end
    end
end

# ── Resume: filter completed ─────────────────────────────────────────────────
if cfg[:resume] && isfile(raw_csv)
    completed = Set{NTuple{4,String}}()
    for line in eachline(raw_csv)
        startswith(line, "method") && continue
        parts = split(line, ",")
        length(parts) >= 4 && push!(completed, (parts[1], parts[2], parts[3], parts[4]))
    end
    before = length(work)
    filter!(w -> (w.mname, "P$(w.prob_id)", string(w.dim), w.x0_label) ∉ completed, work)
    println(tee, "Resume: $(before - length(work)) already done, $(length(work)) remaining")
end

# ── Print config ─────────────────────────────────────────────────────────────
println(tee, "=" ^ 75)
println(tee, "Benchmark")
@tprintf(tee, "  Problems: %s\n", join(cfg[:problems], ","))
@tprintf(tee, "  Dims:     %s\n", join(cfg[:dims], ","))
@tprintf(tee, "  Methods:  %s\n", join(cfg[:methods], ","))
@tprintf(tee, "  Runs:     %d\n", length(work))
println(tee, "=" ^ 75)

if isempty(work)
    println(tee, "Nothing to do.")
    teardown_logging(tee, logpath)
    exit(0)
end

# ── Open CSV ─────────────────────────────────────────────────────────────────
raw_io = open(raw_csv, cfg[:resume] && isfile(raw_csv) ? "a" : "w")
if !(cfg[:resume] && isfile(raw_csv) && filesize(raw_csv) > 0)
    println(raw_io, join(RAW_COLS, ","))
    flush(raw_io)
end

# ── Main loop with single progress bar ───────────────────────────────────────
t_total = time()
n_conv = Ref(0)
n_fail = Ref(0)

# Single progress bar for the entire run
prog = Progress(length(work); barlen=40, showspeed=true, desc="  Running: ")
cb = ProgressCallback(prog, "", MAXITER, Ref(0), length(work), n_conv, n_fail)

for (i, w) in enumerate(work)
    prob = get_problem(w.prob_id, w.dim)
    cb.label = "$(w.mname) $(prob.name) n=$(w.dim) x0=$(w.x0_label)"
    cb.maxiter = MAXITER

    result = try
        solve(w.method, prob, w.x0; eps=EPS, maxiter=MAXITER, cb=cb)
    catch e
        _pcb_done!(cb, false)
        SolverResult(false, 0, 0, NaN, 0.0, w.x0)
    end

    @printf(raw_io, "%s,%s,%d,%s,%s,%d,%d,%.10e,%.6f\n",
            w.mname, prob.name, w.dim, w.x0_label,
            result.converged, result.iterations, result.f_evals,
            result.residual, result.cpu_time)
    flush(raw_io)
end
finish!(prog)
close(raw_io)

# ── Final summary ────────────────────────────────────────────────────────────
elapsed = time() - t_total
println(tee, "")
println(tee, "-" ^ 75)
@tprintf(tee, "Done: %d runs in %.1fs (%.1f min)\n", length(work), elapsed, elapsed/60)
@tprintf(tee, "  Converged: %d   Failed: %d\n", n_conv[], n_fail[])
println(tee, "  Results:   $raw_csv")
println(tee, "-" ^ 75)
println(tee, "Run with --summary to see aggregate statistics.")

teardown_logging(tee, logpath)
