# utils.jl — IO, logging

# ── Project root (jcode/) ──────────────────────────────────────────────────────

const JCODE_ROOT = normpath(joinpath(@__DIR__, ".."))

# ── TeeIO for dual logging ─────────────────────────────────────────────────────
# Not an IO subtype — avoids method ambiguities with Base.write(::IO, ::StridedArray).
# Use println(tee, ...) and @printf(tee, ...) via the forwarding methods below.

struct TeeIO
    out::IO
    log::IO
end

Base.println(t::TeeIO, xs...) = (println(t.out, xs...); println(t.log, xs...); flush(t.log))
Base.print(t::TeeIO, xs...)   = (print(t.out, xs...); print(t.log, xs...); flush(t.log))
Base.write(t::TeeIO, x)       = (write(t.out, x); write(t.log, x))
Base.flush(t::TeeIO)           = (flush(t.out); flush(t.log))

# Formatted printing to TeeIO. Usage: @tprintf tee "fmt" args...
# The fmt must be a string literal (same restriction as @printf).
macro tprintf(tee, fmt, args...)
    # fmt is already a string literal — pass it unescaped to @printf
    pf = esc(:(let buf = IOBuffer()
        @printf(buf, $fmt, $(args...))
        String(take!(buf))
    end))
    return :( print($(esc(tee)), $pf) )
end

function setup_logging(experiment_name::String)
    logdir = joinpath(JCODE_ROOT, "results", "logs")
    mkpath(logdir)
    timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
    logpath = joinpath(logdir, "$(experiment_name)_$(timestamp).log")
    logfile = open(logpath, "w")
    tee = TeeIO(stdout, logfile)
    return logpath, tee, logfile
end

function teardown_logging(tee::TeeIO, logpath::String)
    flush(tee)
    close(tee.log)
    println("Log saved to: $logpath")
end
