# ============================================================================
# s10: Smoke Test — verify all problems, projections, and solvers
# ============================================================================
#
# Usage:
#   cd jcode
#   julia --project=. scripts/s10_smoke_test.jl
# ============================================================================

include(joinpath(@__DIR__, "..", "src", "includes.jl"))

logpath, tee, logfile = setup_logging("smoke_test")

println(tee, "=" ^ 80)
println(tee, "Smoke Test: All problems, projections, initial points, and solvers")
println(tee, "=" ^ 80)

# ── Part 1: Verify all 20 problems evaluate without error ────────────────────
println(tee, "\n--- Part 1: Problem evaluation (n=100) ---")
n_test = 100
n_prob_pass = Ref(0)
for pid in PROBLEM_IDS
    try
        prob = get_problem(pid, n_test)
        x = ones(n_test)
        Gx = prob.G(x)
        px = prob.proj(x)
        @assert length(Gx) == n_test "G output dimension mismatch"
        @assert length(px) == n_test "proj output dimension mismatch"
        @assert all(isfinite, Gx) "G returned non-finite values"
        @assert all(isfinite, px) "proj returned non-finite values"
        @tprintf(tee, "  P%-2d %-12s  G(1)=%.2e  proj OK\n", pid, prob.name, norm(Gx))
        n_prob_pass[] += 1
    catch e
        @tprintf(tee, "  P%-2d  FAIL: %s\n", pid, sprint(showerror, e))
    end
end
@tprintf(tee, "  Result: %d/%d problems OK\n", n_prob_pass[], NUM_PROBLEMS)

# ── Part 2: Verify initial points ────────────────────────────────────────────
println(tee, "\n--- Part 2: Initial points (n=100) ---")
inits = get_initial_points(n_test)
@tprintf(tee, "  %d initial points defined\n", length(inits))
n_init_ok = Ref(0)
for (x0, label) in inits
    ok = length(x0) == n_test && all(isfinite, x0)
    if ok
        n_init_ok[] += 1
    else
        @tprintf(tee, "  FAIL: x0='%s' len=%d finite=%s\n", label, length(x0), all(isfinite, x0))
    end
end
@tprintf(tee, "  Result: %d/%d initial points OK\n", n_init_ok[], length(inits))

# ── Part 3: Verify projections ───────────────────────────────────────────────
println(tee, "\n--- Part 3: Projections ---")
x_neg = -5.0 * ones(n_test)
# proj_Rn_plus should clamp to [0, ∞)
px = proj_Rn_plus(x_neg)
@assert all(px .>= 0.0) "proj_Rn_plus failed: not >= 0"
@assert all(px .== 0.0) "proj_Rn_plus failed: expected 0 for input -5"
println(tee, "  proj_Rn_plus(x) = max(x,0)  ✓")

# ── Part 4: All 5 solvers on all 20 problems (n=100, x0=ones) ────────────────
println(tee, "\n--- Part 4: All solvers × all problems (n=100, x0=ones, eps=1e-6) ---")

methods = [
    GMOPCGMMethod(),
    GCGPMMethod(),
    MOPCGMMethod(),
    CGPMMethod(),
    STTDFPMMethod(),
]
mnames = method_name.(methods)

# Header
@tprintf(tee, "  %-4s", "Prob")
for mn in mnames
    @tprintf(tee, "  %10s", mn)
end
println(tee, "")
println(tee, "  " * "-"^60)

pass_count = zeros(Int, length(methods))
fail_list = Tuple{Int,String}[]

for pid in PROBLEM_IDS
    prob = get_problem(pid, n_test)
    x0 = ones(n_test)
    # Problem 18 needs positive x0 (log)
    if pid == 18
        x0 = 0.5 * ones(n_test)
    end

    @tprintf(tee, "  %-4s", prob.name)
    for (j, method) in enumerate(methods)
        result = try
            solve(method, prob, x0; eps=1e-6, maxiter=500, cb=ProgressCallback())
        catch e
            SolverResult(false, 0, 0, NaN, 0.0, x0)
        end

        if result.converged
            pass_count[j] += 1
            @tprintf(tee, "  %4d/%4dF", result.iterations, result.f_evals)
        else
            push!(fail_list, (pid, mnames[j]))
            @tprintf(tee, "     FAIL  ")
        end
    end
    println(tee, "")
end

println(tee, "  " * "-"^60)
@tprintf(tee, "  %-4s", "PASS")
for j in 1:length(methods)
    @tprintf(tee, "  %5d/%-4d", pass_count[j], NUM_PROBLEMS)
end
println(tee, "")

# ── Part 5: Test across dimensions ───────────────────────────────────────────
println(tee, "\n--- Part 5: Dimension scaling (GMOPCGM on P1, x0=ones) ---")
gm = GMOPCGMMethod()
for dim in [100, 1000, 5000]
    prob = get_problem(1, dim)
    x0 = ones(dim)
    r = solve(gm, prob, x0; eps=1e-8, maxiter=1000, cb=ProgressCallback())
    status = r.converged ? "OK" : "FAIL"
    @tprintf(tee, "  n=%-6d  %4s  IT=%-5d  FE=%-5d  ||G||=%.2e  %.3fs\n",
             dim, status, r.iterations, r.f_evals, r.residual, r.cpu_time)
end

# ── Summary ──────────────────────────────────────────────────────────────────
println(tee, "\n" * "=" ^ 80)
if isempty(fail_list)
    println(tee, "ALL TESTS PASSED.")
else
    @tprintf(tee, "%d failures:\n", length(fail_list))
    for (pid, mn) in fail_list
        @tprintf(tee, "  P%d — %s\n", pid, mn)
    end
end

teardown_logging(tee, logpath)
