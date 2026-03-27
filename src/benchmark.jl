# benchmark.jl — Multi-solver benchmarking utilities

function run_single(method::AbstractMethod, prob::TestProblem, x0::Vector{Float64};
                    eps=1e-11, maxiter=2000)
    try
        result = solve(method, prob, x0; eps=eps, maxiter=maxiter)
        return result
    catch e
        return SolverResult(false, 0, 0, NaN, 0.0, x0)
    end
end

function run_benchmark(methods::Vector, prob_ids, dimensions, initial_points_fn;
                       eps=1e-11, maxiter=2000, verbose=false)
    results = []

    for n in dimensions
        inits = initial_points_fn(n)
        for pid in prob_ids
            prob = get_problem(pid, n)
            for (x0, x0_name) in inits
                for method in methods
                    r = run_single(method, prob, x0; eps=eps, maxiter=maxiter)
                    push!(results, (
                        method = method_name(method),
                        problem = prob.name,
                        n = n,
                        x0_label = x0_name,
                        converged = r.converged,
                        iterations = r.iterations,
                        f_evals = r.f_evals,
                        residual = r.residual,
                        cpu_time = r.cpu_time,
                    ))
                    if verbose
                        status = r.converged ? "OK" : "FAIL"
                        @printf("  %-8s %-4s n=%-6d x0=%-6s  %4s  IT=%-5d FE=%-5d  ||G||=%.2e  %.3fs\n",
                                method_name(method), prob.name, n, x0_name,
                                status, r.iterations, r.f_evals, r.residual, r.time_seconds)
                    end
                end
            end
        end
    end

    return results
end
