# solvers.jl — Solver implementations for all 5 methods
# Each verified against oldcode/ and reference papers.

# ── Progress callback ─────────────────────────────────────────────────────────
# The caller creates a ProgressCallback and passes it to solve().
# Inside the iteration loop, the solver calls _pcb(cb, k, normG) to update
# the external progress bar's showvalues with live iteration info.

mutable struct ProgressCallback
    prog::Any          # Progress bar (or nothing)
    label::String      # e.g. "SOPP P4 n=10000"
    maxiter::Int
    n_done::Ref{Int}   # total runs done (shared across all solves)
    n_total::Int       # total runs planned
    n_conv::Ref{Int}
    n_fail::Ref{Int}
end

ProgressCallback() = ProgressCallback(nothing, "", 0, Ref(0), 0, Ref(0), Ref(0))

function _pcb(cb::ProgressCallback, k::Int, normG::Float64)
    cb.prog === nothing && return
    ProgressMeter.update!(cb.prog, cb.n_done[];
        showvalues=[
            (:done, "$(cb.n_done[])/$(cb.n_total)"),
            (:converged, cb.n_conv[]),
            (:failed, cb.n_fail[]),
            (:current, "$(cb.label) ($(k)/$(cb.maxiter)) ||G||=$(@sprintf("%.1e", normG))")
        ])
end

function _pcb_done!(cb::ProgressCallback, converged::Bool)
    cb.prog === nothing && return
    cb.n_done[] += 1
    converged ? (cb.n_conv[] += 1) : (cb.n_fail[] += 1)
end

# Non-finite guard: terminate early if residual or direction blows up
_isnan_bail(normG, p) = !isfinite(normG) || any(v -> !isfinite(v), p)

_elapsed_seconds(t0_ns::UInt64, t1_ns::UInt64=time_ns()) =
    Float64(t1_ns - t0_ns) / 1e9

function _finish(cb::ProgressCallback, converged::Bool, status::Symbol,
                 iterations::Int, fe::Int, residual::Real, t0_ns::UInt64,
                 x::Vector{Float64}; restarts::Int=0)
    elapsed = _elapsed_seconds(t0_ns)
    _pcb_done!(cb, converged)
    SolverResult(converged, iterations, fe, residual, elapsed, x;
                 status=status, restarts=restarts)
end

function _is_feasible(prob::TestProblem, x::Vector{Float64})
    px = prob.proj(x)
    scale = max(1.0, norm(x), norm(px))
    norm(px .- x) <= sqrt(eps(Float64)) * scale
end

_update_gamma(::SOPPMethod, gamma::Real, improved::Bool) =
    clamp(improved ? 1.1 * gamma : gamma, 1.0, 1.8)

_update_gamma(::SDLPMethod, gamma::Real, improved::Bool) =
    clamp((improved ? 1.1 : 1.05) * gamma, 1.0, 1.95)

function _restart_direction(Gnext, lambda_current, alpha_min, alpha_max)
    if isfinite(lambda_current) && alpha_min <= lambda_current <= alpha_max
        return -lambda_current .* Gnext, Float64(lambda_current)
    end
    return -Gnext, 1.0
end

function _sopp_spectral_candidate(m::SOPPMethod, k::Integer, s2, sq, q2)
    bb1 = s2 / sq
    bb2 = sq / q2
    m.srule === :max && return max(bb1, bb2)
    m.srule === :bb1 && return bb1
    m.srule === :bb2 && return bb2
    m.srule === :abb && return iseven(k) ? bb1 : bb2
    throw(ArgumentError("Unsupported SOPP spectral rule: $(m.srule)"))
end

function _sopp_t_parameter(m::SOPPMethod, lambda, s2, sq, q2)
    m.tsel === :cond && return lambda / 2 * (sq / s2 + q2 / sq)
    m.tsel === :gap && return lambda * (sq / s2)
    throw(ArgumentError("Unsupported SOPP Perry parameter rule: $(m.tsel)"))
end

function _sdlp_spectral_candidate(m::SDLPMethod, k::Integer, p2, pw, w2)
    r1 = w2 / pw
    r2 = pw / p2
    m.srule === :max && return max(r1, r2)
    m.srule === :r1 && return r1
    m.srule === :r2 && return r2
    m.srule === :alt && return iseven(k) ? r1 : r2
    throw(ArgumentError("Unsupported SDLP spectral rule: $(m.srule)"))
end

function _sopp_next_direction(m::SOPPMethod, Gx, Gz, Gnext, s, p,
                              lambda_current, k::Integer=0;
                              on_spectral::Union{Nothing,Function}=nothing)
    q = Gz .- Gx .+ m.tau .* s
    s2 = dot(s, s)
    sq = dot(s, q)
    q2 = dot(q, q)
    pq = dot(p, q)
    if !all(isfinite, q) || !all(isfinite, (s2, sq, q2, pq)) ||
       !(s2 > 0 && sq > 0 && q2 > 0 && pq > 0)
        direction, lambda = _restart_direction(
            Gnext, lambda_current, m.alpha_min, m.alpha_max)
        return (direction=direction, lambda=lambda, restarted=true)
    end

    on_spectral === nothing || on_spectral(k, s2 / sq, sq / q2)
    lambda = spectral_proj(
        _sopp_spectral_candidate(m, k, s2, sq, q2),
        m.alpha_min, m.alpha_max)
    tstar = _sopp_t_parameter(m, lambda, s2, sq, q2)
    theta = dot(q .- tstar .* s, Gnext) / pq
    G2 = dot(Gnext, Gnext)
    if !all(isfinite, (lambda, tstar, theta, G2)) || !(G2 > 0)
        direction, fallback = _restart_direction(
            Gnext, lambda, m.alpha_min, m.alpha_max)
        return (direction=direction, lambda=fallback, restarted=true)
    end

    pperp = p .- (dot(Gnext, p) / G2) .* Gnext
    pperp_norm = norm(pperp)
    bar_theta = if pperp_norm == 0.0
        0.0
    else
        clip_limit = sqrt(m.c^2 - lambda^2) * sqrt(G2) / pperp_norm
        sign(theta) * min(abs(theta), clip_limit)
    end
    direction = -lambda .* Gnext .+ bar_theta .* pperp
    if !isfinite(bar_theta) || !all(isfinite, pperp) || !all(isfinite, direction)
        direction, fallback = _restart_direction(
            Gnext, lambda, m.alpha_min, m.alpha_max)
        return (direction=direction, lambda=fallback, restarted=true)
    end
    return (direction=direction, lambda=Float64(lambda), restarted=false)
end

function _sdlp_next_direction(m::SDLPMethod, Gx, Gnext, p, lambda_current,
                              k::Integer=0;
                              on_spectral::Union{Nothing,Function}=nothing)
    y = Gnext .- Gx
    p2 = dot(p, p)
    if !all(isfinite, y) || !isfinite(p2) || !(p2 > 0)
        direction, lambda = _restart_direction(
            Gnext, lambda_current, m.alpha_min, m.alpha_max)
        return (direction=direction, lambda=lambda, restarted=true)
    end

    r = 1.0 + max(0.0, -dot(y, p) / p2)
    w = y .+ r .* p
    pw = dot(p, w)
    w2 = dot(w, w)
    if !isfinite(r) || !all(isfinite, w) || !all(isfinite, (pw, w2)) ||
       !(pw > 0 && w2 > 0)
        direction, lambda = _restart_direction(
            Gnext, lambda_current, m.alpha_min, m.alpha_max)
        return (direction=direction, lambda=lambda, restarted=true)
    end

    on_spectral === nothing || on_spectral(k, w2 / pw, pw / p2)
    lambda = spectral_proj(
        _sdlp_spectral_candidate(m, k, p2, pw, w2),
        m.alpha_min, m.alpha_max)
    Gp = dot(Gnext, p)
    a = Gp / pw
    theta = dot(Gnext, w) / pw - lambda * (w2 / pw) * (Gp / pw)
    direction = -lambda .* Gnext .+ theta .* p .+ m.tau .* a .* w
    if !all(isfinite, (lambda, a, theta)) || !all(isfinite, direction)
        direction, fallback = _restart_direction(
            Gnext, lambda, m.alpha_min, m.alpha_max)
        return (direction=direction, lambda=fallback, restarted=true)
    end
    return (direction=direction, lambda=Float64(lambda), restarted=false)
end

# ── Shared line search for SOPP and SDLP ─────────────────────────────────────

function _backtrack_ours(G_func, x, p, rho, beta, zeta, zeta1, zeta2, _fe)
    pp = dot(p, p)
    for j in 0:50
        alpha = beta * rho^j
        z = x .+ alpha .* p
        Gz = G_func(z)
        normGz = norm(Gz)
        proj_val = clamp_scalar(normGz, zeta1, zeta2)
        if dot(Gz, p) <= -zeta * alpha * pp * proj_val
            return alpha, z, Gz
        end
        alpha <= 1e-15 && break
    end
    return nothing, x, nothing  # caller already holds G(x)
end

# ═══════════════════════════════════════════════════════════════════════════════
# SOPP
# ═══════════════════════════════════════════════════════════════════════════════

function solve(m::SOPPMethod, prob::TestProblem, x0::Vector{Float64};
               eps=1e-11, maxiter=2000, cb::ProgressCallback=ProgressCallback(),
               stall_limit::Int=typemax(Int),
               on_iter::Union{Nothing,Function}=nothing,
               on_spectral::Union{Nothing,Function}=nothing)
    t0_ns = time_ns()
    fe = Ref(0)
    G(x) = (fe[] += 1; prob.G(x))

    x = prob.proj(copy(x0)); Gx = G(x); normGx = norm(Gx)
    on_iter === nothing || on_iter(0, normGx)
    normGx <= eps && return _finish(
        cb, true, :converged_initial, 0, fe[], normGx, t0_ns, x)

    p = -Gx; phi = m.lambda0; gamma_val = m.gamma; stall = 0; restarts = 0

    for k in 1:maxiter
        _pcb(cb, k, normGx)
        _isnan_bail(normGx, p) && return _finish(
            cb, false, :nonfinite, k, fe[], NaN, t0_ns, x; restarts=restarts)
        stall > stall_limit && return _finish(
            cb, false, :stalled, k, fe[], normGx, t0_ns, x; restarts=restarts)

        alpha, z, Gz = _backtrack_ours(G, x, p, m.rho, m.beta, m.zeta, m.zeta1, m.zeta2, fe)
        alpha === nothing && return _finish(
            cb, false, :line_search_failed, k, fe[], normGx, t0_ns, x; restarts=restarts)
        normGz = norm(Gz)
        if normGz <= eps && _is_feasible(prob, z)
            on_iter === nothing || on_iter(k, normGz)
            return _finish(cb, true, :converged_trial, k, fe[], normGz, t0_ns, z;
                           restarts=restarts)
        end
        (!isfinite(normGz) || normGz == 0.0) && return _finish(
            cb, false, :invalid_trial_residual, k, fe[], normGz, t0_ns, x;
            restarts=restarts)

        mu_k = dot(Gz, x .- z) / normGz^2
        x_new = prob.proj(x .- gamma_val .* mu_k .* Gz)
        Gx_new = G(x_new); normGx_new = norm(Gx_new)
        on_iter === nothing || on_iter(k, normGx_new)
        if normGx_new <= eps
            return _finish(cb, true, :converged_projected, k, fe[], normGx_new,
                           t0_ns, x_new; restarts=restarts)
        end
        !isfinite(normGx_new) && return _finish(
            cb, false, :nonfinite, k, fe[], normGx_new, t0_ns, x_new;
            restarts=restarts)

        if normGx_new < normGx
            gamma_val = _update_gamma(m, gamma_val, true); stall = 0
        else
            gamma_val = _update_gamma(m, gamma_val, false); stall += 1
        end

        s = z .- x
        update = _sopp_next_direction(
            m, Gx, Gz, Gx_new, s, p, phi, k; on_spectral=on_spectral)
        p = update.direction
        phi = update.lambda
        restarts += update.restarted

        x = x_new; Gx = Gx_new; normGx = normGx_new
        norm(p) * 10 < eps && return _finish(
            cb, false, :stalled_direction, k, fe[], normGx, t0_ns, x;
            restarts=restarts)
    end
    return _finish(cb, false, :maxiter, maxiter, fe[], normGx, t0_ns, x;
                   restarts=restarts)
end

# ═══════════════════════════════════════════════════════════════════════════════
# SDLP
# ═══════════════════════════════════════════════════════════════════════════════

function solve(m::SDLPMethod, prob::TestProblem, x0::Vector{Float64};
               eps=1e-11, maxiter=2000, cb::ProgressCallback=ProgressCallback(),
               on_iter::Union{Nothing,Function}=nothing,
               on_spectral::Union{Nothing,Function}=nothing, kwargs...)
    t0_ns = time_ns()
    fe = Ref(0)
    G(x) = (fe[] += 1; prob.G(x))

    x = prob.proj(copy(x0)); Gx = G(x); normGx = norm(Gx)
    on_iter === nothing || on_iter(0, normGx)
    normGx <= eps && return _finish(
        cb, true, :converged_initial, 0, fe[], normGx, t0_ns, x)

    d = -Gx; phi = m.lambda0; gamma_val = m.gamma; restarts = 0

    for k in 1:maxiter
        _pcb(cb, k, normGx)
        _isnan_bail(normGx, d) && return _finish(
            cb, false, :nonfinite, k, fe[], NaN, t0_ns, x; restarts=restarts)

        alpha, z, Gz = _backtrack_ours(G, x, d, m.rho, m.eta, m.zeta, m.zeta1, m.zeta2, fe)
        alpha === nothing && return _finish(
            cb, false, :line_search_failed, k, fe[], normGx, t0_ns, x; restarts=restarts)
        normGz = norm(Gz)
        if normGz <= eps && _is_feasible(prob, z)
            on_iter === nothing || on_iter(k, normGz)
            return _finish(cb, true, :converged_trial, k, fe[], normGz, t0_ns, z;
                           restarts=restarts)
        end
        (!isfinite(normGz) || normGz == 0.0) && return _finish(
            cb, false, :invalid_trial_residual, k, fe[], normGz, t0_ns, x;
            restarts=restarts)

        tau_k = dot(Gz, x .- z) / normGz^2
        x_new = prob.proj(x .- gamma_val .* tau_k .* Gz)
        Gx_new = G(x_new); normGx_new = norm(Gx_new)
        on_iter === nothing || on_iter(k, normGx_new)
        if normGx_new <= eps
            return _finish(cb, true, :converged_projected, k, fe[], normGx_new,
                           t0_ns, x_new; restarts=restarts)
        end
        !isfinite(normGx_new) && return _finish(
            cb, false, :nonfinite, k, fe[], normGx_new, t0_ns, x_new;
            restarts=restarts)

        gamma_val = _update_gamma(m, gamma_val, normGx_new < normGx)

        update = _sdlp_next_direction(
            m, Gx, Gx_new, d, phi, k; on_spectral=on_spectral)
        d = update.direction
        phi = update.lambda
        restarts += update.restarted

        x = x_new; Gx = Gx_new; normGx = normGx_new
        norm(d) * 10 < eps && return _finish(
            cb, false, :stalled_direction, k, fe[], normGx, t0_ns, x;
            restarts=restarts)
    end
    return _finish(cb, false, :maxiter, maxiter, fe[], normGx, t0_ns, x;
                   restarts=restarts)
end

# ═══════════════════════════════════════════════════════════════════════════════
# MOPCGM (Sabi'u et al. 2023)
# ═══════════════════════════════════════════════════════════════════════════════

function solve(m::MOPCGMMethod, prob::TestProblem, x0::Vector{Float64};
               eps=1e-11, maxiter=2000, cb::ProgressCallback=ProgressCallback(),
               on_iter::Union{Nothing,Function}=nothing, kwargs...)
    t0_ns = time_ns()
    fe = Ref(0)
    F(x) = (fe[] += 1; prob.G(x))

    x = copy(x0); Fx = F(x); normFx = norm(Fx)
    on_iter === nothing || on_iter(0, normFx)
    normFx <= eps && return _finish(
        cb, true, :converged_initial, 0, fe[], normFx, t0_ns, x)

    d = -Fx

    for k in 1:maxiter
        _pcb(cb, k, normFx)
        _isnan_bail(normFx, d) && return _finish(
            cb, false, :nonfinite, k, fe[], NaN, t0_ns, x)

        dd = dot(d, d); alpha = m.rho; z = x; Fz = Fx
        ls_ok = false
        for j in 0:50
            alpha = m.rho * m.eta^j
            z = x .+ alpha .* d; Fz = F(z)
            if -dot(Fz, d) >= m.zeta * alpha * dd; ls_ok = true; break; end
            alpha <= 1e-5 && break
        end
        ls_ok || return _finish(
            cb, false, :line_search_failed, k, fe[], normFx, t0_ns, x)

        normFmu = norm(Fz)
        if normFmu <= eps && _is_feasible(prob, z)
            on_iter === nothing || on_iter(k, normFmu)
            return _finish(cb, true, :converged_trial, k, fe[], normFmu, t0_ns, z)
        end
        (!isfinite(normFmu) || normFmu == 0.0) && return _finish(
            cb, false, :invalid_trial_residual, k, fe[], normFmu, t0_ns, x)

        bk = dot(Fz, x .- z) / normFmu^2
        x_new = prob.proj(x .- bk .* Fz)
        Fx_new = F(x_new)
        normFx_new = norm(Fx_new)
        on_iter === nothing || on_iter(k, normFx_new)
        normFx_new <= eps && return _finish(
            cb, true, :converged_projected, k, fe[], normFx_new, t0_ns, x_new)
        !isfinite(normFx_new) && return _finish(
            cb, false, :nonfinite, k, fe[], normFx_new, t0_ns, x_new)

        s = z .- x; y = Fx_new .- Fx .+ m.lambda .* s
        ss = dot(s, s); sy = dot(s, y)
        theta_star = ss > 1e-30 ? sy / ss : 0.0
        dy = dot(d, y)
        beta_mop = abs(dy) > 1e-30 ? dot(y .- theta_star .* s, Fx_new) / dy : 0.0
        Fn2 = dot(Fx_new, Fx_new)
        r_coeff = Fn2 > 1e-30 ? dot(Fx_new, d) / Fn2 : 0.0
        d = -(1.0 + beta_mop * r_coeff) .* Fx_new .+ beta_mop .* d

        x = x_new; Fx = Fx_new; normFx = normFx_new
        norm(d) * 10 < eps && return _finish(
            cb, false, :stalled_direction, k, fe[], normFx, t0_ns, x)
    end
    return _finish(cb, false, :maxiter, maxiter, fe[], normFx, t0_ns, x)
end

# ═══════════════════════════════════════════════════════════════════════════════
# CGPM (Zheng et al. 2020)
# ═══════════════════════════════════════════════════════════════════════════════

function solve(m::CGPMMethod, prob::TestProblem, x0::Vector{Float64};
               eps=1e-11, maxiter=2000, cb::ProgressCallback=ProgressCallback(),
               on_iter::Union{Nothing,Function}=nothing, kwargs...)
    t0_ns = time_ns()
    fe = Ref(0)
    F(x) = (fe[] += 1; prob.G(x))

    x = copy(x0); Fx = F(x); normFx = norm(Fx)
    on_iter === nothing || on_iter(0, normFx)
    normFx <= eps && return _finish(
        cb, true, :converged_initial, 0, fe[], normFx, t0_ns, x)

    d = -Fx

    for k in 1:maxiter
        _pcb(cb, k, normFx)
        _isnan_bail(normFx, d) && return _finish(
            cb, false, :nonfinite, k, fe[], NaN, t0_ns, x)

        dnorm2 = dot(d, d); alpha = m.beta_ls; z = x; Fz = Fx
        ls_ok = false
        for j in 0:50
            alpha = m.beta_ls * m.rho^j
            z = x .+ alpha .* d; Fz = F(z)
            if -dot(Fz, d) >= m.sigma * alpha * norm(Fz) * dnorm2; ls_ok = true; break; end
            alpha <= 1e-5 && break
        end
        ls_ok || return _finish(
            cb, false, :line_search_failed, k, fe[], normFx, t0_ns, x)

        normFz = norm(Fz)
        if normFz <= eps && _is_feasible(prob, z)
            on_iter === nothing || on_iter(k, normFz)
            return _finish(cb, true, :converged_trial, k, fe[], normFz, t0_ns, z)
        end
        (!isfinite(normFz) || normFz == 0.0) && return _finish(
            cb, false, :invalid_trial_residual, k, fe[], normFz, t0_ns, x)

        tau_k = dot(Fz, x .- z) / normFz^2
        x_new = prob.proj(x .- tau_k .* Fz)
        Fx_new = F(x_new)
        normFx_new = norm(Fx_new)
        on_iter === nothing || on_iter(k, normFx_new)
        normFx_new <= eps && return _finish(
            cb, true, :converged_projected, k, fe[], normFx_new, t0_ns, x_new)
        !isfinite(normFx_new) && return _finish(
            cb, false, :nonfinite, k, fe[], normFx_new, t0_ns, x_new)

        y = Fx_new .- Fx; dd = dot(d, d)
        lambda_cgpm = 1.0 + max(0.0, -dot(y, d) / max(dd, 1e-30))
        w = y .+ lambda_cgpm .* d; wd = dot(w, d)

        if abs(wd) > 1e-30
            c_k = dot(Fx_new, d) / wd
            beta_k = (dot(Fx_new, w) - 2.0 * c_k * dot(w, w)) / wd
            d = -m.sigma1 .* Fx_new .+ beta_k .* d .+ m.sigma2 .* c_k .* w
        else
            d = -m.sigma1 .* Fx_new
        end

        x = x_new; Fx = Fx_new; normFx = normFx_new
        norm(d) * 10 < eps && return _finish(
            cb, false, :stalled_direction, k, fe[], normFx, t0_ns, x)
    end
    return _finish(cb, false, :maxiter, maxiter, fe[], normFx, t0_ns, x)
end

# ═══════════════════════════════════════════════════════════════════════════════
# STTDFPM (Ibrahim et al. 2023)
# ═══════════════════════════════════════════════════════════════════════════════

function solve(m::STTDFPMMethod, prob::TestProblem, x0::Vector{Float64};
               eps=1e-11, maxiter=2000, cb::ProgressCallback=ProgressCallback(),
               on_iter::Union{Nothing,Function}=nothing, kwargs...)
    t0_ns = time_ns()
    fe = Ref(0)
    F(x) = (fe[] += 1; prob.G(x))

    x = copy(x0); Fx = F(x); normFx = norm(Fx)
    on_iter === nothing || on_iter(0, normFx)
    normFx <= eps && return _finish(
        cb, true, :converged_initial, 0, fe[], normFx, t0_ns, x)

    d = -Fx

    for k in 1:maxiter
        _pcb(cb, k, normFx)
        _isnan_bail(normFx, d) && return _finish(
            cb, false, :nonfinite, k, fe[], NaN, t0_ns, x)
        norm(d) * 10 < eps && return _finish(
            cb, false, :stalled_direction, k, fe[], normFx, t0_ns, x)

        dd = dot(d, d); tau = 1.0; z = x; Fz = Fx; ls_ok = false
        for j in 0:50
            tau = m.beta_ls^j
            z = x .+ tau .* d; Fz = F(z)
            proj_val = clamp_scalar(norm(Fz), m.eta1, m.eta2)
            if -dot(Fz, d) >= m.sigma * tau * proj_val * dd; ls_ok = true; break; end
            tau <= 1e-5 && break
        end
        ls_ok || return _finish(
            cb, false, :line_search_failed, k, fe[], normFx, t0_ns, x)

        normFz = norm(Fz)
        if normFz <= eps && _is_feasible(prob, z)
            on_iter === nothing || on_iter(k, normFz)
            return _finish(cb, true, :converged_trial, k, fe[], normFz, t0_ns, z)
        end
        (!isfinite(normFz) || normFz == 0.0) && return _finish(
            cb, false, :invalid_trial_residual, k, fe[], normFz, t0_ns, x)

        mu_k = dot(Fz, x .- z) / normFz^2
        x_new = prob.proj(x .- m.gamma .* mu_k .* Fz)
        Fx_new = F(x_new)
        normFx_new = norm(Fx_new)
        on_iter === nothing || on_iter(k, normFx_new)
        normFx_new <= eps && return _finish(
            cb, true, :converged_projected, k, fe[], normFx_new, t0_ns, x_new)
        !isfinite(normFx_new) && return _finish(
            cb, false, :nonfinite, k, fe[], normFx_new, t0_ns, x_new)

        y = Fx_new .- Fx
        s_sttd = (x_new .- x) .+ m.r_param .* y
        yy = dot(y, y); sy = dot(s_sttd, y)
        alpha_I = yy > 1e-30 ? spectral_proj(sy / yy, m.alpha_min, m.alpha_max) : 1.0
        v_k = max(m.psi * norm(d) * norm(y), normFx^2)

        if abs(v_k) > 1e-30
            beta_k = dot(Fx_new, y) / v_k
            alpha_II = dot(Fx_new, d) / v_k
        else
            beta_k = 0.0; alpha_II = 0.0
        end
        d = -alpha_I .* Fx_new .+ beta_k .* d .- alpha_II .* y

        x = x_new; Fx = Fx_new; normFx = normFx_new
    end
    return _finish(cb, false, :maxiter, maxiter, fe[], normFx, t0_ns, x)
end
