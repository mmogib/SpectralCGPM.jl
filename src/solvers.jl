# solvers.jl — Solver implementations for all 5 methods
# Each verified against oldcode/ and reference papers.

# ── Progress callback ─────────────────────────────────────────────────────────
# The caller creates a ProgressCallback and passes it to solve().
# Inside the iteration loop, the solver calls _pcb(cb, k, normG) to update
# the external progress bar's showvalues with live iteration info.

mutable struct ProgressCallback
    prog::Any          # Progress bar (or nothing)
    label::String      # e.g. "GMOPCGM P4 n=10000"
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

# NaN guard: terminate early if residual or direction blows up
_isnan_bail(normG, p) = isnan(normG) || isinf(normG) || any(isnan, p)

# ── Shared line search for GMOPCGM and GCGPM ─────────────────────────────────

function _backtrack_ours(G_func, x, p, rho, beta, zeta, zeta1, zeta2, fe)
    pp = dot(p, p)
    for j in 0:50
        alpha = beta * rho^j
        z = x .+ alpha .* p
        Gz = G_func(z)
        fe[] += 1
        normGz = norm(Gz)
        proj_val = clamp_scalar(normGz, zeta1, zeta2)
        if dot(Gz, p) <= -zeta * alpha * pp * proj_val
            return alpha, z, Gz
        end
        alpha <= 1e-15 && break
    end
    return nothing, x, G_func(x)  # line search failed
end

# ═══════════════════════════════════════════════════════════════════════════════
# GMOPCGM
# ═══════════════════════════════════════════════════════════════════════════════

function solve(m::GMOPCGMMethod, prob::TestProblem, x0::Vector{Float64};
               eps=1e-11, maxiter=2000, cb::ProgressCallback=ProgressCallback(),
               stall_limit::Int=10)
    t0 = time()
    fe = Ref(0)
    G(x) = (fe[] += 1; prob.G(x))

    x = copy(x0); Gx = G(x); normGx = norm(Gx)
    normGx < eps && (_pcb_done!(cb, true); return SolverResult(true, 0, fe[], normGx, time()-t0, x))

    p = -Gx; phi = m.lambda0; gamma_val = m.gamma; stall = 0

    for k in 1:maxiter
        _pcb(cb, k, normGx)
        _isnan_bail(normGx, p) && (_pcb_done!(cb, false); return SolverResult(false, k, fe[], NaN, time()-t0, x))
        stall > stall_limit && (_pcb_done!(cb, false); return SolverResult(false, k, fe[], normGx, time()-t0, x))

        alpha, z, Gz = _backtrack_ours(G, x, p, m.rho, m.beta, m.zeta, m.zeta1, m.zeta2, fe)
        alpha === nothing && (_pcb_done!(cb, false); return SolverResult(false, k, fe[], normGx, time()-t0, x))
        normGz = norm(Gz)
        if normGz < eps; _pcb_done!(cb, true); return SolverResult(true, k, fe[], normGz, time()-t0, z); end

        mu_k = dot(Gz, x .- z) / normGz^2
        x_new = prob.proj(x .- gamma_val .* mu_k .* Gz)
        Gx_new = G(x_new); normGx_new = norm(Gx_new)

        if normGx_new < normGx
            gamma_val = min(gamma_val * 1.1, 1.8); stall = 0
        else
            gamma_val = max(gamma_val * 1.0, 1.0); stall += 1
        end

        s = z .- x; y = Gx_new .- Gx; v = y .+ m.tau .* s
        norms2 = dot(s, s); sv = dot(s, v); normv2 = dot(v, v)

        if !(normGx_new < 0.75 * normGx) && norms2 > 1e-30 && abs(sv) > 1e-30
            phi = spectral_proj(max(norms2/sv, sv/normv2), m.alpha_min, m.alpha_max)
        end

        if norms2 > 1e-30 && abs(sv) > 1e-30
            theta_star = phi * sv / norms2
            dv = dot(p, v)
            beta_mop = abs(dv) > 1e-30 ? dot(v .- theta_star .* s, Gx_new) / dv : 0.0
            Gn2 = dot(Gx_new, Gx_new)
            r_coeff = Gn2 > 1e-30 ? dot(Gx_new, p) / Gn2 : 0.0
            p = -(phi + beta_mop * r_coeff) .* Gx_new .+ beta_mop .* p
        else
            p = -phi .* Gx_new
        end

        x = x_new; Gx = Gx_new; normGx = normGx_new
        normGx < eps && (_pcb_done!(cb, true); return SolverResult(true, k, fe[], normGx, time()-t0, x))
        norm(p) * 10 < eps && (_pcb_done!(cb, true); return SolverResult(true, k, fe[], normGx, time()-t0, x))
    end
    _pcb_done!(cb, false)
    return SolverResult(false, maxiter, fe[], normGx, time()-t0, x)
end

# ═══════════════════════════════════════════════════════════════════════════════
# GCGPM
# ═══════════════════════════════════════════════════════════════════════════════

function solve(m::GCGPMMethod, prob::TestProblem, x0::Vector{Float64};
               eps=1e-11, maxiter=2000, cb::ProgressCallback=ProgressCallback(), kwargs...)
    t0 = time()
    fe = Ref(0)
    G(x) = (fe[] += 1; prob.G(x))

    x = copy(x0); Gx = G(x); normGx = norm(Gx)
    normGx < eps && (_pcb_done!(cb, true); return SolverResult(true, 0, fe[], normGx, time()-t0, x))

    d = -Gx; phi = m.lambda0; gamma_val = m.gamma

    for k in 1:maxiter
        _pcb(cb, k, normGx)
        _isnan_bail(normGx, d) && (_pcb_done!(cb, false); return SolverResult(false, k, fe[], NaN, time()-t0, x))

        alpha, z, Gz = _backtrack_ours(G, x, d, m.rho, m.eta, m.zeta, m.zeta1, m.zeta2, fe)
        alpha === nothing && (_pcb_done!(cb, false); return SolverResult(false, k, fe[], normGx, time()-t0, x))
        normGz = norm(Gz)
        if normGz < eps; _pcb_done!(cb, true); return SolverResult(true, k, fe[], normGz, time()-t0, z); end

        tau_k = dot(Gz, x .- z) / normGz^2
        x_new = prob.proj(x .- gamma_val .* tau_k .* Gz)
        Gx_new = G(x_new); normGx_new = norm(Gx_new)

        gamma_val = normGx_new < normGx ? min(gamma_val * 1.1, 1.70) : min(max(gamma_val * 1.05, 1.05), 1.95)

        y = Gx_new .- Gx
        Gp = dot(Gx_new, d); yp = dot(y, d)
        r_k = 1.0 + max(0.0, -Gp / (yp + sign(yp)*1e-30))
        w = y .+ r_k .* d; wd = dot(w, d)

        if abs(wd) > 1e-30
            c_k = Gp / wd; ww = dot(w, w); dd = dot(d, d)
            if !(normGx_new < 0.6 * normGx) && dd > 1e-30
                phi = spectral_proj(max(ww/wd, wd/dd), m.alpha_min, m.alpha_max)
            end
            beta_k = (dot(Gx_new, w) - phi * c_k * ww) / wd
            d = -phi .* Gx_new .+ beta_k .* d .+ m.tau .* c_k .* w
        else
            d = -phi .* Gx_new
        end

        x = x_new; Gx = Gx_new; normGx = normGx_new
        normGx < eps && (_pcb_done!(cb, true); return SolverResult(true, k, fe[], normGx, time()-t0, x))
        norm(d) * 10 < eps && (_pcb_done!(cb, true); return SolverResult(true, k, fe[], normGx, time()-t0, x))
    end
    _pcb_done!(cb, false)
    return SolverResult(false, maxiter, fe[], normGx, time()-t0, x)
end

# ═══════════════════════════════════════════════════════════════════════════════
# MOPCGM (Sabi'u et al. 2023)
# ═══════════════════════════════════════════════════════════════════════════════

function solve(m::MOPCGMMethod, prob::TestProblem, x0::Vector{Float64};
               eps=1e-11, maxiter=2000, cb::ProgressCallback=ProgressCallback(), kwargs...)
    t0 = time()
    fe = Ref(0)
    F(x) = (fe[] += 1; prob.G(x))

    x = copy(x0); Fx = F(x); normFx = norm(Fx)
    normFx < eps && (_pcb_done!(cb, true); return SolverResult(true, 0, fe[], normFx, time()-t0, x))

    d = -Fx

    for k in 1:maxiter
        _pcb(cb, k, normFx)
        _isnan_bail(normFx, d) && (_pcb_done!(cb, false); return SolverResult(false, k, fe[], NaN, time()-t0, x))

        dd = dot(d, d); alpha = m.rho; z = x; Fz = Fx
        ls_ok = false
        for j in 0:50
            alpha = m.rho * m.eta^j
            z = x .+ alpha .* d; Fz = F(z)
            if -dot(Fz, d) >= m.zeta * alpha * dd; ls_ok = true; break; end
            alpha <= 1e-5 && break
        end
        ls_ok || (_pcb_done!(cb, false); return SolverResult(false, k, fe[], normFx, time()-t0, x))

        normFmu = norm(Fz)
        if normFmu < eps; _pcb_done!(cb, true); return SolverResult(true, k, fe[], normFmu, time()-t0, z); end

        bk = dot(Fz, x .- z) / normFmu^2
        x_new = prob.proj(x .- bk .* Fz)
        Fx_new = F(x_new)

        s = z .- x; y = Fx_new .- Fx .+ m.lambda .* s
        ss = dot(s, s); sy = dot(s, y)
        theta_star = ss > 1e-30 ? sy / ss : 0.0
        dy = dot(d, y)
        beta_mop = abs(dy) > 1e-30 ? dot(y .- theta_star .* s, Fx_new) / dy : 0.0
        Fn2 = dot(Fx_new, Fx_new)
        r_coeff = Fn2 > 1e-30 ? dot(Fx_new, d) / Fn2 : 0.0
        d = -(1.0 + beta_mop * r_coeff) .* Fx_new .+ beta_mop .* d

        x = x_new; Fx = Fx_new; normFx = norm(Fx)
        normFx < eps && (_pcb_done!(cb, true); return SolverResult(true, k, fe[], normFx, time()-t0, x))
        norm(d) * 10 < eps && (_pcb_done!(cb, true); return SolverResult(true, k, fe[], normFx, time()-t0, x))
    end
    _pcb_done!(cb, false)
    return SolverResult(false, maxiter, fe[], normFx, time()-t0, x)
end

# ═══════════════════════════════════════════════════════════════════════════════
# CGPM (Zheng et al. 2020)
# ═══════════════════════════════════════════════════════════════════════════════

function solve(m::CGPMMethod, prob::TestProblem, x0::Vector{Float64};
               eps=1e-11, maxiter=2000, cb::ProgressCallback=ProgressCallback(), kwargs...)
    t0 = time()
    fe = Ref(0)
    F(x) = (fe[] += 1; prob.G(x))

    x = copy(x0); Fx = F(x); normFx = norm(Fx)
    normFx < eps && (_pcb_done!(cb, true); return SolverResult(true, 0, fe[], normFx, time()-t0, x))

    d = -Fx

    for k in 1:maxiter
        _pcb(cb, k, normFx)
        _isnan_bail(normFx, d) && (_pcb_done!(cb, false); return SolverResult(false, k, fe[], NaN, time()-t0, x))

        dnorm2 = dot(d, d); alpha = m.beta_ls; z = x; Fz = Fx
        ls_ok = false
        for j in 0:50
            alpha = m.beta_ls * m.rho^j
            z = x .+ alpha .* d; Fz = F(z)
            if -dot(Fz, d) >= m.sigma * alpha * norm(Fz) * dnorm2; ls_ok = true; break; end
            alpha <= 1e-5 && break
        end
        ls_ok || (_pcb_done!(cb, false); return SolverResult(false, k, fe[], normFx, time()-t0, x))

        normFz = norm(Fz)
        if normFz < eps; _pcb_done!(cb, true); return SolverResult(true, k, fe[], normFz, time()-t0, z); end

        tau_k = dot(Fz, x .- z) / normFz^2
        x_new = prob.proj(x .- tau_k .* Fz)
        Fx_new = F(x_new)

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

        x = x_new; Fx = Fx_new; normFx = norm(Fx)
        normFx < eps && (_pcb_done!(cb, true); return SolverResult(true, k, fe[], normFx, time()-t0, x))
        norm(d) * 10 < eps && (_pcb_done!(cb, true); return SolverResult(true, k, fe[], normFx, time()-t0, x))
    end
    _pcb_done!(cb, false)
    return SolverResult(false, maxiter, fe[], normFx, time()-t0, x)
end

# ═══════════════════════════════════════════════════════════════════════════════
# STTDFPM (Ibrahim et al. 2023)
# ═══════════════════════════════════════════════════════════════════════════════

function solve(m::STTDFPMMethod, prob::TestProblem, x0::Vector{Float64};
               eps=1e-11, maxiter=2000, cb::ProgressCallback=ProgressCallback(), kwargs...)
    t0 = time()
    fe = Ref(0)
    F(x) = (fe[] += 1; prob.G(x))

    x = copy(x0); Fx = F(x); normFx = norm(Fx)
    normFx < eps && (_pcb_done!(cb, true); return SolverResult(true, 0, fe[], normFx, time()-t0, x))

    d = -Fx

    for k in 1:maxiter
        _pcb(cb, k, normFx)
        _isnan_bail(normFx, d) && (_pcb_done!(cb, false); return SolverResult(false, k, fe[], NaN, time()-t0, x))
        norm(d) * 10 < eps && (_pcb_done!(cb, true); return SolverResult(true, k, fe[], normFx, time()-t0, x))

        dd = dot(d, d); tau = 1.0; z = x; Fz = Fx; ls_ok = false
        for j in 0:50
            tau = m.beta_ls^j
            z = x .+ tau .* d; Fz = F(z)
            proj_val = clamp_scalar(norm(Fz), m.eta1, m.eta2)
            if -dot(Fz, d) >= m.sigma * tau * proj_val * dd; ls_ok = true; break; end
            tau <= 1e-5 && break
        end
        ls_ok || (_pcb_done!(cb, false); return SolverResult(false, k, fe[], normFx, time()-t0, x))

        normFz = norm(Fz)
        if normFz < eps; _pcb_done!(cb, true); return SolverResult(true, k, fe[], normFz, time()-t0, z); end

        mu_k = dot(Fz, x .- z) / normFz^2
        x_new = prob.proj(x .- m.gamma .* mu_k .* Fz)
        Fx_new = F(x_new)

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

        x = x_new; Fx = Fx_new; normFx = norm(Fx)
        normFx < eps && (_pcb_done!(cb, true); return SolverResult(true, k, fe[], normFx, time()-t0, x))
    end
    _pcb_done!(cb, false)
    return SolverResult(false, maxiter, fe[], normFx, time()-t0, x)
end
