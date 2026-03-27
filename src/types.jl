# types.jl — Type definitions for TwoGenDFM

# ── Problem definition ─────────────────────────────────────────────────────────

struct TestProblem
    id::Int
    name::String
    G::Function                # G: R^n -> R^n
    proj::Function             # projection onto Gamma
    source::String             # citation key
end

# ── Method types (dispatch on these) ───────────────────────────────────────────

abstract type AbstractMethod end

# --- Our method 1: GMOPCGM ---
struct GMOPCGMMethod <: AbstractMethod
    tau::Float64       # modification parameter in v_{k-1} = y_{k-1} + tau * s_{k-1}
    rho::Float64       # backtracking contraction
    beta::Float64      # initial step size for line search
    zeta::Float64      # line search parameter
    alpha_min::Float64 # spectral projection lower bound
    alpha_max::Float64 # spectral projection upper bound
    lambda0::Float64   # initial spectral parameter
    gamma::Float64     # projection relaxation parameter in (0,2)
    zeta1::Float64     # line search norm projection lower bound
    zeta2::Float64     # line search norm projection upper bound
end

function GMOPCGMMethod(; tau=1.0, rho=0.8, beta=0.5, zeta=1e-4,
        alpha_min=0.1, alpha_max=2.0, lambda0=1.0, gamma=1.1,
        zeta1=1.0, zeta2=1.0)
    GMOPCGMMethod(tau, rho, beta, zeta, alpha_min, alpha_max, lambda0, gamma, zeta1, zeta2)
end

# --- Our method 2: GCGPM ---
struct GCGPMMethod <: AbstractMethod
    tau::Float64
    rho::Float64
    eta::Float64       # initial step size for line search
    zeta::Float64
    alpha_min::Float64
    alpha_max::Float64
    lambda0::Float64
    gamma::Float64
    zeta1::Float64
    zeta2::Float64
end

function GCGPMMethod(; tau=0.001, rho=0.5, eta=0.6, zeta=0.1,
        alpha_min=0.55, alpha_max=4.9, lambda0=1.0, gamma=1.8,
        zeta1=1.0, zeta2=1.0)
    GCGPMMethod(tau, rho, eta, zeta, alpha_min, alpha_max, lambda0, gamma, zeta1, zeta2)
end

# --- Competitor: MOPCGM (Sabi'u et al. 2023, Algorithm 2.1) ---
# Parameters from Section 4.1: rho=0.1, eta=0.9, zeta=0.0001, lambda=0.1
struct MOPCGMMethod <: AbstractMethod
    rho::Float64
    eta::Float64
    zeta::Float64
    lambda::Float64    # parameter in y_{k-1} = F_{k+1} - F_k + lambda * s_k
end
MOPCGMMethod() = MOPCGMMethod(0.1, 0.9, 0.0001, 0.1)

# --- Competitor: CGPM (Zheng et al. 2020, Algorithm 2.1) ---
# Parameters from oldcode/CGPM.jl: σ₁=0.7, σ₂=0.3, β=1.0, ρ=0.6, σ=0.001
struct CGPMMethod <: AbstractMethod
    sigma1::Float64
    sigma2::Float64
    beta_ls::Float64   # initial step size (called β in paper)
    rho::Float64       # backtracking contraction (called ρ in paper)
    sigma::Float64     # line search parameter (called σ in paper)
end
CGPMMethod() = CGPMMethod(0.7, 0.3, 1.0, 0.6, 0.001)

# --- Competitor: STTDFPM (Ibrahim et al. 2023, Algorithm 1) ---
# Parameters from page 12: t=0.11, β=0.5, σ=0.01, γ=1.8,
#   α_min=1e-10, α_max=1e30, r=0.1, ψ=0.2, η₁=0.001, η₂=0.6
struct STTDFPMMethod <: AbstractMethod
    alpha_min::Float64 # spectral projection lower bound
    alpha_max::Float64 # spectral projection upper bound
    r_param::Float64   # parameter in s_{k-1} = x_k - x_{k-1} + r * y_{k-1}  (Eq 10)
    beta_ls::Float64   # backtracking base: τ_k = β^{i_k}
    sigma::Float64     # line search parameter (Eq 8)
    gamma::Float64     # projection relaxation in (0,2)
    psi::Float64       # denominator safeguard in v_k (Eq 11)
    eta1::Float64      # norm projection lower bound (Eq 8)
    eta2::Float64      # norm projection upper bound (Eq 8)
end
STTDFPMMethod() = STTDFPMMethod(1e-10, 1e30, 0.1, 0.5, 0.01, 1.8, 0.2, 0.001, 0.6)

# ── Solver result ──────────────────────────────────────────────────────────────

struct SolverResult
    converged::Bool
    iterations::Int
    f_evals::Int
    residual::Float64
    cpu_time::Float64          # CPU time in seconds
    x::Vector{Float64}
end

# ── Convenience ────────────────────────────────────────────────────────────────

method_name(::GMOPCGMMethod) = "GMOPCGM"
method_name(::GCGPMMethod) = "GCGPM"
method_name(::MOPCGMMethod) = "MOPCGM"
method_name(::CGPMMethod) = "CGPM"
method_name(::STTDFPMMethod) = "STTDFPM"
