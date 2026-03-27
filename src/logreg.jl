# logreg.jl — Logistic regression problem builder for LIBSVM datasets
#
# Formulation:
#   min_{x ∈ Γ}  f(x) = (1/N) Σ log(1 + exp(-bᵢ aᵢᵀx)) + (μ/2)||x||²
#   G(x) = ∇f(x) = -(1/N) Σ bᵢaᵢ / (1 + exp(bᵢ aᵢᵀx)) + μx
#   Γ = [-C, C]^n  (box constraint)

using Statistics: mean, std

# ── Load CSV (pre-converted by s20_libsvm_to_csv.jl) ─────────────────────────

function load_libsvm_csv(path::String)
    df = CSV.read(path, DataFrame)
    b = Float64.(df.label)
    A = Matrix{Float64}(df[:, 2:end])
    return A, b
end

# ── Feature normalization ────────────────────────────────────────────────────

function normalize_features!(A::Matrix{Float64})
    _N, n = size(A)
    for j in 1:n
        col = @view A[:, j]
        μ = mean(col)
        σ = std(col; corrected=false)
        if σ > 1e-12
            col .= (col .- μ) ./ σ
        else
            col .= 0.0
        end
    end
    return A
end

# ── Numerically stable 1/(1 + exp(t)) ───────────────────────────────────────
# σ(-t): safe for both large positive and negative t.

function _inv_logistic!(s::Vector{Float64}, t::Vector{Float64})
    @inbounds for i in eachindex(t)
        ti = t[i]
        if ti >= 0
            e = exp(-ti)
            s[i] = e / (1.0 + e)
        else
            e = exp(ti)
            s[i] = 1.0 / (1.0 + e)
        end
    end
    return s
end

# ── Problem builder ──────────────────────────────────────────────────────────

function make_logreg_problem(A::Matrix{Float64}, b::Vector{Float64};
                              mu::Float64=0.1, C::Float64=10.0)
    N, n = size(A)
    inv_N = 1.0 / N

    # Precompute bA[i,:] = bᵢ * aᵢᵀ  (N × n matrix)
    bA = b .* A

    # Preallocate work vectors (avoid allocations in G)
    _bAx = Vector{Float64}(undef, N)
    _s   = Vector{Float64}(undef, N)

    function G(x::Vector{Float64})
        mul!(_bAx, bA, x)                  # _bAx = bA * x = [b₁a₁ᵀx, ..., bₙaₙᵀx]
        _inv_logistic!(_s, _bAx)           # _s[i] = 1/(1 + exp(bᵢaᵢᵀx))
        return -(inv_N) * (bA' * _s) + mu * x
    end

    proj(x::Vector{Float64}) = clamp.(x, -C, C)

    prob = TestProblem(0, "LogReg", G, proj, "logreg")
    return prob, n
end
