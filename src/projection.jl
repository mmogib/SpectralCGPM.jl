# projection.jl — Projection operators for convex sets

# Projection onto interval [a, b]
clamp_scalar(x, a, b) = max(a, min(x, b))

# Spectral parameter projection: Pi_{[amin, amax]}(x)
spectral_proj(x, amin, amax) = clamp_scalar(x, amin, amax)
