# problems.jl — 20 test problems for nonlinear monotone equations

# Projection: Gamma = R^n_+ = {x >= 0} (used for most problems, matching oldcode)
proj_Rn_plus(x) = max.(x, 0.0)

# Projection: Gamma = [1, inf)^n (for P18, minimal function requires x > 0)
proj_box_one(x) = max.(x, 1.0)

# ── Problem functions (defined at top level to avoid closure name collisions) ──

G1(x)  = 2.0 .* x .- sin.(x)
G2(x, n) = log.(x .+ 1.0) .- x ./ n
G3(x)  = exp.(x) .- 1.0

function G4(x)
    m = length(x)
    out = similar(x)
    for i in 1:m
        xprev = i > 1 ? x[i-1] : 0.0
        xnext = i < m ? x[i+1] : x[i-1]
        out[i] = 4x[i] + (xnext - 2x[i]) - xprev^2 / 3
    end
    return out
end

function G5(x)
    m = length(x)
    out = similar(x)
    for i in 1:m
        if i == 1
            s = (x[1] + x[2]) / (m + 1)
        elseif i == m
            s = (x[m-1] + x[m]) / (m + 1)
        else
            s = (x[i-1] + x[i] + x[i+1]) / (m + 1)
        end
        out[i] = x[i] - exp(cos(s))
    end
    return out
end

function G6(x)
    m = length(x)
    out = similar(x)
    out[1] = x[1] + sin(x[1]) - 1.0
    for i in 2:m-1
        out[i] = -x[i-1] + 2x[i] + sin(x[i]) - 1.0
    end
    out[m] = x[m] + sin(x[m]) - 1.0
    return out
end

function G7(x)
    m = length(x)
    out = similar(x)
    s = sum(x .^ 2)
    out[1] = s
    for i in 2:m
        out[i] = -2.0 * x[1] * x[i]
    end
    return out
end

function G8(x)
    m = length(x)
    out = similar(x)
    out[1] = x[1] * (x[1]^2 + x[2]^2) - 1.0
    for i in 2:m-1
        out[i] = x[i] * (x[i-1]^2 + 2x[i]^2 + x[i+1]^2) - 1.0
    end
    out[m] = x[m] * (x[m-1]^2 + x[m]^2)
    return out
end

function G9(x)
    m = length(x)
    out = similar(x)
    out[1] = 3x[1]^3 + 2x[2] - 5 + sin(abs(x[1]-x[2]))*sin(abs(x[1]+x[2]))
    for i in 2:m-1
        out[i] = -x[i-1]*exp(x[i-1]-x[i]) + x[1]*(4+3x[i]^3) + 2x[i+1] - 5 +
                 sin(abs(x[i]-x[i+1]))*sin(abs(x[i]+x[i+1]))
    end
    out[m] = -x[m-1]*exp(x[m-1]-x[m]) + 4x[m] - 3
    return out
end

G10(x) = (x .- 1.0).^2 .- 1.01
G11(x, n) = [(i/n)*exp(x[i]) - 1.0 for i in 1:length(x)]
G12(x) = x .- sin.(abs.(x .- 1.0))
G13(x) = 2.0 .* x .- sin.(abs.(x .- 1.0))
G14(x) = x .- 2.0 .* sin.(abs.(x .- 1.0))
G15(x) = exp.(x).^2 .+ 3.0 .* sin.(x) .* cos.(x) .- 1.0

function G16(x)
    m = length(x)
    out = similar(x)
    out[1] = 2.5x[1] + x[2] - 1.0
    for i in 2:m-1
        out[i] = x[i-1] + 2.5x[i] + x[i+1] - 1.0
    end
    out[m] = x[m-1] + 2.5x[m] - 1.0
    return out
end

G17(x) = 2.0 .* x .- sin.(abs.(x))

function G18(x)
    lx = log.(x)
    ex = exp.(x)
    return 0.5 .* (lx .+ ex .- sqrt.((lx .- ex).^2 .+ 1e-10))
end

function G19(x)
    s = sum(x .^ 2)
    return 2e-5 .* (x .- 1.0) .+ 4.0 .* x .* s .- x
end

function G20(x)
    m = length(x)
    s = sum(x) / m
    return [x[i] * cos(x[i] - 1/m) * (sin(x[i]) - 1 - (1 - x[i])^2 - s) for i in 1:m]
end

# ── Problem lookup ─────────────────────────────────────────────────────────────

function get_problem(id::Int, n::Int)
    P = proj_Rn_plus  # Gamma = R^n_+ for all problems (matching oldcode)
    problems = Dict(
        1  => TestProblem(1,  "P1",  G1,              P, "sabi2020two-4.1"),
        2  => TestProblem(2,  "P2",  x -> G2(x, n),   P, "la2006spectral-10"),
        3  => TestProblem(3,  "P3",  G3,              P, "zheng2020conjugate-4.1"),
        4  => TestProblem(4,  "P4",  G4,              P, "sabi2020two-4.5"),
        5  => TestProblem(5,  "P5",  G5,              P, "zheng2020conjugate-4.4a"),
        6  => TestProblem(6,  "P6",  G6,              P, "zheng2020conjugate-4.4b"),
        7  => TestProblem(7,  "P7",  G7,              P, "la2006spectral-19"),
        8  => TestProblem(8,  "P8",  G8,              P, "song2024efficient-14"),
        9  => TestProblem(9,  "P9",  G9,              P, "la2006spectral-12"),
        10 => TestProblem(10, "P10", G10,             P, "song2024efficient-2"),
        11 => TestProblem(11, "P11", x -> G11(x, n),  P, "song2024efficient-4"),
        12 => TestProblem(12, "P12", G12,             P, "ibrahim2024two-11"),
        13 => TestProblem(13, "P13", G13,             P, "waziri2022two-4.5"),
        14 => TestProblem(14, "P14", G14,             P, "song2024efficient-6"),
        15 => TestProblem(15, "P15", G15,             P, "song2024efficient-11"),
        16 => TestProblem(16, "P16", G16,             P, "waziri2020descent-5"),
        17 => TestProblem(17, "P17", G17,             P, "zhou2007limited-1"),
        18 => TestProblem(18, "P18", G18,             proj_box_one, "la2006spectral-32"),
        19 => TestProblem(19, "P19", G19,             P, "li2021scaled-4.11"),
        20 => TestProblem(20, "P20", G20,             P, "sabi2023modified-4.6"),
    )
    haskey(problems, id) || error("Unknown problem id: $id")
    return problems[id]
end

# ── Initial points ─────────────────────────────────────────────────────────────

function get_initial_points(n::Int)
    e = ones(n)
    return [
        (0.4  .* e, "0.4"),         # small constant
        (0.5  .* e, "0.5"),         # moderate constant
        (0.6  .* e, "0.6"),         # moderate constant
        (0.8  .* e, "0.8"),         # near-one constant
        (1.0  .* e, "1.0"),         # standard starting point
        (1.1  .* e, "1.1"),         # slightly above one
        (2.0  .* e, "2.0"),         # farther from solution
        (5.0  .* e, "5.0"),         # far from solution
        ([1.0/i for i in 1:n], "1/k"),       # decaying: 1, 1/2, 1/3, ...
        ([i/n for i in 1:n], "k/n"),         # ramp: 1/n, 2/n, ..., 1
    ]
end

const PROBLEM_IDS = [1,2,3,4,5,6,8,10,11,12,13,14,15,16,17,18,19,20]  # F7 (zero-Jacobian), F9 (Trigexp, too expensive) skipped
const NUM_PROBLEMS = length(PROBLEM_IDS)
const DIMENSIONS = [1_000, 5_000, 10_000, 50_000, 100_000, 120_000]
