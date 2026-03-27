# includes.jl — Single entry point for all source files
# Usage: include("src/includes.jl") from scripts or REPL at jcode/ directory
#
# Include order matters: each file may depend on files included before it.

include("deps.jl")
include("types.jl")
include("problems.jl")
include("projection.jl")
include("utils.jl")
include("solvers.jl")
include("benchmark.jl")
include("logreg.jl")
