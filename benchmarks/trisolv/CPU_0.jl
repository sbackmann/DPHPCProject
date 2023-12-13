using LinearAlgebra
using PrettyTables

include("utils.jl")
include("../../timing/dphpc_timing.jl")

DEV = true
TIME = true
DEBUG = false

function kernel(L, x, b)
    N = length(x)
    for i in 1:N
        # println(dot(L[i, 1:i-1], x[1:i-1]))
        x[i] = (b[i] - dot(L[i, 1:i-1], x[1:i-1])) / L[i, i]
    end
    return x
end

function main()
    correctness_check(false, ["S", "M"])
    run_benchmarks()
end

main()