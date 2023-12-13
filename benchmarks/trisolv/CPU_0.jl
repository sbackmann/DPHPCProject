using LinearAlgebra
using PrettyTables

include("utils.jl")
include("../../timing/dphpc_timing.jl")

DEV = true
TIME = true
DEBUG = false

function kernel(L, x, b)
    println("Running kernel...")
    N = length(x)
    for i in 1:N
        # println(dot(L[i, 1:i-1], x[1:i-1]))
        x[i] = (b[i] - dot(L[i, 1:i-1], x[1:i-1])) / L[i, i]
    end
    return x
end

function main()
    correctness_check(false, ["S", "M", "paper"])
    println("Running benchmarks...")

    # N = 16000
    # print(@dphpc_time(data = initialize(N), kernel(data...), "paper"))
    run_benchmarks()
end

main()