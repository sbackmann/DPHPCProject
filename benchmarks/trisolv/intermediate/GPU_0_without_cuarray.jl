using LinearAlgebra
using CUDA
using PrettyTables

include("utils.jl")
include("../../timing/dphpc_timing.jl")

DEV = true
TIME = false
DEBUG = false


function main()
    correctness_check(true, ["S", "M"])
    run_benchmarks(cuda=false)
end


function kernel(L, x, b)
    N = length(x)
    for i in 1:N
        dp = CUDA.dot(L[i, 1:i-1], x[1:i-1])
        CUDA.@allowscalar x[i] = (b[i] - dp) / L[i, i]
    end

    return x
end


main()