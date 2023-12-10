using LinearAlgebra
using CUDA
using PrettyTables

include("utils.jl")
include("../../timing/dphpc_timing.jl")

DEV = true
TIME = false
DEBUG = false


function main()
    # correctness_check(true, ["S", "M"])
    correctness_check(false, ["S"])
    run_benchmarks(cuda=false)
end


function pre_comp_kernel(L, Lx, x, j, N)
    start_row = j

    for i in start_row:N # cuda this; skip j=1
        CUDA.@allowscalar Lx[i, j] = L[i, j] * x[j] + Lx[i, j - 1]
    end

    return
end


function kernel(L, x, b)
    N = length(x)

    # Lx = CUDA.zeros(eltype(L), N, N)
    Lx = zeros(eltype(L), N, N)

    CUDA.@allowscalar x[1] = (b[1]) / L[1, 1] 
    for i in 1:N 
        CUDA.@allowscalar Lx[i, 1] = L[i, 1] * x[1] 
    end

    for i in 2:N
        CUDA.@allowscalar dp = Lx[i, i-1]  #  CUDA.dot(L[i, 1:i-1], x[1:i-1])
        CUDA.@allowscalar x[i] = (b[i] - dp) / L[i, i]
        pre_comp_kernel(L, Lx, x, i, N)
    end

    return x
end


main()