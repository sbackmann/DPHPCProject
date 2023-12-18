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
    correctness_check(true, ["S"])
    run_benchmarks(cuda=true)
end


function pre_comp_kernel(L, Lx, x, j, N)
    start_row = j

    i = (blockIdx().x - 1) * blockDim().x + (threadIdx().x - 1 ) + start_row
    if i <= N
        Lx[i, j] = L[i, j] * x[j] + Lx[i, j - 1]
    end

    return
end

function scalar_mult_kernel(L, Lx, scalar, j, N)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x 
    if i <= N
        Lx[i, j] = L[i, j] * scalar
    end

    return
end


function kernel(L, x, b)
    N = length(x)
    t = 256
    blocks = ceil(Int, N / t)

    Lx = CUDA.zeros(eltype(L), N, N)

    CUDA.@allowscalar x[1] = (b[1]) / L[1, 1] 
    CUDA.@allowscalar Lx[:, 1] = L[:, 1] * x[1]
    
    for i in 2:N
        CUDA.@allowscalar dp = Lx[i, i-1]  #  CUDA.dot(L[i, 1:i-1], x[1:i-1])
        CUDA.@allowscalar x[i] = (b[i] - dp) / L[i, i]
        
        blocks = ceil(Int, (N - i + 1) / t)
        @cuda threads=t blocks=blocks pre_comp_kernel(L, Lx, x, i, N)
    end

    return x
end


main()

