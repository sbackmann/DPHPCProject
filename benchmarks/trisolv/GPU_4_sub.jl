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
        # if i == j-1
        #     Lx[i, j] = L[i, j] * x[j] + Lx[i, j - 1]
        # else
        #     Lx[i, j] = L[i, j] * x[j] + Lx[i, j - 1]
        # end
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

function inv_diag_kernel(matrix, diag, N)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x 
    if i <= N
        diag[i] = 1 / matrix[i, i]
    end

    return
end

function kernel(L, x, b)
    N = length(x)
    t = 256
    blocks = ceil(Int, N / t)
    
    inv_diag = CUDA.zeros(eltype(L), N)
    @cuda threads=t blocks=blocks inv_diag_kernel(L, inv_diag, N)

    b_prod_inv_diag = CUDA.zeros(eltype(L), N)
    b_prod_inv_diag = b .* inv_diag

    Lx = CUDA.zeros(eltype(L), N, N)

    x = CuArray(b_prod_inv_diag)
    CUDA.@allowscalar Lx[:, 1] = L[:, 1] * b_prod_inv_diag[1]
    
    for i in 2:N
        CUDA.@allowscalar x[i] -= (Lx[i, i-1] * inv_diag[i])
        
        blocks = ceil(Int, (N - i + 1) / t)
        @cuda threads=t blocks=blocks pre_comp_kernel(L, Lx, x, i, N)
    end

    return x
end


main()
