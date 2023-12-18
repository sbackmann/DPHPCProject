using Statistics
using LinearAlgebra
using PrettyTables
using DelimitedFiles
using CUDA

include("utils.jl")
include("../../timing/dphpc_timing.jl")

VALIDATE = false

function dot_prod_store_kernel(M, data, cov)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    N = size(data, 2)
    if i <= M && j <= i

        dot1 = 0.0
        dot2 = 0.0
        dot3 = 0.0
        dot4 = 0.0

        @inbounds @simd for k in 1:4:N-3
            dot1 += data[i, k  ] * data[j, k  ]
            dot2 += data[i, k+1] * data[j, k+1]
            dot3 += data[i, k+2] * data[j, k+2]
            dot4 += data[i, k+3] * data[j, k+3]
        end

        cov[i, j] = (dot1 + dot2 + dot3 + dot4) / (N - 1.0)
        cov[j, i] = cov[i, j]
    end

    return
end

function transpose_kernel(transposed, orig)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    N = size(orig, 1)
    M = size(orig, 2)
    @inbounds if i <= N && j <= M
        transposed[j,i] = orig[i,j]
    end

    return
end

function kernel(M, N, orig_data)
    threads = 16
    threads_per_block = (threads, threads)
    blocks = (ceil(Int, N / threads), ceil(Int, M / threads))
    data = CUDA.zeros(eltype(orig_data), M, N)
    @cuda threads=threads_per_block blocks=blocks transpose_kernel(data, orig_data)

    threads = 16
    threads_per_block = (threads, threads)
    blocks = (ceil(Int, M / threads), ceil(Int, M / threads))
    
    mean_data = CUDA.sum(data, dims=2) / N

    data .-= mean_data

    cov = CUDA.zeros(eltype(orig_data), M, M)
    CUDA.@sync blocking=true (@cuda threads=threads_per_block blocks=blocks dot_prod_store_kernel(M, data, cov))

    return cov 
end


function main()
    if VALIDATE
        data = initialize(3,4, cuda=true)
        covar = kernel(3, 4, data)
        println("Got")
        CUDA.@allowscalar pretty_table(covar)
        println("Expected")
        pretty_table(cov(initialize(3,4)))

        correctness_check(true, ["S", "M"])
    end

    run_benchmarks(cuda = true)
end



main()
