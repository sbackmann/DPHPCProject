using Statistics
using LinearAlgebra
using PrettyTables
using DelimitedFiles
using CUDA

include("utils.jl")
include("../../timing/dphpc_timing.jl")


function dot_prod_store_kernel(M, data, cov)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    N = size(data, 2)
    if i <= M && j <= M

        local_dot = 0.0

        @inbounds @simd for k in 1:4:N-3
            local_dot += data[i, k  ] * data[j, k  ]
            local_dot += data[i, k+1] * data[j, k+1]
            local_dot += data[i, k+2] * data[j, k+2]
            local_dot += data[i, k+3] * data[j, k+3]
        end

        # Handle the remaining elements
        @inbounds for k in (N - rem(N, 4) + 1):N
            local_dot += data[i, k] * data[i, k]
        end

        cov[j,i] = local_dot / (N - 1.0)
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
    

    # TODO maybe faster if use custom kernel

    mean_data = CUDA.sum(data, dims=2) / N

    data .-= mean_data
    cov = CUDA.zeros(eltype(data), M, M)
    CUDA.@sync blocking=true (@cuda threads=threads_per_block blocks=blocks dot_prod_store_kernel(M, data, cov))

    return cov 
end


function main()
    data = initialize(3,4, cuda=true)
    covar = kernel(3, 4, data)
    println("Got")
    CUDA.@allowscalar pretty_table(covar)
    println("Expected")
    pretty_table(cov(initialize(3,4)))

    run_benchmarks(cuda = true)
end



main()
