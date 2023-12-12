using Statistics
using LinearAlgebra
using PrettyTables
using DelimitedFiles
using CUDA

include("utils.jl")
include("../../timing/dphpc_timing.jl")


function mean_adjust_kernel(data, N, M)
    col_idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x

    # TODO optimize this stride
    @inbounds for j = col_idx:stride:M
        local_sum = 0.0
        @inbounds for row_idx = 1:N
            local_sum += data[row_idx, j]
        end
        mean_j = local_sum / N

        @inbounds for row_idx = 1:N
            data[row_idx, j] -= mean_j
        end
    end
    return
end

function normalise_kernel(M, cov, normalising_factor)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if i <= M && j <= M
        cov[j, i] *= normalising_factor
    end

    return
end

function kernel(M, N, data)
    threads_per_block = 256
    blocks = div(M + threads_per_block - 1, threads_per_block)

    @cuda threads=threads_per_block blocks=blocks mean_adjust_kernel(data, N, M)

    cov = CUDA.zeros(eltype(data), M, M)
    cov = transpose(data) * data

    threads = 16
    threads_per_block = (threads, threads)
    blocks = (ceil(Int, M / threads), ceil(Int, M / threads))

    normalising_factor = 1 / (N - 1.0)    
    CUDA.@sync blocking=true (@cuda threads=threads_per_block blocks=blocks normalise_kernel(M, cov, normalising_factor))

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
