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
    N = size(data, 1)

    if i <= M && j <= M
        local_dot = 0.0

        @simd for k in 1:4:N-3
            local_dot1 = data[k, i] * data[k, j]
            local_dot2 = data[k+1, i] * data[k+1, j]
            local_dot3 = data[k+2, i] * data[k+2, j]
            local_dot4 = data[k+3, i] * data[k+3, j]
            local_dot += local_dot1 + local_dot2 + local_dot3 + local_dot4
        end

        for k in (N - rem(N, 4) + 1):N
            local_dot += data[k, i] * data[k, j]
        end

        cov[j, i] = local_dot / (N - 1.0)
    end

    return
end

function mean_kernel(data, mean_data, N, M)
    col_idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x

    for j = col_idx:stride:M
        local_sum = 0.0
        for row_idx = 1:N
            local_sum += data[row_idx, j]
        end
        mean_data[j] = local_sum / N
    end
    return
end

function kernel(M, N, data)

    threads_per_block = 256
    blocks = div(M + threads_per_block - 1, threads_per_block)

    mean_data = CUDA.zeros(M)
    @cuda threads=threads_per_block blocks=blocks mean_kernel(data, mean_data, N, M)
    mean_data = reshape(mean_data, (1, M)) # TODO could be expensive

    threads = 16
    threads_per_block = (threads, threads)
    blocks = (ceil(Int, M / threads), ceil(Int, M / threads))

    data .-= mean_data
    cov = CUDA.zeros(eltype(data), M, M)
    CUDA.@sync blocking=true (@cuda threads=threads_per_block blocks=blocks dot_prod_store_kernel(M, data, cov))

    return cov 
end

function alt_kernel(M, N, data)
    # for i in 1:M
    #     for j in 1:M
    #         CUDA.@allowscalar cov[j, i] = CUDA.dot(data[:, i], data[:, j]) 
    #     end
    # end
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
