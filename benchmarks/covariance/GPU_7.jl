using Statistics
using LinearAlgebra
using PrettyTables
using DelimitedFiles
using CUDA

include("utils.jl")
include("../../timing/dphpc_timing.jl")

function ydot_prod_store_kernel(N, M, data, cov)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if i <= M && j <= M
        local_dot = 0.0

        @inbounds @simd for k in 1:4:N-3
            local_dot1 = data[k, i] * data[k, j]
            local_dot2 = data[k+1, i] * data[k+1, j]
            local_dot3 = data[k+2, i] * data[k+2, j]
            local_dot4 = data[k+3, i] * data[k+3, j]
            local_dot += local_dot1 + local_dot2 + local_dot3 + local_dot4
        end

        @inbounds for k in (N - rem(N, 4) + 1):N
            local_dot += data[k, i] * data[k, j]
        end

        cov[j, i] = local_dot / (N - 1.0)
    end

    return
end


function dot_prod_store_kernel(N, M, data, datat, cov)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if i <= M && j <= M
        local_dot = 0.0

        for k in 1:N
            local_dot += data[k, i] * data[k, j]
        end

        cov[j, i] = local_dot 
    end

    return
end

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

function normalize_kernel(cov, M, divider)
    col_idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x

    for j in col_idx:stride:M
        for i in 1:M
            cov[i, j] *= divider
        end
    end
    return
end

function transpose_kernel(A, B, m, n)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    if i <= m && j <= n
        B[i, j] = A[j, i]
    end
    return
end

    
function kernel(M, N, data)

    data_t = CUDA.zeros(eltype(data), M, M)
    cov = CUDA.zeros(eltype(data), M, M)
    x = CUDA.@profile begin
        threads_per_block = 256
        blocks = div(M + threads_per_block - 1, threads_per_block)
        CUDA.@sync(@cuda threads=threads_per_block blocks=blocks mean_adjust_kernel(data, N, M))

        # mean_data = CUDA.zeros(M)
        # mean_data = reshape(mean_data, (1, M)) # TODO could be expensive

        threads = 16
        threads_per_block = (threads, threads)
        blocks = (ceil(Int, M / threads), ceil(Int, M / threads))

        
        # data .-= mean_data
        CUDA.@sync(@cuda threads=threads_per_block blocks=blocks dot_prod_store_kernel(N, M, data, data, cov))

        divider = 1.0 / (N - 1.0)
        threads_per_block = 256
        blocks = div(M + threads_per_block - 1, threads_per_block)
        CUDA.@sync(@cuda threads=threads_per_block blocks=blocks normalize_kernel(cov, M, divider))
    end
    print(x)
    # prof = CUDA.@profile externel=true @cuda threads=threads_per_block blocks=blocks dot_prod_store_kernel(N, M, data, cov)
    # print(prof)
    # @cuda threads=threads_per_block blocks=blocks dot_prod_store_kernel(N, M, data, cov)

    # CUDA.@sync blocking=true ()

    return cov 
end


function main()
    data = initialize(3,4, cuda=true)
    covar = kernel(3, 4, data)
    println("Got")
    CUDA.@allowscalar pretty_table(covar)
    println("Expected")
    pretty_table(cov(initialize(3,4)))


    data = initialize(1400,1800, cuda=true)
    # print(CUDA.@profile kernel(1400, 1800, data))
    kernel(1400, 1800, data)

    # run_benchmarks(cuda = true)
end



main()
