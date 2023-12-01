using Statistics
using LinearAlgebra
using PrettyTables
using DelimitedFiles
using CUDA

include("utils.jl")
include("../../timing/dphpc_timing.jl")


function dot_prod_store_kernel(M, data, cov)
    i = threadIdx().x
    j = threadIdx().y
    N = size(data, 1)

    if i <= M && j <= M

        local_dot = 0.0

        # Unrolled loop
        for k in 1:4:N-3
            local_dot += data[k, i] * data[k, j]
            local_dot += data[k+1, i] * data[k+1, j]
            local_dot += data[k+2, i] * data[k+2, j]
            local_dot += data[k+3, i] * data[k+3, j]
        end

        # Handle the remaining elements
        for k in (N - rem(N, 4) + 1):N
            local_dot += data[k, i] * data[k, j]
        end

        cov[j, i] = local_dot / (N - 1.0)
    end

    return
end


function kernel(M, float_n, data)
    threads = 16
    threads_per_block = (threads, threads)
    blocks = (ceil(Int, M / threads), ceil(Int, M / threads))

    # TODO maybe faster if use custom kernel
    mean_data = CUDA.sum(data, dims=1) / float_n

    data .-= mean_data
    cov = CUDA.zeros(eltype(data), M, M)
    @cuda threads=threads_per_block blocks=blocks dot_prod_store_kernel(M, data, cov)

    return cov 
end

function alt_kernel(M, float_n, data)
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
