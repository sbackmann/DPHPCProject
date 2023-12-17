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

    N = size(data, 1)
    if i <= M && j <= i

        local_dot = 0.0

        for k in 1:N
            local_dot += data[k, i] * data[k, j]
        end
        cov[i,j] = cov[j,i] = local_dot / (N - 1.0)
    end

    return
end


function kernel(M, N, data)
    threads = 16
    threads_per_block = (threads, threads)
    blocks = (ceil(Int, M / threads), ceil(Int, M / threads))

    mean_data = CUDA.sum(data, dims=1) / N

    data .-= mean_data
    cov = CUDA.zeros(eltype(data), M, M)
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
