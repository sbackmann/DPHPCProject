using Statistics
using LinearAlgebra
using PrettyTables
using DelimitedFiles
using CUDA

include("../../timing/dphpc_timing.jl")
DEV = true
TIME = false
ALLOW_SCALAR = false

eval_benchmarks = Dict(
    "S" => (500, 600),
    "M" => (1400, 1800),
    "L" => (3200, 4000),
    "paper" => (1200, 1400)
)

dev_benchmarks = Dict(
    # "S" => (2, 2),
    "M" => (5, 7),
    # "L" => (10, 20),
)

function main() 
    if DEV
        benchmark_sizes = dev_benchmarks
    else
        benchmark_sizes = eval_benchmarks
    end

    for (preset, dims) in benchmark_sizes
        println("Benchmarking $preset")
        println("Dims are $dims")
        M, N = dims

        data = initialize(M, N) 

        if TIME
            res = @dphpc_time(nothing, kernel(M, N, data), "missing")
        else
            res = CUDA.@sync kernel(M, N, data)
            CUDA.@allowscalar pretty_table(res)
        end
    end
end

function dot_prod_store_kernel(M, data, cov)
    i = threadIdx().x  # Get the thread index in the x direction
    j = threadIdx().y  # Get the thread index in the y direction

    N = size(data, 1)
    if i <= M && j <= M

        local_dot = 0.0

        # local_dot = dot(data[:, i], data[:, j])
        for k in 1:N
            local_dot += data[k, i] * data[k, j]
        end
        cov[j,i] = local_dot / (N - 1.0)
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

function initialize(M, N, datatype=Float64)
    return CuArray([
        datatype((i-1) * (j-1)) / M for i in 1:N, j in 1:M
        ])
end


main()
