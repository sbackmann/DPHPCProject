using CUDA
using BenchmarkTools
include("../../../timing/dphpc_timing.jl")

function dot_prod_store_kernel(M, data, cov)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if i <= M && j <= M

        local_dot = 0.0
        for k in 1:M
            local_dot += data[k, i] * data[k, j]
        end

        cov[j, i] = local_dot 
    end

    return
end

function kernel(data, cov, M)
    threads = 16
    threads_per_block = (threads, threads)
    blocks = (ceil(Int, M / threads), ceil(Int, M / threads))

    CUDA.@sync blocking=true (@cuda threads=threads_per_block blocks=blocks dot_prod_store_kernel(M, data, cov))
end


function alt_kernel(data, cov, M)
    return data * data
end

function main()
    M = 1400  
    data = CUDA.fill(2.0, M, M)
    # cov = CUDA.zeros(eltype(data), M, M)
    cov = zeros(eltype(data), M, M)

    # print(@dphpc_time(nothing, kernel(data, cov, M), "missing"))
    print(@dphpc_time(nothing, alt_kernel(data, cov, M), "missing"))
end

main()
