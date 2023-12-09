using LinearAlgebra
using CUDA
using PrettyTables

include("utils.jl")
include("../../timing/dphpc_timing.jl")

DEV = true
TIME = false
DEBUG = false


function main()
    correctness_check(true, ["S", "M"])
    run_benchmarks(cuda=true)
end


function kernel(L, x, b)
    N = length(x)
    for i in 1:N
        dp = CUDA.dot(L[i, 1:i-1], x[1:i-1])
        CUDA.@allowscalar x[i] = (b[i] - dp) / L[i, i]
    end

    return x
end

function cuda_kernel(L, x, b)
    CUDA.@allowscalar pretty_table(L)
    CUDA.@allowscalar pretty_table(x)
    CUDA.@allowscalar pretty_table(b)

    N = length(x)
    dp = CUDA.zeros(N)
    @cuda threads=N cuda_kernel_helper(L, x, b, dp)
    println(dp)
    return x
end

function cuda_kernel_helper(L, x, b, dp)
    i = threadIdx().x

    if i <= length(x)
        local_dot = 0.0
        for k in 1:i-1
            local_dot += L[i, k] * x[k]
        end

        dp[i] = local_dot

        x[i] = (b[i] - local_dot) / L[i, i]
        # x[i] = (b[i] - dot(L[i, 1:i-1], x[1:i-1])) / L[i, i]
    end
    return
end

main()