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


function compute_dp(L, x, i)
    col_idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x

    dp = 0.0
    for j in col_idx:stride:i-1
        dp += L[i, j] * x[j]
    end
    # x[i] += dp # tmp store in x[i]
    attomicAdd!(x[i], dp)

    return
end

function kernel(L, x, b)
    N = length(x)
    threads_per_block = 256
    blocks = div(N + threads_per_block - 1, threads_per_block)

    res = CUDA.@profile begin
        for i in 1:N
            @cuda threads=threads_per_block blocks=blocks compute_dp(L, x, i) # tmp store in x[i]
            # dp = CUDA.dot(L[i, 1:i-1], x[1:i-1])
            x[i] = (b[i] - x[i]) / L[i, i]
        end
    end

    println(res)

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