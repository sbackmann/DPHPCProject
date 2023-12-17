using LinearAlgebra
using CUDA
using PrettyTables

include("utils.jl")
include("../../timing/dphpc_timing.jl")

DEV = true
TIME = false
DEBUG = false


function main()
    correctness_check(true, ["S"])
    run_benchmarks(cuda=true)
end

function dot_product_kernel(L, x, dp, i, N)
    j = threadIdx().x + (blockIdx().x - 1) * blockDim().x

    if j < i
        dp[j] = L[j, i] * x[j]
        sync_threads()

        stride = 1
        while j + stride < i
            if (j % (2 * stride)) == 1
                dp[j] += dp[j + stride]
            end
            sync_threads()
        
            stride <<= 1  
        end
        
    end
    return
end

function scalar_update_kernel(x, i, b, dp, L)
    x[i] = (b[i] - dp[1]) / L[i, i]
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

function kernel(L, x, b)
    N = length(x)

    dp = CUDA.zeros(N)

    threads = 16
    threads_per_block = (threads, threads)
    blocks = (ceil(Int, N / threads), ceil(Int, N / threads))
    Lt = CUDA.zeros(N, N)
    @cuda threads=threads_per_block blocks=blocks transpose_kernel(Lt, L)

    
    thr = 256

    for i in 1:N
        # dp = CUDA.dot(L[i, 1:i-1], x[1:i-1])
        blo = Int(ceil( i / thr))
        @cuda threads=thr blocks=blo dot_product_kernel(Lt, x, dp, i, N)
        @cuda threads=1 blocks=1 scalar_update_kernel(x, i, b, dp, Lt)
    end

    CUDA.synchronize()
    return x
end


main()