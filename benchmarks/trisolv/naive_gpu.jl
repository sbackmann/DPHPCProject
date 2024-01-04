using LinearAlgebra
using CUDA
using PrettyTables

include("utils.jl")
include("../../timing/dphpc_timing.jl")



function dot_product_kernel(L, x, dp, i, N)
    j = threadIdx().x + (blockIdx().x - 1) * blockDim().x

    if j < i
        dp[j] = L[i, j] * x[j]
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

function kernel(L, x, b)
    N = length(x)

    dp = CUDA.zeros(N)
    thr = 256

    for i in 1:N
        blo = Int(ceil( i / thr))
        @cuda threads=thr blocks=blo dot_product_kernel(L, x, dp, i, N)
        @cuda threads=1 blocks=1 scalar_update_kernel(x, i, b, dp, L)
    end

    CUDA.synchronize()
    return x
end


main()