include("../../timing/dphpc_timing.jl")

using CUDA 

include("_init_matrices_gpu.jl")

function gemm_kernel(N, M, K, A, B, C)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    j = threadIdx().y + (blockIdx().y - 1) * blockDim().y
    alpha = 1.5
    beta = 1.2

    if i <= M && j <= N
        C[j, i] *= beta
        for k = 1:K
            C[j, i] += alpha * A[j, k] * B[k, i]
        end
    end
    nothing
end

function run_gemm_kernel(N, M, K, A, B, C)
    threadsPerBlock = (16, 16)
    numBlocks = ((M - 1) ÷ 16 + 1, (N - 1) ÷ 16 + 1)

    @cuda threads=threadsPerBlock blocks=numBlocks gemm_kernel(N, M, K, A, B, C)
    CUDA.synchronize()
end


include("_main.jl")

main()

