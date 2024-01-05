include("../../timing/dphpc_timing.jl")

using CUDA 

const alpha = 1.5
const beta = 1.2

include("_init_matrices_gpu.jl")

function gemm_kernel(N, M, K, A, B, C)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    j = threadIdx().y + (blockIdx().y - 1) * blockDim().y

    if i <= N && j <= M
        C[i, j] *= beta
        for k = 1:K
            C[i, j] += alpha * A[i, k] * B[k, j]
        end
    end
    nothing
end

function run_gemm_kernel(N, M, K, A, B, C)
    threadsPerBlock = (16, 16)
    numBlocks = ((N - 1) ÷ 16 + 1, (M - 1) ÷ 16 + 1)

    @cuda threads=threadsPerBlock blocks=numBlocks gemm_kernel(N, M, K, A, B, C)
    CUDA.synchronize()
end

include("_main.jl")

main()


