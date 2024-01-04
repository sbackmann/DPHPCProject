include("../../timing/dphpc_timing.jl")
include("./validation.jl")

using CUDA 

# eliminate global variables + accumulators + unroll by x8 + inbounds

validation = false

# for validation 
function initialize_matrices_val(N, M, K)
    A = fill(0.5, N, K)
    B = fill(0.7, K, M)
    C = fill(0.3, N, M)
    return CuArray(A), CuArray(B), CuArray(C)
end

include("_init_matrices_gpu.jl")

function gemm_kernel(N, M, K, A, B, C)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    j = threadIdx().y + (blockIdx().y - 1) * blockDim().y

    alpha = 1.5
    beta = 1.2

    if i <= N && j <= M
        @inbounds C[i, j] *= beta
        acc1 = 0.0  
        acc2 = 0.0
        acc3 = 0.0
        acc4 = 0.0
        acc5 = 0.0
        acc6 = 0.0
        acc7 = 0.0
        acc8 = 0.0
        acc9 = 0.0

        for k = 1:8:K-7
            @inbounds acc1 += alpha * A[i, k] * B[k, j]
            @inbounds acc2 += alpha * A[i, k+1] * B[k+1, j]
            @inbounds acc3 += alpha * A[i, k+2] * B[k+2, j]
            @inbounds acc4 += alpha * A[i, k+3] * B[k+3, j]
            @inbounds acc5 += alpha * A[i, k+4] * B[k+4, j]
            @inbounds acc6 += alpha * A[i, k+5] * B[k+5, j]
            @inbounds acc7 += alpha * A[i, k+6] * B[k+6, j]
            @inbounds acc8 += alpha * A[i, k+7] * B[k+7, j]
        end

        for k = (K - rem(K, 8)) + 1:K
            @inbounds acc9 += alpha * A[i, k] * B[k, j]
        end

        @inbounds C[i, j] += (acc1 + acc2 + acc3+ acc4 + acc5+ acc6+ acc7+ acc8+ acc9)
    end
    return
end

function run_gemm_kernel(N, M, K, A, B, C)
    threadsPerBlock = (16, 16)
    numBlocks = ((N - 1) ÷ 16 + 1, (M - 1) ÷ 16 + 1)
    @cuda threads=threadsPerBlock blocks=numBlocks gemm_kernel(N, M, K, A, B, C)
    CUDA.synchronize()
end


include("_main.jl")

main()


