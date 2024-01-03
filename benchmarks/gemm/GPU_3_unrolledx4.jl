include("../../timing/dphpc_timing.jl")
include("./validation.jl")

using CUDA 

# eliminate global variables + accumulators + unroll by x4 

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
        acc1 = C[i, j] * beta
        acc2 = 0.0

        acc3 = 0.0
        acc4 = 0.0
        acc5 = 0.0
        acc6 = 0.0

        # Unroll the loop by 4
        for k = 1:4:K-3
            acc3 += alpha * A[i, k] * B[k, j]
            acc4 += alpha * A[i, k+1] * B[k+1, j]
            acc5 += alpha * A[i, k+2] * B[k+2, j]
            acc6 += alpha * A[i, k+3] * B[k+3, j]
        end

        # Handle the remaining values
        for k = (K - rem(K, 4)) + 1:K
            acc2 += alpha * A[i, k] * B[k, j]
        end

        C[i, j] = acc1 + acc2 + acc3 + acc4 + acc5 + acc6
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


