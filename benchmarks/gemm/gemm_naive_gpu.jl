include("../../timing/dphpc_timing.jl")

using CUDA 

const alpha = 1.5
const beta = 1.2

function init_matrices(N, M, K)

    A = zeros(Float64, N, K)
    B = zeros(Float64, K, M)
    C = zeros(Float64, N, M)

    A = [(i*j+1) % K / K for i in 1:N, j in 1:K]
    B = [(i*j+1) % M / M for i in 1:K, j in 1:M]
    C = [(i*j+1) % M / M for i in 1:N, j in 1:M]

    return CuArray(A), CuArray(B), CuArray(C)

end

function gemm_kernel(N, M, K, A, B, C)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    j = threadIdx().y + (blockIdx().y - 1) * blockDim().y

    if i <= N && j <= M
        C[i, j] *= beta
        for k = 1:K
            C[i, j] += alpha * A[i, k] * B[k, j]
        end
    end
end

function run_gemm_kernel(N, M, K, A, B, C)
    threadsPerBlock = (16, 16)
    numBlocks = ((N - 1) ÷ 16 + 1, (M - 1) ÷ 16 + 1)

    @cuda threads=threadsPerBlock blocks=numBlocks gemm_kernel(N, M, K, A, B, C)
    CUDA.synchronize()

end


function main()


    N, M, K = 1000, 1100, 1200
    A, B, C = init_matrices(N,M,K)
    @dphpc_time(C, run_gemm_kernel(N, M, K, A, B, C), "S")

    N, M, K = 2500, 2750, 3000
    A, B, C = init_matrices(N,M,K)
    @dphpc_time(C, run_gemm_kernel(N, M, K, A, B, C), "M")

    N, M, K = 7000, 7500, 8000
    A, B, C = init_matrices(N,M,K)
    @dphpc_time(C, run_gemm_kernel(N, M, K, A, B, C), "L")

    N, M, K = 2000, 2300, 2600
    A, B, C = init_matrices(N,M,K)
    @dphpc_time(C, run_gemm_kernel(N, M, K, A, B, C), "paper")


end

main()


