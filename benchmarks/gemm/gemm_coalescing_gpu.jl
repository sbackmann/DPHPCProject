include("../../timing/dphpc_timing.jl")

using CUDA 

# eliminate global variables + accumulators 
# global memory coalescing 

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
    i = (threadIdx().x/32) + blockIdx().x * 32
    j = (threadIdx().x % 32) + blockIdx().y * 32

    alpha = 1.5
    beta = 1.2

    if i <= N && j <= M
        acc1 = C[i, j] * beta
        acc2 = 0.0
        for k = 1:K
            acc2 += alpha * A[i, k] * B[k, j]
        end
        C[i, j] = acc1 + acc2
    end
end

function run_gemm_kernel(N, M, K, A, B, C)
    threadsPerBlock = (32*32)
    # The following line could be wrong 
    # block.x = 32*32 
    # block.y = 1
    # I changed the blocksize here, the prev one could be wrong
    numBlocks = ((M+ 32*32 -1 ) ÷ 32*32, N)

    @cuda threads=threadsPerBlock blocks=numBlocks gemm_kernel(N, M, K, A, B, C)
    CUDA.synchronize()

end

function main()


        N, M, K = 1000, 1100, 1200
        @dphpc_time((A, B, C) = init_matrices(N,M,K), run_gemm_kernel(N, M, K, A, B, C), "S")

        N, M, K = 2500, 2750, 3000
        @dphpc_time((A, B, C) = init_matrices(N,M,K), run_gemm_kernel(N, M, K, A, B, C), "M")

        N, M, K = 7000, 7500, 8000
        @dphpc_time((A, B, C) = init_matrices(N,M,K), run_gemm_kernel(N, M, K, A, B, C), "L")

        N, M, K = 2000, 2300, 2600
        @dphpc_time((A, B, C) = init_matrices(N,M,K), run_gemm_kernel(N, M, K, A, B, C), "paper")



end

main()


