include("../../timing/dphpc_timing.jl")
include("./validation.jl")

using CUDA 

# eliminate global variables + accumulators 
# global memory coalescing 

const BLOCKSIZE = 32
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
    i, j = blockIdx().x * BLOCKSIZE + div(threadIdx().x, BLOCKSIZE), blockIdx().y * BLOCKSIZE + mod(threadIdx().x, BLOCKSIZE)

    alpha = 1.5
    beta = 1.2

    if i <= N && j <= M
        @inbounds acc1 = C[i, j] * beta
        acc2 = 0.0
        for k = 1:K
            @inbounds acc2 += alpha * A[i, k] * B[k, j]
        end
        C[i, j] = acc1 + acc2
    end
    nothing 
end

function run_gemm_kernel(N, M, K, A, B, C)
    block = (BLOCKSIZE,BLOCKSIZE)
    grid = ((M + block[1] - 1) ÷ block[1], (N + block[2] - 1) ÷ block[2])
    # The following line could be wrong 
    # block.x = 32*32 
    # block.y = 1
    # I changed the blocksize here, the prev one could be wrong
    #numBlocks = ((M+ 32*32 -1 ) ÷ 32*32, N)

    #@cuda threads=threadsPerBlock blocks=numBlocks gemm_kernel(N, M, K, A, B, C)

    @cuda threads=block blocks=grid gemm_kernel(N, M, K, A, B, C)
    CUDA.synchronize()

end

include("_main.jl")

main()


