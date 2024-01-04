include("../../timing/dphpc_timing.jl")
include("./validation.jl")

using CUDA 

# global memory coalescing 

const BLOCKSIZE = 32


include("_init_matrices_gpu.jl")

function gemm_kernel(N, M, K, A, B, C)
    alpha = 1.5
    beta = 1.2

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
    block = (BLOCKSIZE,BLOCKSIZE)
    grid = ((N + block[1] - 1) ÷ block[1], (M + block[2] - 1) ÷ block[2])

    @cuda threads=block blocks=grid gemm_kernel(N, M, K, A, B, C)
    CUDA.synchronize()

end

include("_main.jl")

main()


