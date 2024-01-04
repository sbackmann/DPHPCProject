include("../../timing/dphpc_timing.jl")
include("./validation.jl")

using CUDA 

# global memory coalescing 

const BLOCKSIZE = 32
validation = false

const alpha = 1.5
const beta = 1.2

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
    grid = ((M + block[1] - 1) ÷ block[1], (N + block[2] - 1) ÷ block[2])

    @cuda threads=block blocks=grid gemm_kernel(N, M, K, A, B, C)
    CUDA.synchronize()

end

include("_main.jl")

main()


