include("../../timing/dphpc_timing.jl")
include("./validation.jl")

using CUDA 

const alpha = 1.5
const beta = 1.2

validation = false


include("_init_matrices_gpu.jl")


function run_gemm_kernel(N, M, K, A, B, C)
    C .= alpha .* (A * B) .+ beta .* C
    CUDA.synchronize()
end



# function initialize_matrices(N, M, K) #delete this after
#     A = fill(0.5, K, N)
#     B = fill(0.7, M, K)
#     C = fill(0.3, M, N)
#     return CuArray(A), CuArray(B), CuArray(C)
# end


# function validate(N, M, K) #delete this after
#     A, B, C = initialize_matrices(N,M,K)
#     threadsPerBlock = (16, 16)
#     numBlocks = ((N - 1) ÷ 16 + 1, (M - 1) ÷ 16 + 1)
#     @cuda threads=threadsPerBlock blocks=numBlocks gemm_kernel(N, M, K, A, B, C)
#     CUDA.synchronize()
#     C_empty = zeros(Float64, N, M)
#     C_cpu_ = CUDA.copyto!(C_empty, C) 
#     return C_cpu_
# end

include("_main.jl")

main()


