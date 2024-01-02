include("../../timing/dphpc_timing.jl")

using CUDA 

const alpha = 1.5
const beta = 1.2

# validation = true

# TODO: Transpose C back to NM before doing the validation 
# validated  
# Slightly better than oneline on CPU but worse than GPU_col_major.cu
# 

function init_matrices(N, M, K) # not tested during validation 

    # transpose all matrices 
    A = zeros(Float64, K, N)
    B = zeros(Float64, M, K)
    C = zeros(Float64, M, N)

    # swapped here 
    A = [(i*j+1) % K / K for j in 1:K, i in 1:N]
    B = [(i*j+1) % M / M for j in 1:M, i in 1:K]
    C = [(i*j+1) % M / M for j in 1:M, i in 1:N] 

    return CuArray(A), CuArray(B), CuArray(C)

end

function gemm_kernel(N, M, K, A, B, C)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    j = threadIdx().y + (blockIdx().y - 1) * blockDim().y

    if i <= N && j <= M
        C[j, i] *= beta
        for k = 1:K
            C[j, i] += alpha * B[j, k] * A[k, i]
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


function main()

        if validation 

        c_matrix = validate(30,40,50)
        println(c_matrix) 

        else 

        N, M, K = 1000, 1100, 1200
        @dphpc_time((A, B, C) = init_matrices(N,M,K), run_gemm_kernel(N, M, K, A, B, C), "S")

        N, M, K = 2500, 2750, 3000
        @dphpc_time((A, B, C) = init_matrices(N,M,K), run_gemm_kernel(N, M, K, A, B, C), "M")

        N, M, K = 7000, 7500, 8000
        @dphpc_time((A, B, C) = init_matrices(N,M,K), run_gemm_kernel(N, M, K, A, B, C), "L")

        N, M, K = 2000, 2300, 2600
        @dphpc_time((A, B, C) = init_matrices(N,M,K), run_gemm_kernel(N, M, K, A, B, C), "paper")

        end 

end 

main()


