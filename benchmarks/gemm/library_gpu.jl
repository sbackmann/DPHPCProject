include("../../timing/dphpc_timing.jl")
include("./validation.jl")

using CUDA 

const alpha = 1.5
const beta = 1.2

validation = false



function init_matrices(N, M, K)

    A = zeros(Float64, N, K)
    B = zeros(Float64, K, M)
    C = zeros(Float64, N, M)

    A = [(i*j+1) % K / K for i in 1:N, j in 1:K]
    B = [(i*j+1) % M / M for i in 1:K, j in 1:M]
    C = [(i*j+1) % M / M for i in 1:N, j in 1:M]

    return CuArray(A), CuArray(B), CuArray(C)

end


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


function main()

    if validation 

        N, M, K = 30, 40, 50
        A, B, C = initialize_matrices_val(N, M, K)
        run_gemm_kernel(N,M,K,A,B,C)
        C_empty = zeros(Float64, N, M)
        C_cpu = CUDA.copyto!(C_empty, C)  
        is_valid = validate(C_cpu)

        print(is_valid)

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


