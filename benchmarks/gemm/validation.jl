using CUDA

# Runs naive kernel 
# compares the result of the naive kernel w/ optimized 

function initialize_matrices(N, M, K)
    A = fill(0.5, N, K)
    B = fill(0.7, K, M)
    C = fill(0.3, N, M)
    return CuArray(A), CuArray(B), CuArray(C)
end


function gemm_naive_kernal(N, M, K, A, B, C)
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


function run_naive_kernel(N, M, K, A, B, C)
    threadsPerBlock = (16, 16)
    numBlocks = ((N - 1) ÷ 16 + 1, (M - 1) ÷ 16 + 1)
    @cuda threads=threadsPerBlock blocks=numBlocks gemm_kernel(N, M, K, A, B, C)
    CUDA.synchronize()
    C_empty = zeros(Float64, N, M)
    C_cpu_ = CUDA.copyto!(C_empty, C) 
    return C_cpu_
end


function validate(result) 

    N, M, K = 30, 40, 50
    A, B, C = initialize_matrices(N, M, K)
    c_cpu_ = run_naive_kernel(N,M,K,A,B,C)
    #print("C from validation script")
    #println(c_cpu_)

    if c_cpu_ == result 
        return "Correct"
    else 
        return "lol"
    end

end 


