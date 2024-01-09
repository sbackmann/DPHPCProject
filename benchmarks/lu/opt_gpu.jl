include("../../timing/dphpc_timing.jl")
using CUDA 


function kernel_col(A, N, i)
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x + i + 1

    if j <= N
        A[j, i] /= A[i, i]
    end

    nothing
end


function kernel_submat(A, N, i)
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x + i + 1
    k = (blockIdx().y - 1) * blockDim().y + threadIdx().y + i + 1

    if j <= N && k <= N
        A[j, k] -= (A[i, k] * A[j, i])
    end

    nothing
end


function run_lu_kernel(N, A)
    threadsPerBlock1D = 256
    threadsPerBlock2D = (16, 16)
    
    for i in 1:N
        blocks1D = div(N - i - 2, threadsPerBlock1D + 1)
        @cuda threads=threadsPerBlock1D blocks=blocks1D kernel_col(A, N, i)
        
        blocks2D = (div(N - i - 2, threadsPerBlock2D[1] + 1), div(N - i - 2, threadsPerBlock2D[2] + 1))
        @cuda threads=threadsPerBlock2D blocks=blocks2D kernel_submat(A, N, i)
    end
    
    CUDA.synchronize()
end


include("_main_gpu.jl")

main()
