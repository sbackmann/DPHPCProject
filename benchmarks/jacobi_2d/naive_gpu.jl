include("../../timing/dphpc_timing.jl")
using CUDA
using Printf



function kernel_j2d(n, A, B)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    
    if i > 1 && i < n && j > 1 && j < n
        B[i, j] = 0.2 * (A[i, j] + A[i, j-1] + A[i, j+1] + A[i+1, j] + A[i-1, j])
    end
    nothing
end


function run_kernel_j2d(tsteps, n, A, B)
    threadsPerBlock = (16, 16)
    numBlocks = (div(n + threadsPerBlock[1] - 1, threadsPerBlock[1]), div(n + threadsPerBlock[2] - 1, threadsPerBlock[2]))
    
    for t in 1:tsteps
        @cuda threads=threadsPerBlock blocks=numBlocks kernel_j2d(n, A, B)
        @cuda threads=threadsPerBlock blocks=numBlocks kernel_j2d(n, B, A)
    end
    
    CUDA.synchronize()
end


include("_main_gpu.jl")
main()