include("../../timing/dphpc_timing.jl")
using CUDA 
using LinearAlgebra
# naive GPU implemenation but unrolled 



function lu_kernel(N, A)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    if i <= N
        @inbounds @simd for j in 1:i
            for k in 1:j
                A[i, j] -= A[i, k] * A[k, j]
            end
            A[i, j] /= A[j, j]
        end

        # synchronization is not necessary here as julia handles it

        @inbounds @simd for j in i:N
            for k in 1:i
                A[i, j] -= A[i, k] * A[k, j]
            end
        end
    end

    return
end


function run_lu_kernel(N, A)

    threadsPerBlock = 256
    blocksPerGrid = (N + threadsPerBlock - 1) ÷ threadsPerBlock
    @cuda threads=threadsPerBlock blocks=blocksPerGrid lu_kernel(N, A)
    CUDA.synchronize()


end


include("_main_gpu.jl")

main()
