include("../../timing/dphpc_timing.jl")
using CUDA 



function lu_kernel(N, A)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    if i <= N
        for j in 1:i
            for k in 1:j
                A[j, i] -= A[k, i] * A[j, k]
            end
            A[j, i] /= A[j, j]
        end

        for j in i:N
            for k in 1:i
                A[j,i] -= A[k, i] * A[j, k]
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
