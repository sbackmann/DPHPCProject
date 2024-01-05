include("../../timing/dphpc_timing.jl")
using CUDA 




function lu_kernel(N, A)
    j = threadIdx().x + (blockIdx().x - 1) * blockDim().x

    if j <= N
        for i in 1:j
            for k in 1:i
                A[i, j] -= A[i, k] * A[k, j]
            end
            A[i, j] /= A[j, j]
        end

        for i in (j + 1):N
            for k in 1:j
                A[i, j] -= A[i, k] * A[k, j]
            end
        end
    end
    return
end


function run_lu_kernel(N, A)
    threadsPerBlock = 256
    numBlocks = (N - 1) ÷ threadsPerBlock + 1

    @cuda threads=threadsPerBlock blocks=numBlocks lu_kernel(N, A)
    CUDA.synchronize()

end


include("_main_gpu.jl")

main()
