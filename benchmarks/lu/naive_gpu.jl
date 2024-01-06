include("../../timing/dphpc_timing.jl")
using CUDA 


function lu_kernel(N, A)
    for i in 1:N
        for j in 1:i-1
            for k in 1:j-1
                A[i, j] -= A[i, k] * A[k, j]
            end
            A[i, j] /= A[j, j]
        end
        
        for j in i:N
            for k in 1:i-1
                A[i, j] -= A[i, k] * A[k, j]
            end
        end
    end

    return
end



function run_lu_kernel(N, A)
    @cuda threads=1 blocks=1 lu_kernel(N, A)
    CUDA.synchronize()
end



include("_main_gpu.jl")

main()
