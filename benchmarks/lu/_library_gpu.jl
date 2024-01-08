include("../../timing/dphpc_timing.jl")
using CUDA 
using LinearAlgebra





function run_lu_kernel(N, A)
    
    CUDA.@sync begin 
        dcp = lu(A)
        A .= dcp.L + dcp.U - I
    end

end

include("_main_gpu.jl")

main()
