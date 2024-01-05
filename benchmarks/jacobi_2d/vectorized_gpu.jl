include("../../timing/dphpc_timing.jl")
using CUDA
using Printf




function run_kernel_j2d(tsteps, n, A, B)
    cA, lA, rA, tA, bA = @views (A[2:n-1, 2:n-1], A[2:n-1, 1:n-2], A[2:n-1, 3:n], A[1:n-2, 2:n-1], A[3:n, 2:n-1]) # precompute views... I guess 🤷
    cB, lB, rB, tB, bB = @views (B[2:n-1, 2:n-1], B[2:n-1, 1:n-2], B[2:n-1, 3:n], B[1:n-2, 2:n-1], B[3:n, 2:n-1])
    for t in 1:tsteps
        @. cB = 0.2 * (cA + lA + rA + tA + bA)
        @. cA = 0.2 * (cB + lB + rB + tB + bB)
    end
    
    CUDA.synchronize()
end

include("_main_gpu.jl")

main()