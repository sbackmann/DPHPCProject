include("../../timing/dphpc_timing.jl")
using CUDA
using Printf


function init_arrays(n)
    A = zeros(Float64, n, n)
    B = zeros(Float64, n, n)

    A = [((i - 1) * (j + 1) + 2) / n for i in 1:n, j in 1:n]
    B = [((i - 1) * (j + 2) + 3) / n for i in 1:n, j in 1:n]

    return A, B
end


function reset(A, B)
    return CuArray(A), CuArray(B)
end


function print_array(A)
    for i in 1:size(A, 1)
        for j in 1:size(A, 2)
            @printf("%.2f ", A[i, j])
        end
        println()
    end
end


function run_kernel_j2d(tsteps, n, A, B)
    cA, lA, rA, tA, bA = @views (A[2:n-1, 2:n-1], A[2:n-1, 1:n-2], A[2:n-1, 3:n], A[1:n-2, 2:n-1], A[3:n, 2:n-1]) # precompute views... I guess 🤷
    cB, lB, rB, tB, bB = @views (B[2:n-1, 2:n-1], B[2:n-1, 1:n-2], B[2:n-1, 3:n], B[1:n-2, 2:n-1], B[3:n, 2:n-1])
    for t in 1:tsteps
        @. cB = 0.2 * (cA + lA + rA + tA + bA)
        @. cA = 0.2 * (cB + lB + rB + tB + bB)
    end
    
    CUDA.synchronize()
end


function main()
    tsteps, n = 50, 150
    A_cpu, B_cpu = init_arrays(n)   # it doesn't write through to these
    A, B = reset(A_cpu, B_cpu)
    @time run_kernel_j2d(tsteps, n, A, B)
    res = @dphpc_time((A,B)=reset(A_cpu, B_cpu), run_kernel_j2d(tsteps, n, A, B), "S")
    # A_res = CUDA.copyto!(Array{Float64}(undef, n, n), A)
    # println("matrix A out:")
    # print_array(A_res)
    println(res)
    

    tsteps, n = 80, 350
    A_cpu, B_cpu = init_arrays(n)
    A, B = reset(A_cpu, B_cpu)
    res = @dphpc_time((A,B)=reset(A_cpu, B_cpu), run_kernel_j2d(tsteps, n, A, B), "M")
    # A_res = CUDA.copyto!(Array{Float64}(undef, n, n), A)
    # println("matrix A out:")
    # print_array(A_res)
    println(res)

    tsteps, n = 200, 700
    A_cpu, B_cpu = init_arrays(n)
    A, B = reset(A_cpu, B_cpu)
    res = @dphpc_time((A,B)=reset(A_cpu, B_cpu), run_kernel_j2d(tsteps, n, A, B), "L")
    # A_res = CUDA.copyto!(Array{Float64}(undef, n, n), A)
    # println("matrix A out:")
    # print_array(A_res)
    println(res)

    tsteps, n = 1000, 2800
    A_cpu, B_cpu = init_arrays(n)
    A, B = reset(A_cpu, B_cpu)
    res = @dphpc_time((A,B)=reset(A_cpu, B_cpu), run_kernel_j2d(tsteps, n, A, B), "paper")
    # A_res = CUDA.copyto!(Array{Float64}(undef, n, n), A)
    # println("matrix A out:")
    # print_array(A_res)
    println(res)
end 

main()