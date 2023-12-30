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


function main()
    tsteps, n = 50, 150
    A_cpu, B_cpu = init_arrays(n)   # it doesn't write through to these
    A, B = reset(A_cpu, B_cpu)
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