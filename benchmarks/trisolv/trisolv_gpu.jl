using LinearAlgebra
using CUDA
using PrettyTables

include("../../timing/dphpc_timing.jl")

DEV = true
TIME = false
DEBUG = false

eval_benchmarks = Dict(
    "S" => 2000,
    "M" => 5000,
    "L" => 14000,
    "paper" => 16000
)

dev_benchmarks = Dict(
    "S" => 4,
    "M" => 6
)

function main()
    if DEV 
        benchmark_sizes = dev_benchmarks 
    else 
        benchmark_sizes = eval_benchmarks 
    end

    validate()

    for (preset, N) in benchmark_sizes
        L, x, b = initialize(N, eltype(Float64))

        if DEBUG 
            print_all(L, x, b) 
        end

        if TIME
            println("Benchmarking $preset")
            res = @dphpc_time nothing kernel(L, x, b) 
            println(res)
        else
            x = kernel(L, x, b)
            if DEBUG 
                println(x) 
            end
        end
    end
end

function kernel(L, x, b)
    N = length(x)
    for i in 1:N
        dp = CUDA.dot(L[i, 1:i-1], x[1:i-1])
        x[i] = (b[i] - dp) / L[i, i]
    end

    return x
end

function initialize(N, datatype=Float64)
    L = [((i + N - j + 1) * 2 / N) for i in 1:N, j in 1:N]
    x = fill(-999.0, N)
    b = [i - 1 for i in 1:N]

    return L, x, b
end

function validate()
    args = cuda_initialize(4)

    x = kernel(args...)
    if !is_correct(args[1], x, args[3])
        throw("Validation failed")
    end
end

function is_correct(L, x, b)
    L = Array(L); x = Array(x); b = Array(b)

    N = size(L, 1)
    for i in 1:N, j in 1:N
        if i < j
            L[i, j] = 0
        end
    end

    expected_solution = L \ b
    println(expected_solution)

    display(x)

    # Compare the result with the expected solution
    if isapprox(x, expected_solution, atol=1e-4)
        println("Test passed: The solution matches the expected result.")
        return true
    else
        println("Test failed: The solution does not match the expected result.")
        return false
    end
end

function print_all(L, x, b)
    println("Data")
    println("Running on set N=$(size(L, 1))")
    println("L:")
    display(L)
    println("x:")
    display(x)
    println("b:")
    display(b)
end

function reset() end

function cuda_initialize(N, datatype=Float64)
    L = CuArray([datatype(((i + N - j + 1) * 2) / N) for i in 1:N, j in 1:N])
    x = CUDA.zeros(N)
    b = CuArray([datatype(i - 1) for i in 1:N])

    return L, x, b
end

function cuda_kernel(L, x, b)
    CUDA.@allowscalar pretty_table(L)
    CUDA.@allowscalar pretty_table(x)
    CUDA.@allowscalar pretty_table(b)

    N = length(x)
    dp = CUDA.zeros(N)
    @cuda threads=N cuda_kernel_helper(L, x, b, dp)
    println(dp)
    return x
end

function cuda_kernel_helper(L, x, b, dp)
    i = threadIdx().x

    if i <= length(x)
        local_dot = 0.0
        for k in 1:i-1
            local_dot += L[i, k] * x[k]
        end

        dp[i] = local_dot

        x[i] = (b[i] - local_dot) / L[i, i]
        # x[i] = (b[i] - dot(L[i, 1:i-1], x[1:i-1])) / L[i, i]
    end
    return
end

main()