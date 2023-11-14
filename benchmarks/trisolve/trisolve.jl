using LinearAlgebra
include("../../timing/dphpc_timing.jl")

DEV = true
TIME = true
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
    if DEV benchmark_sizes = dev_benchmarks 
    else benchmark_sizes = eval_benchmarks end

    validate()

    for (preset, N) in benchmark_sizes
        L, x, b = initialize(N, eltype(Float64))

        if DEBUG print_all(L, x, b) end

        if TIME
            println("Benchmarking $preset")
            res = @dphpc_time nothing kernel(L, x, b) 
            println(res)
        else
            x = kernel(L, x, b)
            if DEBUG println(x) end
        end
    end
end


function kernel(L, x, b)
    N = length(x)
    for i in 1:N
        x[i] = (b[i] - dot(L[i, 1:i-1], x[1:i-1])) / L[i, i]
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
    args = initialize(5)
    kernel(args...)
    if !is_correct(args...)
        throw("Validation failed")
    end
end

function is_correct(L, x, b)
    N = size(L, 1)
    for i in 1:N, j in 1:N
        if i < j
            L[i, j] = 0
        end
    end

    expected_solution = L\b
    println(expected_solution)

    # Compare the result with the expected solution
    if isapprox(x, expected_solution)
        println("Test passed: The solution matches the expected result.")
        display(x)
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

main()
