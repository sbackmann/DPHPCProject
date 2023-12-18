using LinearAlgebra
using Serialization
using Statistics

include("utils.jl")

benchmark_sizes = Dict(
    "S"     => 2000, 
    "M"     => 5000, 
    "L"     => 14000, 
    "paper" => 16000, 
    "dev"   => 4, 
)

DEBUG = false

function main()
    for (preset, N) in benchmark_sizes
        println("Creating $preset test")

        L, x, b = initialize(N, eltype(Float64))

        if DEBUG print_all(L, x, b) end

        result = kernel(L, x, b)
        is_correct(L, result, b)

        create_testfile(result, preset)
    end
end

function create_testfile(solution, prefix)
    test_cases_dir = "benchmarks/trisolv/test_cases"
    if !isdir(test_cases_dir)
        test_cases_dir = "test_cases"
    end

    open("$test_cases_dir/$prefix.jls", "w") do io
        Serialization.serialize(io, solution)
    end
    
    open("$test_cases_dir/$prefix.tsv", "w") do io
        for row in eachrow(solution)
            println(io, join(row, "\t"))
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


function is_correct(L, x, b)
    N = size(L, 1)
    for i in 1:N, j in 1:N
        if i < j
            L[i, j] = 0
        end
    end

    expected_solution = L\b
    # println(expected_solution)
    # println(x)

    # Compare the result with the expected solution
    if isapprox(x, expected_solution)
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

main()