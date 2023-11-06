using LinearAlgebra
# using Statistics
# using PrettyTables
# using DelimitedFiles

include("../../timing/dphpc_timing.jl")
DEV = true
TIME = false

eval_benchmarks = Dict(
    "S" => (500, 600),
    "M" => (1400, 1800),
    "L" => (3200, 4000),
    "paper" => (1200, 1400)
)

dev_benchmarks = Dict(
    "S" => 4,
    # "N" => 2000,
    # "M" => 5000,
    # "L" => 14000,
    # "paper" => 16000
)

function main()
    for (benchmark, N) in dev_benchmarks
        println("Running on set N=$N")

        L, x, b = initialize(N, eltype(Float64))
        println("Data")
        println("L:")
        display(L)
        println("x:")
        display(x)
        println("b:")
        display(b)

        x = kernel(L, x, b)
        println(x)
        is_correct(L, x, b)

        println("End")
    end
end

function initialize(N, datatype=Float64)
    L = [((i + N - j + 1) * 2 / N) for i in 1:N, j in 1:N]

    x = fill(-999.0, N)
    b = [i - 1 for i in 1:N]

    return L, x, b
end

function kernel(L, x, b)
    N = length(x)
    for i in 1:N
        dp = dot(L[i, 1:i-1], x[1:i-1])
        println("dp: ")
        println(dp)

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
    println(expected_solution)

    # Compare the result with the expected solution
    if isapprox(x, expected_solution)
        println("Test passed: The solution matches the expected result.")
        return true
    else
        println("Test failed: The solution does not match the expected result.")
        return false
    end
end

main()
