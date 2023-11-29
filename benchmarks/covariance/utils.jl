using Serialization
using Statistics
using CUDA

include("../../timing/dphpc_timing.jl")

function initialize(M, N, datatype=Float64; cuda=false)
    data = [datatype((i-1) * (j-1)) / M for i in 1:N, j in 1:M]
    return cuda ? CuArray(data) : data
end

benchmark_sizes = Dict(
    "S"     => (500, 600),
    "M"     => (1400, 1800),
    "L"     => (3200, 4000),
    "paper" => (1200, 1400)
)

function run_benchmarks(; cuda = false, create_tests = false)

    assert_correctness(cuda)

    for (preset, dims) in benchmark_sizes
        M, N = dims

        data = initialize(M, N, cuda=cuda)
        
        if create_tests
            solution = kernel(M, N, data)
            create_testfile(solution, preset)
        end

        @dphpc_time(nothing, kernel(M, N, data), preset)
    end
end

function create_testfile(solution, prefix)
    test_cases_dir = "benchmarks/covariance/test_cases"
    if !isdir(test_cases_dir)
        test_cases_dir = "test_cases"
    end

    open("$test_cases_dir/$prefix.jls", "w") do io
        Serialization.serialize(io, solution)
    end
end

function assert_correctness(cuda)
    prefix = "S"
    data = initialize(benchmark_sizes[prefix]..., cuda=cuda)
    solution = kernel(benchmark_sizes[prefix]..., data)

    test_cases_dir = "benchmarks/covariance/test_cases"
    if !isdir(test_cases_dir)
        test_cases_dir = "test_cases"
    end

    expected = open("$test_cases_dir/$prefix.jls" ) do io
        Serialization.deserialize(io)
    end
    @assert isequal(solution, expected)
end