using Serialization
using Statistics
using CUDA

include("../../timing/dphpc_timing.jl")

DEBUG = false

function initialize(N, datatype=Float64; cuda=false)
    L = [((i + N - j + 1) * 2 / N) for i in 1:N, j in 1:N]

    x = fill(-999.0, N)
    b = [i - 1 for i in 1:N]

    if DEBUG
        pretty_table(L)
        pretty_table(x)
        pretty_table(b)
    end

    return L, x, b
end

benchmark_sizes = Dict(
    "S"     => 2000, 
    "M"     => 5000, 
    "L"     => 14000, 
    "paper" => 16000, 
    "dev"   => 4, 
)

function correctness_check(cuda, prefix=["S", "M", "L", "paper"])
    for preset in prefix
        println("Checking correctness for $preset")
        assert_correctness(cuda, preset)
    end
end

function run_benchmarks(; cuda = false, create_tests = false)
    correctness_check(cuda)
    return

    if !create_tests
        assert_correctness(cuda)
        assert_correctness(cuda, "S")
    end

    for (preset, dims) in benchmark_sizes
        N = dims

        data = initialize(N, cuda=cuda)
        
        if create_tests
            solution = kernel(data...)
            create_testfile(solution, preset)
        end

        # print(@dphpc_time(nothing, kernel(data...), preset))
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

function assert_correctness(cuda, prefix="dev")
    data = initialize(benchmark_sizes[prefix]..., cuda=cuda)
    solution = kernel(data...)

    test_cases_dir = "benchmarks/trisolv/test_cases"
    if !isdir(test_cases_dir)
        test_cases_dir = "test_cases"
    end

    expected = open("$test_cases_dir/$prefix.jls" ) do io
        Serialization.deserialize(io)
    end

    if cuda
        cpu_data = CUDA.copyto!(Matrix{Float64}(undef, size(solution)...), solution)
        copyto!(cpu_data, solution)
        solution = cpu_data
    end

    if !isapprox(solution, expected)
        open("$test_cases_dir/$(prefix)_wrong.tsv", "w") do io
            for row in eachrow(solution)
                println(io, join(row, "\t"))
            end
        end
    end
    @assert isapprox(solution, expected)
end