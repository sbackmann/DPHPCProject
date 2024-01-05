using Serialization
using Statistics
using CUDA

include("../../timing/dphpc_timing.jl")

function initialize(N, transpose; cuda=false)
    # L = [((i + N - j + 1) * 2 / N) for i in 1:N, j in 1:N]
    L = zeros(N, N)
    for j=1:N, i=1:N
        L[i, j] = 2 * (i + N - j + 1)
    end
    L *= (1/N)
    
    if transpose
        L = L'[:, :]
    end

    x = zeros(N)
    b = collect(0:N-1)

    if cuda
        L = CUDA.CuArray(L)
        x = CUDA.CuArray(x)
        b = CUDA.CuArray(b)
    end
    return L, x, b
end


sizes = NPBenchManager.get_parameters("trisolv")

benchmark_sizes = Dict(
    "S"     => (sizes["S"] |> values |> collect)[1], 
    "M"     => (sizes["M"] |> values |> collect)[1], 
    "L"     => (sizes["L"] |> values |> collect)[1], 
    "paper" => (sizes["paper"] |> values |> collect)[1], 
)

function correctness_check(cuda, transpose=false, prefix=["S", "M", "L", "paper"])
    for preset in prefix
        println("Checking correctness for $preset")
        assert_correctness(cuda, transpose, preset)
    end
end

function reset(N, datatype=Float64; cuda=false, transpose=false)
   data = initialize(N, transpose, cuda=cuda) 
   if cuda
        CUDA.synchronize()
   end
   return data
end

function run_benchmarks(; cuda = false, create_tests = false, transpose = false)
    for (preset, dims) in benchmark_sizes
        N = dims

        print(@dphpc_time(data = reset(N, transpose, cuda=cuda), kernel(data...), preset))
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

function assert_correctness(cuda, transpose, prefix="dev")
    data = initialize(benchmark_sizes[prefix], transpose, cuda=cuda)
    solution = kernel(data...)

    test_cases_dir = "benchmarks/trisolv/test_cases"
    if !isdir(test_cases_dir)
        test_cases_dir = "test_cases"
    end

    expected = open("$test_cases_dir/$prefix.jls" ) do io
        Serialization.deserialize(io)
    end

    if cuda
        cpu_data = CUDA.copyto!(Vector{Float64}(undef, size(solution)...), solution)
        copyto!(cpu_data, solution)
        solution = cpu_data
    end

    if !isapprox(solution, expected, atol=1e-2)
        open("$test_cases_dir/$(prefix)_wrong.tsv", "w") do io
            for row in eachrow(solution)
                println(io, join(row, "\t"))
            end
        end
    end
    @assert isapprox(solution, expected, atol=1e-2)
end