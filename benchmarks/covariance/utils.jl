using Serialization
using Statistics
using CUDA

include("../../timing/dphpc_timing.jl")
if !isdefined(Main, :NPBenchManager) include("../../timing/NPBenchManager.jl") end

function initialize(M, N, datatype=Float64; cuda=false)
    # data = [datatype((i-1) * (j-1)) / M for i in 1:N, j in 1:M]
    data = datatype.(collect(0:N-1) * collect(0:M-1)')
    data = data .* (1/M)
    return cuda ? CuArray(data) : data
end


function reset(M, N, datatype=Float64; cuda=false)
   data = initialize(M, N, datatype, cuda=cuda) 
   if cuda
        CUDA.synchronize()
   end
   return data
end

function correctness_check(cuda, prefix=["S", "M", "L", "paper"])
    for preset in prefix
        println("Checking correctness for $preset")
        assert_correctness(cuda, preset)
    end
end

function run_benchmarks(; cuda = false, create_tests = false)
    # correctness_check(true)
    # return

    # if !create_tests
    #     assert_correctness(cuda)
    #     assert_correctness(cuda, "S")
    # end
    
    @dphpc_time(data = reset(300, 400, cuda=cuda), kernel(300, 400, data)) # warmup
    benchmark_sizes = NPBenchManager.get_parameters("covariance")

    for (preset, dims) in benchmark_sizes
        M, N = dims |> values |> collect
        
        if create_tests
            test_data = initialize(M, N, cuda=cuda)
            solution = kernel(M, N, test_data)
            create_testfile(solution, preset)
        end

        @dphpc_time(data = reset(M, N, cuda=cuda), kernel(M, N, data), preset) # |> println
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
    
    open("$test_cases_dir/$prefix.tsv", "w") do io
        for row in eachrow(solution)
            println(io, join(row, "\t"))
        end
    end
end

function assert_correctness(cuda, prefix="dev")
    data = initialize(benchmark_sizes[prefix]..., cuda=cuda)
    solution = kernel(benchmark_sizes[prefix]..., data)

    test_cases_dir = "benchmarks/covariance/test_cases"
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