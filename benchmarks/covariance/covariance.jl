using Statistics
using LinearAlgebra
using PrettyTables
using DelimitedFiles

include("../../timing/dphpc_timing.jl")
DEV = true
TIME = true

eval_benchmarks = Dict(
    "S" => (500, 600),
    "M" => (1400, 1800),
    "L" => (3200, 4000),
    "paper" => (1200, 1400)
)

dev_benchmarks = Dict(
    "S" => (2, 2),
    "M" => (5, 7),
    "L" => (10, 20),
)

function main() 
    if DEV benchmark_sizes = dev_benchmarks 
    else benchmark_sizes = eval_benchmarks end

    for dims in keys(benchmark_sizes)
        println("Benchmarking $dims")
        M, N = benchmark_sizes[dims]

        data = initialize(M, N)

        if TIME
            res = @dphpc_time data, kernel(M, N, data)
            println(res)
        else
            res = kernel(M, N, data)
            pretty_table(res)
        end

    end

end

function kernel(M, float_n, data)
    mean_data = mean(data, dims=1)
    data .-= mean_data
    cov = zeros(eltype(data), M, M)
    for i in 1:M
        for j in i:M
            cov[i, j] = cov[j, i] = dot(data[:, i], data[:, j]) / (float_n - 1.0)
        end
    end
    return cov
end

function initialize(M, N, datatype=Float64)
    return [datatype((i-1) * (j-1)) / M for i in 1:N, j in 1:M]
end

main()
