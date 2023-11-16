
include("../../timing/dphpc_timing.jl")
using Statistics



function initialize(M, N, datatype=Float64)
    return [datatype((i-1) * (j-1) / M) for i in 1:N, j in 1:M]
end

kernel(M, N, data) = cov(data) # using covariance from builtin statistics package


function main() 

    benchmark_sizes = Dict(
        "S"     => (500, 600),
        "M"     => (1400, 1800),
        "L"     => (3200, 4000),
        "paper" => (1200, 1400)
    )

    for (preset, dims) in benchmark_sizes
        M, N = dims

        data = initialize(M, N)

        @dphpc_time(nothing, kernel(M, N, data), preset)
    end

end

main()