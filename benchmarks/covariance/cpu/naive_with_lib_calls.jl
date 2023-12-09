using Statistics
using LinearAlgebra

include("utils.jl")

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

function main()
    run_benchmarks()
end

main()
