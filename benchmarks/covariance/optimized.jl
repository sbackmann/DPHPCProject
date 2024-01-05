using Statistics
using LinearAlgebra

include("utils.jl")
PERFORM_VALIDATION = false

function kernel(M, N, data)
    column_mean = mean(data, dims=1)
    data .-= column_mean
    cov = zeros(eltype(data), M, M)

    for i in 1:M
        for j in i:M
            cov[i, j] = cov[j, i] = (1/(N-1)) * @views dot(data[:, i], data[:, j])
        end
    end

    return cov
end

function main()
    if PERFORM_VALIDATION
        correctness_check(false, ["S", "M"])
    end
    run_benchmarks()
end

main()
