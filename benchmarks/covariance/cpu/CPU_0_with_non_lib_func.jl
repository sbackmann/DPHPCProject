using Statistics
using LinearAlgebra

include("utils.jl")
PERFORM_VALIDATION = true

function kernel(M, N, data)
    column_mean= zeros(eltype(data), 1, M)

    for j in 1:M
        for row_idx in 1:N
            column_mean[1, j] += data[row_idx, j]
        end
        column_mean[1, j] /= N
    end

    data .-= column_mean
    cov = zeros(eltype(data), M, M)
    for i in 1:M
        for j in i:M
            cov[i, j] = cov[j, i] = dot(data[:, i], data[:, j]) / (N - 1.0)
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
