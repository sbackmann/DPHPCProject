using Statistics
using LinearAlgebra

include("utils.jl")
PERFORM_VALIDATION = false

kernel(M, N, data) = cov(data)

function main()
    if PERFORM_VALIDATION
        correctness_check(false, ["S", "M"])
    end
    run_benchmarks()
end

main()
