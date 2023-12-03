using Statistics

include("utils.jl")

kernel(M, N, data) = cov(data) # using covariance from builtin statistics package

function main()
    run_benchmarks(create_tests = true)
    # run_benchmarks()
end

main()