using Statistics
using LinearAlgebra
using PrettyTables
using DelimitedFiles
using CUDA

include("utils.jl")
include("../../timing/dphpc_timing.jl")

VALIDATE = false


function kernel(M, N, data)
    mean = CUDA.sum(data, dims=1) .* (1/N)

    data .-= mean
    
    return (1/(N-1)) .* (data' * data)
end


function main()
    if VALIDATE
        data = initialize(3,4, cuda=true)
        covar = kernel(3, 4, data)
        println("Got")
        display(covar)
        println("Expected")
        display(cov(initialize(3,4)))
        correctness_check(true, ["S", "M"])
    end
    run_benchmarks(cuda = true)
end



main()
