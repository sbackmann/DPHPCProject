using LinearAlgebra
using CUDA
using PrettyTables

include("utils.jl")
include("../../timing/dphpc_timing.jl")

VALIDATE = false

function main()
    if VALIDATE
        correctness_check(true, ["S", "M"])
    end
    run_benchmarks(cuda=true)
end


function kernel(L, x, b)
    ldiv!(x, LowerTriangular(L), b)
    CUDA.synchronize()
    return x
end


main()