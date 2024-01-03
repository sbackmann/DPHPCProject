using LinearAlgebra

include("utils.jl")
include("../../timing/dphpc_timing.jl")

VALIDATE = false

kernel(L, x, b) = ldiv!(x, LowerTriangular(L), b)

function main()
    if VALIDATE
        correctness_check(false, ["S", "M", "paper"])
    end

    println("Running benchmarks...")
    run_benchmarks()

end

main()