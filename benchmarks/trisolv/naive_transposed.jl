using LinearAlgebra

include("old_utils.jl")
include("../../timing/dphpc_timing.jl")

VALIDATE = true

function kernel(L, x, b)
    N = length(x)
    for i in 1:N
        dp = dot(L[1:i-1,i], x[1:i-1])
        x[i] = (b[i] - dp) / L[i, i]
    end
    return x
end

function main()
    if VALIDATE
        # correctness_check(false, ["S", "M", "paper"])
        correctness_check(false, true, ["S"])
    end

    println("Running benchmarks...")
    run_benchmarks(transpose=true)

end

main()