using LinearAlgebra

include("utils.jl")
include("../../timing/dphpc_timing.jl")

VALIDATE = false

function kernel(L, x, b)
    N = length(x)
    @inbounds for i in 1:N
        dp = @views dot(L[i, 1:i-1], x[1:i-1])
        x[i] = (b[i] - dp) / L[i, i]
    end
    return x
end

function main()
    if VALIDATE
        correctness_check(false, ["S", "M", "paper"])
    end

    println("Running benchmarks...")
    run_benchmarks()

end

main()