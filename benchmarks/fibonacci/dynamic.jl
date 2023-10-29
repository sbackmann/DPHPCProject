
include("../../timing/dphpc_timing.jl")

function fib(n)
    if n <= 2
        return 1
    end
    sₙ = 1
    sₙ₋₁ = 1
    for i=3:n
        sₙ₋₁, sₙ = sₙ, sₙ + sₙ₋₁ 
    end
    return sₙ
end

main() = @dphpc_time fib(35)

main()