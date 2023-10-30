
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

function main()

    # nothing as first argument because there is no reset/init code needed here
    # but still need to pass a first argument

    # these are the two ways to call macros in julia
    @dphpc_time(nothing, fib(1000),  "S")  # with brackets, like a function 
    @dphpc_time nothing  fib(10000)  "M"   # or just with some space between the arguments
    @dphpc_time nothing  fib(100000) "L"   # whithout the brackets, everything needs to be on the same line

    # there is also a 2 argument version like in C
    @dphpc_time(nothing, fib(38))
end



main()

