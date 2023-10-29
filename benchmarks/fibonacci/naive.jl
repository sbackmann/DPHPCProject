

include("../../timing/dphpc_timing.jl")

fib(n) = n <= 2 ? 1 : fib(n-1) + fib(n-2)

main() = @dphpc_time fib(35)

main()