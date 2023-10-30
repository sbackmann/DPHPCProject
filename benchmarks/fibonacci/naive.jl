

include("../../timing/dphpc_timing.jl")

fib(n) = n <= 2 ? 1 : fib(n-1) + fib(n-2)

function main()

    result = @dphpc_time fib(38)

    return "blub" # (no longer need to return result 🙃)
end

main()