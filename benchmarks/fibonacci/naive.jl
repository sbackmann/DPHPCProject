

include("../../timing/dphpc_timing.jl")

fib(n) = n <= 2 ? 1 : fib(n-1) + fib(n-2)

function main()
    
    # blablabla...

    result = @dphpc_time fib(38)

    # blablabla...

    return result
end

main() # last expression in the file must be result of @dphpc_time ... !