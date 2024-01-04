using LinearAlgebra

include("utils_cpu.jl")

function kernel(L, x, b)
    N = length(x)
    for i in 1:N
        dp = dot(L[i, 1:i-1], x[1:i-1])
        x[i] = (b[i] - dp) / L[i, i]
    end
    return x
end

main()