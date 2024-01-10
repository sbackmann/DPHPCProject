using LinearAlgebra

include("utils_cpu.jl")

function kernel(L, x, b)
    N = length(x)
    x[1] = b[1] / L[1, 1]
    for i in 2:N
        d = i-1 # x is done up to d
        s = x[d]
        @views x[i:N] .+= s .* L[i:N, d]
        x[i] = (b[i] - x[i]) / L[i, i]
    end
    return x
end

main()