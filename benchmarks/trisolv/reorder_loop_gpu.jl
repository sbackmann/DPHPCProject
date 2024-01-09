using LinearAlgebra
using CUDA
using PrettyTables

include("utils.jl")
include("../../timing/dphpc_timing.jl")



function trisolv_kernel(L, x, b, i, N)
    id = threadIdx().x + (blockIdx().x - 1) * blockDim().x

    j = i+id-1
    d = i-1 # x is done up to d
    if d >= 1 && j <= N
        x[j] += x[d]*L[j, d]
    end

    if id == 1
        # x[i] stores the dot product of L[i, 1:i-1] and x[1:i-1]
        x[i] = (b[i] - x[i]) / L[i, i]
    end

    nothing
end


function kernel(L, x, b)
    N = length(x)

    threads = 64

    for i in 1:N
        blocks = (N - i) ÷ threads + 1
        @cuda threads=threads blocks=blocks trisolv_kernel(L, x, b, i, N)
    end

    CUDA.synchronize()
end


main()