using LinearAlgebra
using CUDA
using PrettyTables

include("utils.jl")
include("../../timing/dphpc_timing.jl")


@inbounds function trisolv_kernel(L, x, b, i, N)
    id = threadIdx().x + (blockIdx().x - Int32(1)) * blockDim().x

    j = i+id-Int32(1)
    d = i-Int32(1) # x is done up to d
    if d >= Int32(1) && j <= N
        @fastmath x[j] += x[d]*L[j, d]
    end

    if id == Int32(1)
        # x[i] stores the dot product of L[i, 1:i-1] and x[1:i-1]
        @fastmath x[i] = (b[i] - x[i]) / L[i, i]
    end

    nothing
end


function kernel(L, x, b)
    N = length(x)

    threads = 256

    for i in 1:N
        blocks = (N - i + 1) ÷ threads + 1
        @cuda threads=threads blocks=blocks trisolv_kernel(L, x, b, Int32(i), Int32(N))
    end

    CUDA.synchronize()
end


main()