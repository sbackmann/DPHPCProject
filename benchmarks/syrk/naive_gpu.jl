include("../../timing/dphpc_timing.jl")

include("_syrk.jl")

using CUDA

# C .= α*A*A' + β*C


function syrk(n, k, α, β, C, A)
    c = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    r = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    if r <= n && c <= n && r >= c
        s = 0.0
        for i=1:k
            s += A[r, i] * A[c, i]
        end
        C[r, c] = β * C[r, c] + α * s
    end
    nothing
end


function run_kernel(n, k, α, β, C, A)
    b = n ÷ 16 + 1
    CUDA.@sync(@cuda threads=(16, 16) blocks=(b, b) syrk(n, k, α, β, C, A))
end


main_gpu()
