include("../../timing/dphpc_timing.jl")

include("_syrk.jl")

using CUDA

# C .= α*A*A' + β*C


function init_gpu(n, k)
    hostA = zeros(Float64, k, n) # swap n and k
    hostC = zeros(Float64, n, n)

    for j=1:k, i=1:n
        hostA[j, i] = ((i*j+1)%n) / n; # here too
    end
    for j=1:n, i=1:n
        hostC[i, j] = ((i*j+2)%k) / k;
    end
    
    return 1.5, 1.2, hostC, CuArray(hostC), transpose(CuArray(hostA)) # allocate A with row major order
end



function syrk(n, k, α, β, C, A)
    c = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x # c and r swapped
    r = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    cond = (r <= n) * (c <= n) * (r >= c)
    if cond
        s = 0.0
        i::Int32 = Int32(1)
        while i <= k
            @inbounds s += A[r, i] * A[c, i]
            i += Int32(1)
        end
        @inbounds C[r, c] = β * C[r, c] + α * s
    end
    nothing
end


function run_kernel(n, k, α, β, C, A)
    threads_per_block = 16
    t = threads_per_block
    b = (n-1) ÷ t + 1
    CUDA.@sync(@cuda threads=(t, t) blocks=(b, b) syrk(Int32(n), Int32(k), α, β, C, A))
end


main_gpu()
