include("../../timing/dphpc_timing.jl")

using CUDA

# C .= α*A*A' + β*C

function init(n, k)
    hostA = zeros(Float64, n, k)
    hostC = zeros(Float64, n, n)

    for j=1:k, i=1:n
        hostA[i, j] = ((i*j+1)%n) / n;
    end
    for j=1:n, i=1:n
        hostC[i, j] = ((i*j+2)%k) / k;
    end
    
    return 1.5, 1.2, hostC, CuArray(hostC), CuArray(hostA)
end


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
    C .= α .* A*A' .+ β .* C
end

function main()
    n, k = 5, 3
    α, β, hostC, C, A = init(n, k)
    display(hostC)
    display(C)
    display(A)
    @dphpc_time(C = CuArray(hostC), run_kernel(n, k, α, β, C, A))
    display(C)



    n, k = 70, 50
    α, β, hostC, C, A = init(n, k)
    @dphpc_time(C = CuArray(hostC), run_kernel(n, k, α, β, C, A), "S")

    n, k = 200, 150
    α, β, hostC, C, A = init(n, k)
    @dphpc_time(C = CuArray(hostC), run_kernel(n, k, α, β, C, A), "M")

    n, k = 600, 500
    α, β, hostC, C, A = init(n, k)
    @dphpc_time(C = CuArray(hostC), run_kernel(n, k, α, β, C, A), "L")

    n, k = 1200, 1000
    α, β, hostC, C, A = init(n, k)
    @dphpc_time(C = CuArray(hostC), run_kernel(n, k, α, β, C, A), "paper")
end

main()
