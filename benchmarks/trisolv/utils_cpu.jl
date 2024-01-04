using LinearAlgebra
using Statistics
using CUDA

include("../../timing/dphpc_timing.jl")

function initialize(N)
    # L = [((i + N - j + 1) * 2 / N) for i in 1:N, j in 1:N]
    L = zeros(N, N)
    for j=1:N, i=1:N
        L[i, j] = 2 * (i + N - j + 1)
    end
    L *= (1/N)
    
    x = zeros(N)
    b = collect(0:N-1)

    return L, x, b
end

benchmark_sizes = Dict(
    "S"     => 2000, 
    "M"     => 5000, 
    "L"     => 14000, 
    "paper" => 16000, 
    "dev"   => 4, 
)

is_valid(L, x, b) = norm(LowerTriangular(L)*x - b) < 1e-4

function main()
    val_N = 100
    L, x, b = initialize(val_N)
    kernel(L, x, b)
    
    if is_valid(L, x, b)
        for (preset, dims) in benchmark_sizes
            N = dims
    
            @dphpc_time(data = initialize(N), kernel(data...), preset)
        end
    else 
        println("VALIDATION FAILED")
    end
end