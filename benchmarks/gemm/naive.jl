include("../../timing/dphpc_timing.jl")

using Pkg
Pkg.add("Printf")
using Printf


const alpha = 1.5
const beta = 1.2

function init_matrices(N, M, K)

    A = zeros(Float64, N, K)
    B = zeros(Float64, K, M)
    C = zeros(Float64, N, M)

    A = [(i*j+1) % K / K for i in 1:N, j in 1:K]
    B = [(i*j+1) % M / M for i in 1:K, j in 1:M]
    C = [(i*j+1) % M / M for i in 1:N, j in 1:M]

    return A, B, C

end


function gemm(N, M, K, A, B, C)
    for i in 1:N
        for j in 1:M
            C[i, j] *= beta
            for k in 1:K
                C[i, j] += alpha * A[i, k] * B[k, j]
            end
        end
    end
end


function main()

    N, M, K = 50, 60, 70
    A, B, C = init_matrices(N,M,K)
    @dphpc_time(nothing, gemm(N, M, K, A, B, C), "S")

    N, M, K = 100, 110, 120
    A, B, C = init_matrices(N,M,K)
    @dphpc_time(nothing, gemm(N, M, K, A, B, C), "M")

    N, M, K = 600, 610, 620
    A, B, C = init_matrices(N,M,K)
    @dphpc_time(nothing, gemm(N, M, K, A, B, C), "L")

    N, M, K = 1000, 1100, 1200
    A, B, C = init_matrices(N,M,K)
    @dphpc_time(nothing, gemm(N, M, K, A, B, C), "paper")


end

main()
