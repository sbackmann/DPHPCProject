include("../../timing/dphpc_timing.jl")

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

    benchmarks = NPBenchManager.get_parameters("gemm")
    for (preset, sizes) in benchmarks
        N, M, K = collect(values(sizes))
        @dphpc_time(
            (A, B, C) = init_matrices(N,M,K), 
            gemm(N, M, K, A, B, C), 
            preset
        )
    end

end

main()
