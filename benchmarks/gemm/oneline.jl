include("../../timing/dphpc_timing.jl")

const alpha = 1.5
const beta = 1.2

function init_matrices(N, M, K)

    A = [(i*j+1) % K / K for i in 1:N, j in 1:K]
    B = [(i*j+1) % M / M for i in 1:K, j in 1:M]
    C = [(i*j+1) % M / M for i in 1:N, j in 1:M]

    return A, B, C

end


gemm(N, M, K, A, B, C) = C .= alpha .* (A * B) .+ beta .* C


# "parameters": {
#     "S": { "NI": 1000, "NJ": 1100, "NK": 1200 },
#     "M": { "NI": 2500, "NJ": 2750, "NK": 3000 },
#     "L": { "NI": 7000, "NJ": 7500, "NK": 8000 },
#     "paper": { "NI": 2000, "NJ": 2300, "NK": 2600 }
# },


function main()


    N, M, K = 1000, 1100, 1200
    @dphpc_time(
        (A, B, C) = init_matrices(N,M,K), 
        gemm(N, M, K, A, B, C), 
        "S"
    )

    N, M, K = 2500, 2750, 3000
    @dphpc_time(
        (A, B, C) = init_matrices(N,M,K), 
        gemm(N, M, K, A, B, C), 
        "M"
    )

    N, M, K = 7000, 7500, 8000
    @dphpc_time(
        (A, B, C) = init_matrices(N,M,K), 
        gemm(N, M, K, A, B, C), 
        "L"
    )

    N, M, K = 2000, 2300, 2600
    @dphpc_time(
        (A, B, C) = init_matrices(N,M,K), 
        gemm(N, M, K, A, B, C), 
        "paper"
    )


end

main()
