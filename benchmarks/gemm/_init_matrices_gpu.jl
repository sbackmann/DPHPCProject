
@inbounds function init_matrices(N, M, K)

    A = zeros(Float64, N, K)
    B = zeros(Float64, K, M)
    C = zeros(Float64, N, M)

    for i in 1:N, j in 1:K
        A[i, j] = (i*j+1) % K * (1/K)
    end

    for i in 1:K, j in 1:M
        B[i, j] = (i*j+1) % M * (1/M)
    end

    for i in 1:N, j in 1:M
        C[i, j] = (i*j+1) % M * (1/M)
    end
    
    return CuArray(A), CuArray(B), CuArray(C)

end