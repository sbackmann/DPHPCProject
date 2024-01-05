
using LinearAlgebra

function init_array(N)

    A = zeros(Float64,N, N)

    for i in 1:N
        for j in 1:i
            A[i, j] = ((-j-1) % N) / N + 1.0
        end
        for j in i+1:N
            A[i, j] = 0.0
        end
        A[i, i] = 1.0
    end

    B = zeros(Float64, N, N)
    
    for t in 1:N
        for r in 1:N
            for s in 1:N
                B[r, s] += A[r,t] * A[s,t]
            end
        end
    end

    A .= B

    return A
end

function is_valid(N, A)
    trueA = init_array(N)
    L = UnitLowerTriangular(A)[:, :]
    R = UpperTriangular(A)[:, :]
    println(norm(L*R - trueA))
    return true
end

function main() 

    N = 60
    A = init_array(N)
    lu(N, A)
    if is_valid(N, A)
        benchmarks = NPBenchManager.get_parameters("lu")
        for (preset, sizes) in benchmarks
            (N,) = collect(values(sizes))
            @dphpc_time(A = init_array(N), lu(N, A), preset)
        end
    end

end