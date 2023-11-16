include("../../timing/dphpc_timing.jl")

function init_array(N, A)
    for i in 1:N
        for j in 1:i
            A[i, j] = (-j % N) / N + 1.0
        end
        for j in i+1:N
            A[i, j] = 0.0
        end
        A[i, i] = 1.0
    end

    B = zeros(N, N)

    for t in 1:N
        for r in 1:N
            for s in 1:N
                B[r, s] += A[t, r] * A[t, s]
            end
        end
    end

    A .= B
end

function lu(N, A)
    for i in 1:N
        for j in 1:i
            for k in 1:j
                A[i, j] -= A[i, k] * A[k, j]
            end
            A[i, j] /= A[j, j]
        end
        for j in i:N
            for k in 1:i
                A[i, j] -= A[i, k] * A[k, j]
            end
        end
    end
end


function main()

    N = 60
    A = zeros(N, N)
    @dphpc_time(init_array(N, A), lu(N, A), "S")

    N = 220
    A = zeros(N, N)
    @dphpc_time(init_array(N, A), lu(N, A), "M")

    N = 700
    A = zeros(N, N)
    @dphpc_time(init_array(N, A), lu(N, A), "L")

    N = 2000
    A = zeros(N, N)
    @dphpc_time(init_array(N, A), lu(N, A), "paper")


end

main()
