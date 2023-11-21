include("../../timing/dphpc_timing.jl")
using CUDA 


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

    return CuArray(A)
end



function lu_kernel(N, A)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x

    if i <= N
        for j in 1:i
            for k in 1:j
                A[i, j] = A[i, j] - (A[i, k] * A[k, j])
            end
            A[i, j] = A[i, j] / A[j, j]
        end

        for j in i:N
            for k in 1:i
                A[i, j] = A[i, j] - (A[i, k] * A[k, j])
            end
        end
    end
    return
end


function run_lu_kernel(N, A)
    threadsPerBlock = (16, 16)
    numBlocks = ((N - 1) ÷ 16 + 1, (N - 1) ÷ 16 + 1)

    @cuda threads=threadsPerBlock blocks=numBlocks lu_kernel(N, A)
    CUDA.synchronize()

end





function main()

    N = 60
    A = zeros(N, N)
    A = init_array(N, A)
    @dphpc_time(A, run_lu_kernel(N, A), "S")

    N = 220
    A = zeros(N, N)
    A = init_array(N, A)
    @dphpc_time(A, run_lu_kernel(N, A), "M")

    N = 700
    A = zeros(N, N)
    A = init_array(N, A)
    @dphpc_time(A, run_lu_kernel(N, A), "L")

    N = 2000
    A = zeros(N, N)
    A = init_array(N, A)
    @dphpc_time(A, run_lu_kernel(N, A), "paper")


end

main()
