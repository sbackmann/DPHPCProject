include("../../timing/dphpc_timing.jl")
using CUDA 
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

    return CuArray(A)
end




function run_lu_kernel(N, A)
    dcp = lu(A)
    CUDA.@sync(result = dcp.L + dcp.U - I) 
    result
end



function main()

    N = 5
    A = init_array(N)
    @dphpc_time(A=init_array(N), run_lu_kernel(N, A)) # warmup

    N = 60
    A = init_array(N)
    @dphpc_time(A=init_array(N), run_lu_kernel(N, A), "S")

    N = 220
    A = init_array(N)
    @dphpc_time(A=init_array(N), run_lu_kernel(N, A), "M")

    N = 700
    A = init_array(N)
    @dphpc_time(A=init_array(N), run_lu_kernel(N, A), "L")

    N = 2000
    A = init_array(N)
    @dphpc_time(A=init_array(N), run_lu_kernel(N, A), "paper")


end

main()
