
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

    return CuArray(A * A')
end

function is_valid(N, A)
    trueA = init_array(N)
    L = UnitLowerTriangular(A[:, :])
    R = UpperTriangular(A[:, :])
    difference = norm(L*R - trueA)
    println(difference)
    if difference < 1e-6
        return true
    else
        # display(A)
        println("VALIDATION FAILED")
        return false
    end

end

function main() 

    N = 300
    A = init_array(N)
    run_lu_kernel(N, A)
    if is_valid(N, A)
        A = init_array(N)
        @dphpc_time(A=init_array(N), run_lu_kernel(N, A)) # warmup

        benchmarks = NPBenchManager.get_parameters("lu")
        for (preset, sizes) in benchmarks
            (N,) = collect(values(sizes))
            @dphpc_time(A=init_array(N), run_lu_kernel(N, A), preset)
        end
    end

end