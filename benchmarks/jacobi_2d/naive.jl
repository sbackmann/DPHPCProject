include("../../timing/dphpc_timing.jl")
using Printf


function init_arrays(n)
    A = zeros(Float64, n, n)
    B = zeros(Float64, n, n)

    A = [((i - 1) * (j + 1) + 2) / n for i in 1:n, j in 1:n]
    B = [((i - 1) * (j + 2) + 3) / n for i in 1:n, j in 1:n]

    return A, B
end


function kernel_j2d(tsteps, n, A, B)
    for t in 1:tsteps
        for i in 2:(n-1)
            for j in 2:(n-1)
                B[i, j] = 0.2 * (A[i, j] + A[i, j-1] + A[i, j+1] + A[i+1, j] + A[i-1, j])
            end
        end
        for i in 2:(n-1)
            for j in 2:(n-1)
                A[i, j] = 0.2 * (B[i, j] + B[i, j-1] + B[i, j+1] + B[i+1, j] + B[i-1, j])
            end
        end
    end
end


function print_array(A)
    #display(stdout, round.(A; digits=2))
    #show(stdout, "text/plain", round.(A; digits=2))
    for i in 1:size(A, 1)
        for j in 1:size(A, 2)
            @printf("%.2f ", A[i, j])
        end
        println()
    end
end

include("_main_cpu.jl")

main()