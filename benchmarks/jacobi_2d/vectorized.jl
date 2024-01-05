include("../../timing/dphpc_timing.jl")
using Printf


# doesnt really use library functions...

function init_arrays(n)
    A = zeros(Float64, n, n)
    B = zeros(Float64, n, n)

    A = [((i - 1) * (j + 1) + 2) / n for i in 1:n, j in 1:n]
    B = [((i - 1) * (j + 2) + 3) / n for i in 1:n, j in 1:n]

    return A, B
end


function kernel_j2d(tsteps, n, A, B)
    cA, lA, rA, tA, bA = @views (A[2:n-1, 2:n-1], A[2:n-1, 1:n-2], A[2:n-1, 3:n], A[1:n-2, 2:n-1], A[3:n, 2:n-1]) # precompute views...
    cB, lB, rB, tB, bB = @views (B[2:n-1, 2:n-1], B[2:n-1, 1:n-2], B[2:n-1, 3:n], B[1:n-2, 2:n-1], B[3:n, 2:n-1])
    for t in 1:tsteps
        @. cB = 0.2 * (cA + lA + rA + tA + bA)
        @. cA = 0.2 * (cB + lB + rB + tB + bB)
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