include("../../timing/dphpc_timing.jl")

using LinearAlgebra



function LinearAlgebra.lu(N::Int, A)
    for j in 1:N
        for i in 1:j-1
            for k in 1:i-1
                A[i, j] -= A[i, k] * A[k, j]
            end
            A[i, j] /= A[j, j]
        end
        for i in (j + 1):N
            for k in 1:j-1
                A[i, j] -= A[i, k] * A[k, j]
            end
        end
    end
end





function printMatrix(matrix)
    for i in 1:size(matrix, 1)
        for j in 1:size(matrix, 2)
            print(matrix[i, j], "\t")
        end
        println()
    end
end


include("_main_cpu.jl")

main()

