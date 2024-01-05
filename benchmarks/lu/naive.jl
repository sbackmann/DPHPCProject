include("../../timing/dphpc_timing.jl")
using LinearAlgebra



# This function works exactly as the C version
# function lu(N, A)
#     i = 1
#     while i <= N
#         j = 1
#         while j < i
#             k = 1
#             while k < j
#                 A[i, j] -= A[i, k] * A[k, j]
#                 k += 1
#             end
#             A[i, j] /= A[j, j]
#             j += 1
#         end
#         j = i
#         while j <= N
#             k = 1
#             while k < i
#                 A[i, j] -= A[i, k] * A[k, j]
#                 k += 1
#             end
#             j += 1
#         end
#         i += 1
#     end
# end

function LinearAlgebra.lu(N::Int, A)
    for i in 1:N
        for j in 1:i-1
            for k in 1:j-1
                A[i, j] -= A[i, k] * A[k, j]
            end
            A[i, j] /= A[j, j]
        end
        
        for j in i:N
            for k in 1:i-1
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
