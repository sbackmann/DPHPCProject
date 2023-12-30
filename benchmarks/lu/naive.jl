include("../../timing/dphpc_timing.jl")

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

function lu(N, A)
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

function main()

    N = 60
    A = init_array(N)
    @dphpc_time(A = init_array(N), lu(N, A), "S")


    N = 220
    @dphpc_time(A = init_array(N), lu(N, A), "M")

    N = 700
    @dphpc_time(A = init_array(N), lu(N, A), "L")

    N = 2000
    @dphpc_time(A = init_array(N), lu(N, A), "paper")



end

main()
