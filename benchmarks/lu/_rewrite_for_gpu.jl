include("../../timing/dphpc_timing.jl")
using LinearAlgebra



function LinearAlgebra.lu(N::Int, A)
    B = 6 # choose B optimally...
    s = zeros(N-B)
    for i in 1:N

        for j in 1:min(B-1, i-1)

            S = 0.0
            for k in 1:j-1
                S += A[i, k] * A[k, j]
            end
            A[i, j] = (A[i, j] - S) / A[j, j]
        end

        if i-1 >= B
            @view(s[1:i-B]) .= 0
            
            for k in 1:B-1                          
                for j in B:i-1                      #     
                    s[j-B+1] += A[i, k] * A[k, j]   #   can be parallelized... 
                end                                 #   it's not exactly a lot...
            end                                     

            for j in B:i-1 # this cannot be parallelized, it's kind of a lot...
                for k in B:j-1
                    s[j-B+1] += A[i, k] * A[k, j]
                end
                A[i, j] = (A[i, j] - s[j-B+1]) / A[j, j]
            end
        end
        
        for k in 1:i-1
            for j in i:N                        # 
                A[i, j] -= A[i, k] * A[k, j]    # can also be parallelized
            end                                 # 
        end
    end
end




include("_main_cpu.jl")

main()
