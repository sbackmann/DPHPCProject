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
    return A
end

function validate_cpu(result)

    A = init_array(30)
    naive_result = lu(30,A)

    if naive_result == result 
        return "Valid"
    else 
        return "Not Valid"

    end 
end 