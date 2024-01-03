
function is_valid(n, k, C, A)
    alpha = 3.
    beta = 5.
    A .= 1
    C .= 1

    syrk(n, k, alpha, beta, C, A)

    C_h = zeros(n, n)
    copyto!(C_h, C)

    E = beta + alpha*k
    
    for c=1:n, r=c:n
        if C_h[r, c] != E
            return false
        end
    end
    return true
end

function is_valid_gpu(n, k, C, A)
    alpha = 3.
    beta = 5.
    A .= 1
    C .= 1

    run_kernel(n, k, alpha, beta, C, A)
    
    C_h = zeros(n, n)
    copyto!(C_h, C)

    E = beta + alpha*k
    
    for c=1:n, r=c:n
        if C_h[r, c] != E
            error("validation failed...")
            return false
        end
    end
    return true
end