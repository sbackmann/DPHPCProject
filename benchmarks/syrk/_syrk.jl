
include("_validate.jl")

function init(n, k)
    A = zeros(Float64, n, k)
    C = zeros(Float64, n, n)

    for j=1:k, i=1:n
        A[i, j] = ((i*j+1)%n) / n;
    end
    reset!(C, k)
    
    return 1.5, 1.2, C, A
end


function init_gpu(n, k)
    hostA = zeros(Float64, n, k)
    hostC = zeros(Float64, n, n)

    for j=1:k, i=1:n
        hostA[i, j] = ((i*j+1)%n) / n;
    end
    for j=1:n, i=1:n
        hostC[i, j] = ((i*j+2)%k) / k;
    end
    
    return 1.5, 1.2, hostC, CuArray(hostC), CuArray(hostA)
end



function reset!(C, k)
    n = size(C, 1)
    for j=1:n, i=1:n
        C[i, j] = ((i*j+2)%k) / k;
    end
end


# "S": { "M": 50, "N": 70 },
# "M": { "M": 150, "N": 200 },
# "L": { "M": 500, "N": 600 },
# "paper": { "M": 1000, "N": 1200 }

function main()

    n, k = 200, 70
    α, β, C, A = init(n, k)

    if is_valid(n, k, C, A)

        n, k = 10, 5
        @dphpc_time((α, β, C, A) = init(n, k), syrk(n, k, α, β, C, A)) # warmup

        n, k = 70, 50
        @dphpc_time((α, β, C, A) = init(n, k), syrk(n, k, α, β, C, A), "S")

        n, k = 200, 150
        @dphpc_time((α, β, C, A) = init(n, k), syrk(n, k, α, β, C, A), "M")

        n, k = 600, 500
        @dphpc_time((α, β, C, A) = init(n, k), syrk(n, k, α, β, C, A), "L")

        n, k = 1200, 1000
        @dphpc_time((α, β, C, A) = init(n, k), syrk(n, k, α, β, C, A), "paper")

    end

end

function main_gpu()

    n, k = 200, 70
    α, β, hostC, C, A = init_gpu(n, k)

    if is_valid_gpu(n, k, C, A)

        n, k = 70, 40
        @dphpc_time(
            (α, β, hostC, C, A) = init_gpu(n, k),   # warmup
            run_kernel(n, k, α, β, C, A)
        )

        n, k = 70, 50
        @dphpc_time(
            (α, β, hostC, C, A) = init_gpu(n, k), 
            run_kernel(n, k, α, β, C, A), 
            "S"
        )

        n, k = 200, 150
        @dphpc_time(
            (α, β, hostC, C, A) = init_gpu(n, k), 
            run_kernel(n, k, α, β, C, A), 
            "M"
        )

        n, k = 600, 500
        @dphpc_time(
            (α, β, hostC, C, A) = init_gpu(n, k), 
            run_kernel(n, k, α, β, C, A), 
            "L"
        )

        n, k = 1200, 1000
        @dphpc_time(
            (α, β, hostC, C, A) = init_gpu(n, k), 
            run_kernel(n, k, α, β, C, A), 
            "paper"
        )

    end

end
