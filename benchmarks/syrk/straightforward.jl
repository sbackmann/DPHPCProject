
include("../../timing/dphpc_timing.jl")



function init(n, k)
    A = zeros(Float64, n, k)
    C = zeros(Float64, n, n)

    for j=1:k, i=1:n
        A[i, j] = ((i*j+1)%n) / n;
    end
    reset!(C, k)
    
    return 1.5, 1.2, C, A
end

function reset!(C, k)
    n = size(C, 1)
    for j=1:n, i=1:n
        C[i, j] = ((i*j+2)%k) / k;
    end
end

# the column major version
function syrk(n, m, α, β, C, A)
    for j=1:n
        C[j:end, j] *= β
        for k=1:m
            C[j:end, j] += α * A'[k, j] * A[j:end, k]
        end
    end
end


# "S": { "M": 50, "N": 70 },
# "M": { "M": 150, "N": 200 },
# "L": { "M": 500, "N": 600 },
# "paper": { "M": 1000, "N": 1200 }

function main()

    n, k = 5, 3
    α, β, C, A = init(n, k)
    # display(C)
    # display(A)
    # @time syrk(n, k, α, β, C, A)
    @dphpc_time(reset!(C, k), syrk(n, k, α, β, C, A))
    # display(C)

    
    n, k = 70, 50
    α, β, C, A = init(n, k)
    @dphpc_time(reset!(C, k), syrk(n, k, α, β, C, A), "S")

    n, k = 200, 150
    α, β, C, A = init(n, k)
    @dphpc_time(reset!(C, k), syrk(n, k, α, β, C, A), "M")

    n, k = 600, 500
    α, β, C, A = init(n, k)
    @dphpc_time(reset!(C, k), syrk(n, k, α, β, C, A), "L")

    # n, k = 1200, 1000
    # α, β, C, A = init(n, k)
    # @dphpc_time(reset!(C, k), syrk(n, k, α, β, C, A), "paper")

end

main()