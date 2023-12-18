
include("../../timing/dphpc_timing.jl")


include("_syrk.jl")

# one to one translation from C version
# julia uses column major, C row major, so this not gonna be good
function syrk(n, m, α, β, C, A)
    for i=1:n
        for j=1:i
            C[i, j] = C[i, j] * β
        end
        for k=1:m, j=1:i
            C[i, j] = C[i, j] + α * A[i, k] * A[j, k]
        end
    end
end


main()