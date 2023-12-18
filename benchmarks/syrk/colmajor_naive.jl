
include("../../timing/dphpc_timing.jl")

include("_syrk.jl")

# the column major version
function syrk(n, m, α, β, C, A)
    for j=1:n
        C[j:end, j] *= β
        for k=1:m
            C[j:end, j] += α * A'[k, j] * A[j:end, k]
        end
    end
end


main()