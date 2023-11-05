
include("../../timing/dphpc_timing.jl")


include("_syrk.jl")

# one to one translation from numpy version
# julia uses column major, but this is written for row major, so it's not gonna be good
function syrk(n, m, α, β, C, A)
    for i=1:n
        C[i, 1:i] *= β
        for k=1:m
            C[i, 1:i] += α * A[i, k] * A[1:i, k];
        end
    end
end


main()