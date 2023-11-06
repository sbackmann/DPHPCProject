
include("../../timing/dphpc_timing.jl")


include("_syrk.jl")

# the column major version, with some dots and views
function syrk(n, m, α, β, C, A)
    for j=1:n
        @views C[j:end, j] .*= β
        for k=1:m
            @views C[j:end, j] .+= α .* A[j, k] .* A[j:end, k]
        end
    end
end


main()