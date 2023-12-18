
include("../../timing/dphpc_timing.jl")

include("_syrk.jl")


syrk(n, k, α, β, C, A) = C .= α .* (A*A') .+ β .* C # do syrk basically using julias BLAS stuff


main()