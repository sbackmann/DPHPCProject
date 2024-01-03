
include("../../timing/dphpc_timing.jl")

include("_syrk.jl")

using LinearAlgebra


syrk(n, k, α, β, C, A) = BLAS.syrk!('L', 'N', α, A, β, C) # do syrk basically using julias BLAS stuff


main()