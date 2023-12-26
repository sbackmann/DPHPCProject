include("../../timing/dphpc_timing.jl")

include("_syrk.jl")

using CUDA

# C .= α*A*A' + β*C

run_kernel(n, k, α, β, C, A) = CUDA.@sync(C .= α .* (A*A') .+ β .* C) # does twice the work though...


main_gpu()
