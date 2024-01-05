include("../../timing/dphpc_timing.jl")

using LinearAlgebra

function LinearAlgebra.lu(N::Int, A)
    dcp = lu(A, NoPivot())
    A .= dcp.L + dcp.U - I 
end # builtin LU, return same result as c implementation

include("_main_cpu.jl")

main()
