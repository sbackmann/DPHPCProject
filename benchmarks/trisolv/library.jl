using LinearAlgebra

include("utils_cpu.jl")


kernel(L, x, b) = ldiv!(x, LowerTriangular(L), b)


main()