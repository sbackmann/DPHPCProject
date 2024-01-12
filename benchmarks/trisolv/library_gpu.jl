using LinearAlgebra
using CUDA

include("utils.jl")


function kernel(L, x, b)
    ldiv!(x, LowerTriangular(L), b)
    CUDA.synchronize()
    return x
end


main()