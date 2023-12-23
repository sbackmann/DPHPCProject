include("_example.jl")
using CUDA

example(A, B, n, out) = out .= A * B # CUDA matrix mult

main(gpu=true)
