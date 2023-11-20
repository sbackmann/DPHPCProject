include("_example.jl")
using CUDA

example(A, B, n, out) = out .= A * B # CUDA matrix mult

function run_kernel(n, preset)
    matrices_h = init(n)
    @dphpc_time(
        (A, B, out) = CuArray.(matrices_h),
        example(A, B, n, out),
        preset
    )
end


main()
