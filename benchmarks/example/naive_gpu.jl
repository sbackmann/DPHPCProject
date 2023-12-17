include("_example.jl")
using CUDA


function kernel(A, B, n, out)
    c = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    r = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    if r <= n && c <= n
        s = 0.0
        for i=1:n
            s += A[r, i] * B[i, c]
        end
        out[r, c] = s
    end
    nothing
end

function example(A, B, n, out)
    b = n ÷ 16 + 1
    @cuda threads=(16, 16) blocks=(b, b) kernel(A, B, n, out)
    synchronize()
end


function run_kernel(n, preset)
    matrices_h = init(n)
    (A, B, out) = CuArray.(matrices_h)
    @dphpc_time(
        out .= 0,
        example(A, B, n, out),
        preset
    )
end


run_kernel(100, "missing") # warmup, need to compile stuff...
main()