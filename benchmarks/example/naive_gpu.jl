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
    CUDA.@sync(@cuda threads=(16, 16) blocks=(b, b) kernel(A, B, n, out))
end


main(gpu=true)