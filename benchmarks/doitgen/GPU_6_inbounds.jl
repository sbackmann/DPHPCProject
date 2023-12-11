include("./utils.jl")

ASSERT = true


function doitgen_gpu!(nr, nq, np, A, C4, sum)
    threads = 16
    threads_per_block = (threads, threads)
    n = max(nr, nq)
    blocks = (ceil(Int, n / threads), ceil(Int, n / threads))

    CUDA.@sync blocking=true (@cuda threads=threads_per_block blocks=blocks doitgen_kernel(nr, nq, np, A, C4, sum))    

end


function doitgen_kernel(nr, nq, np, A, C4, sum)
    p, s = 1:np, 1:np
    r = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    q = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if r <= nr && q <= nq
        @inbounds for pp in p
            temp_sum = 0.0
            @inbounds for ss in s
                temp_sum += A[r, q, ss] * C4[ss, pp]
            end
            sum[r, q, pp] = temp_sum
        end

        @inbounds for pp in p
            A[r, q, pp] = sum[r, q, pp]
        end
    end

    return
end


main()