include("./utils.jl")

ASSERT = true


function doitgen_gpu!(nr, nq, np, A, C4, sum)
    threads = 16
    threads_per_block = (threads, threads)
    n = max(nr, nq)
    blocks = (ceil(Int, n / threads), ceil(Int, n / threads))

    CUDA.@sync(@cuda threads=threads_per_block blocks=blocks doitgen_kernel(nr, nq, np, A, C4, sum))    

end


function doitgen_kernel(nr, nq, np, A, C4, sum)
    p, s = 1:np, 1:np
    r = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    q = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if r <= nr && q <= nq
        for pp in p
            for ss in s
                sum[r, q, ss] += A[r, q, pp] * C4[pp, ss]
            end
        end

        for pp in p
            A[r, q, pp] = sum[r, q, pp]
        end
    end

    return
end


main()