include("./utils.jl")

ASSERT = true


function doitgen_gpu!(nr, nq, np, A, C4, sum)
    threads = 16
    threads_per_block = (threads, threads)
    n = max(nr, nq)
    blocks = (ceil(Int, n / threads), ceil(Int, n / threads))

    @cuda threads=threads_per_block blocks=blocks doitgen_kernel(nr, nq, np, A, C4, sum)
    copyto!(A, sum)
    CUDA.synchronize()



end


function doitgen_kernel(nr, nq, np, A, C4, sum)
    s  = 1:np
    r = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    q = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if r <= nr && q <= nq
        for pp in 1:np
            temp_sum = 0.0
            for ss in s
                temp_sum += A[r, q, ss] * C4[ss, pp]
            end
            sum[r, q, pp] = temp_sum
        end
    end

    return
end


main()