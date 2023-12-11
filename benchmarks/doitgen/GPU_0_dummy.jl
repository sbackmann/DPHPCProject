include("./utils.jl")


ASSERT = false


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
            sum[r, q, pp] = 0.0
            for ss in s
                sum[r, q, pp] += A[r, q, ss] * C4[ss, pp]
            end
        end

        for pp in p
            A[r, q, pp] = sum[r, q, pp]
        end
    end

    return
end


function main2()
    nr = 60
    nq = 60
    np = 128
    A, C4 = init_array(nr, nq, np)
    A_gpu, C4_gpu, sum = reset(A, C4, nr, nq, np)
    @dphpc_time((A_gpu, C4_gpu, sum) = reset(A, C4, nr, nq, np), doitgen_gpu!(nr, nq, np, A_gpu, C4_gpu, sum), "missing")

end

main2()