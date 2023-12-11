include("./utils.jl")

ASSERT = true
const THREADS = 32



function doitgen_gpu!(nr, nq, np, A, C4, sum)
    
    threads_per_block = (THREADS * THREADS)
    blocks = (ceil(Int, np / THREADS), ceil(Int, nr / THREADS))

    CUDA.@sync(@cuda threads=threads_per_block blocks=blocks doitgen_kernel(nr, nq, np, A, C4, sum))    
    copyto!(A, sum)
    CUDA.synchronize()
end


function doitgen_kernel(nr, nq, np, A, C4, sum)
    p = (blockIdx().x - 1) * THREADS + ((threadIdx().x - 1) ÷ THREADS) + 1
    r = (blockIdx().y - 1) * THREADS + ((threadIdx().x - 1) % THREADS) + 1
    odd = np % 2 == 1

    if p <= np && r <= nr
        @inbounds for q in 1:nq
            temp_sum = 0.0
            @inbounds for ss in 1:2:np-1
                temp_sum += A[r, q, ss] * C4[ss, p]
                temp_sum += A[r, q, ss + 1] * C4[ss + 1, p]
            end
            if odd
                temp_sum += A[r, q, np] * C4[np, p]
            end
            sum[r, q, p] = temp_sum
        end

    end

    return
end


main()