include("../../timing/dphpc_timing.jl")
using CUDA 

# based on _rewrite_for_gpu.jl

function kernel1(A, i, B)

    for j in 1:min(B-1, i-1)

        S = 0.0
        for k in 1:j-1
            S += A[i, k] * A[k, j]
        end
        A[i, j] = (A[i, j] - S) / A[j, j]
    end

    nothing
end

function kernel2(A, s, i, B, k)
    id = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = id-1+B

    # for j in B:i-1 
    if B <= j <= i-1
        s[j-B+1] += A[i, k] * A[k, j] 
    end

    nothing
end

function kernel3(A, s, i, B)
    for j in B:i-1 # this cannot be parallelized, it's kind of a lot...
        for k in B:j-1
            s[j-B+1] += A[i, k] * A[k, j]
        end
        A[i, j] = (A[i, j] - s[j-B+1]) / A[j, j]
    end

    nothing
end

function kernel4(A, i, k, N)
    id = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = id-1+i

    # for j in i:N 
    if i <= j <= N
        A[i, j] -= A[i, k] * A[k, j]
    end

    nothing
end


function run_lu_kernel(N, A)
    B = N * 0.3 |> round |> Int
    s = CuArray(zeros(N-B))
    threads_per_block = 64
    for i in 1:N

        @cuda threads=1 blocks=1 kernel1(A, i, B)                                    # min(B, i)^2

        if i-1 >= B
            @view(s[1:i-B]) .= 0                                                     # 1
            
            for k in 1:B-1                                                           # B
                total_threads = i-B
                blocks = (total_threads-1) ÷ threads_per_block + 1                 
                @cuda threads=threads_per_block blocks=blocks kernel2(A, s, i, B, k)
            end 

            @cuda threads=1 blocks=1 kernel3(A, s, i, B)                             # (i-B)^2
            
        end
        
        for k in 1:i-1                                                               # i
            total_threads = N-i+1
            blocks = (total_threads-1) ÷ threads_per_block + 1                 
            @cuda threads=threads_per_block blocks=blocks kernel4(A, i, k, N)
        end
    end
    CUDA.synchronize()
end

ops(N, B) = sum(min(B, i)^2 + (i > B ? (1 + B + (i-B)^2) : 0) + i for i = 1:N)
plot_ops(N) = plot(1:N, [ops(N, x) for x in 1:N]) # B≈0.3N seems to give the fewest operations..


include("_main_gpu.jl")

main()
