#= DOES NOT RESOLVE RACE CONDITION - BUGGY

include("./utils.jl")

ASSERT = true


function floyd_warshall_gpu!(n, graph)

    threads = 16
    threads_per_block = (threads, threads)
    blocks = (ceil(Int, n / threads), ceil(Int, n / threads))

    CUDA.@sync blocking=true (@cuda threads=threads_per_block blocks=blocks floyd_kernel(graph, n))
    return graph
end


function floyd_kernel(graph, n)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if i <= n && j <= n
        # Unroll loop by factor 4
        @inbounds for kk in 1:4:n-3
            tmp = graph[i, kk] + graph[kk, j]
            tmp2 = graph[i, kk + 1] + graph[kk + 1, j]
            tmp3 = graph[i, kk + 2] + graph[kk + 2, j]
            tmp4 = graph[i, kk + 3] + graph[kk + 3, j]
            
            tmp_min = min(tmp, tmp2, tmp3, tmp4)
            if tmp_min < graph[i, j]
                graph[i, j] = tmp_min
            end
        end

        @inbounds for kk in n-2:n
            tmp = graph[i, kk] + graph[kk, j]
            if tmp < graph[i, j]
                graph[i, j] = tmp
            end
        end
        
    end
    return
end

main()
=#