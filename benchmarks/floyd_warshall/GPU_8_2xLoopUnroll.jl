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
        # Unroll loop by factor 2
        @inbounds for kk in 1:2:n-1
            tmp = graph[i, kk] + graph[kk, j]
            
            tmp2 = graph[i, kk + 1] + graph[kk + 1, j]
            
            tmp_min = min(tmp, tmp2)
            if tmp_min < graph[i, j]
                graph[i, j] = tmp_min
            end
        end
        @inbounds tmp = graph[i, n] + graph[n, j]
        @inbounds if tmp < graph[i, j]
            graph[i, j] = tmp
        end
    end
    return
end

main()