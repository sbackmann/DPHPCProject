include("./utils.jl")

ASSERT = true


function floyd_warshall_gpu!(n, graph)

    threads = 16
    threads_per_block = (threads, threads)
    blocks = (ceil(Int, n / threads), ceil(Int, n / threads))

    CUDA.@sync(@cuda threads=threads_per_block blocks=blocks floyd_kernel(graph, n))
    return graph
end


function floyd_kernel(graph, n)
    k = 1:n
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    if i <= n && j <= n
        for kk in k
            # Replace graph[i, kk] + graph[kk, j] with tmp
            tmp = graph[i, kk] + graph[kk, j]
            if graph[i, j] < tmp
                graph[i, j] = graph[i, j]
            else
                graph[i, j] = tmp
            end
        end
    end
    return
end


main()