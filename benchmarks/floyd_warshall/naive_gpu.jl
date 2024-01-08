include("./utils.jl")

ASSERT = false

function floyd_warshall_gpu!(n, graph)

    threads = 16
    threads_per_block = (threads, threads)
    blocks = (ceil(Int, n / threads), ceil(Int, n / threads))

    for k in 1:n
        CUDA.@sync blocking=true (@cuda threads=threads_per_block blocks=blocks floyd_kernel(graph, n, k))
    end
    return graph
end


function floyd_kernel(graph, n, k)
    
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    if i <= n && j <= n
        if graph[i, j] < graph[i, k] + graph[k, j]
            graph[i, j] = graph[i, j]
        else
            graph[i, j] = graph[i, k] + graph[k, j]
        end
    end
    return
end

#=

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
        for kk in 1:n
            if graph[i, j] < graph[i, kk] + graph[kk, j]
                graph[i, j] = graph[i, j]
            else
                graph[i, j] = graph[i, kk] + graph[kk, j]
            end
        end
    end
    return
end
=#

main()
