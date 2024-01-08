include("./utils.jl")

ASSERT = true

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
        # evaluate to temporary variable and remove obsolete else case
        tmp1 = graph[i, k] + graph[k, j]
        tmp = graph[i, j] < tmp1
        graph[i, j] = tmp * graph[i, j] + (1 - tmp) * tmp1
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
    k = 1:n
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    if i <= n && j <= n
        for kk in k
            # evaluate to temporary variable and remove obsolete else case
            tmp1 = graph[i, kk] + graph[kk, j]
            tmp = graph[i, j] < tmp1
            graph[i, j] = tmp * graph[i, j] + (1 - tmp) * tmp1
        end
    end
    return
end
=#

main()