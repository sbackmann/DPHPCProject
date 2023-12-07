include("./utils.jl")

ASSERT = false


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
            # Use min function instead of if-else statement
            graph[i, j] = min(graph[i, j], graph[i, kk] + graph[kk, j])
        end
    end
    return
end

function main2()

    # Dummy Run so that first run is ignored
    n = 200
    graph = init_graph(n)
    graph_gpu = reset_graph(graph)
    @dphpc_time(graph_gpu = reset_graph(graph), floyd_warshall_gpu!(n, graph_gpu), "missing")

end

main2()
