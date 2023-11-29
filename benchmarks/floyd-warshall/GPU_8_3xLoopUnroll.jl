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
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if i <= n && j <= n
        # Unroll loop by factor 3
        for kk in 1:3:n-2
            @inbounds tmp = graph[i, kk] + graph[kk, j]
            @inbounds if tmp < graph[i, j]
                graph[i, j] = tmp
            end
            @inbounds tmp = graph[i, kk + 1] + graph[kk + 1, j]
            @inbounds if tmp < graph[i, j]
                graph[i, j] = tmp
            end
            @inbounds tmp = graph[i, kk + 2] + graph[kk + 2, j]
            @inbounds if tmp < graph[i, j]
                graph[i, j] = tmp
            end
        end
        @inbounds tmp = graph[i, n-1] + graph[n-1, j]
        @inbounds if tmp < graph[i, j]
            graph[i, j] = tmp
        end
        @inbounds tmp = graph[i, n] + graph[n, j]
        @inbounds if tmp < graph[i, j]
            graph[i, j] = tmp
        end
    end
    return
end

main()