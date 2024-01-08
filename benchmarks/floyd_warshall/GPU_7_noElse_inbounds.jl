include("./utils.jl")

ASSERT = true

function floyd_warshall_gpu!(n, graph)

    threads = 16
    threads_per_block = (threads, threads)
    blocks = (ceil(Int, n / threads), ceil(Int, n / threads))

    @inbounds for k in 1:n
        CUDA.@sync blocking=true (@cuda threads=threads_per_block blocks=blocks floyd_kernel(graph, n, k))
    end
    return graph
end


function floyd_kernel(graph, n, k)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    if i <= n && j <= n
        # Remove bounds checking
        @inbounds tmp = graph[i, k] + graph[k, j]
        @inbounds if tmp < graph[i, j]
            @inbounds graph[i, j] = tmp
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
        @inbounds for kk in 1:n
            # Remove bounds checking
            tmp = graph[i, kk] + graph[kk, j]
            if tmp < graph[i, j]
                graph[i, j] = tmp
            end
        end
    end
    return
end
=#
main()