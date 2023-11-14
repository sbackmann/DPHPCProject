include("../../timing/dphpc_timing.jl")
using Serialization
using CUDA
using BenchmarkTools
# CUDA.set_runtime_version!(v"12.0")
# using CUDA
# CUDA.precompile_runtime()
import Base.Broadcast

ASSERT = true

function init_graph(n)
    tmp = 0
    graph = zeros(Int, n, n)
    for i in 0:n-1
        for j in 0:n-1
            graph[i+1, j+1] = i * j % 7 + 1
            tmp = (i + j) % 13 == 0 || (i + j) % 7 == 0 || (i + j) % 11 == 0 ? 1 : 0
            if tmp == 0
                graph[i+1, j+1] = 999
            end
        end
    end
    graph_gpu = CUDA.fill(-1, n, n)
    CUDA.copyto!(graph_gpu, graph)
    return graph_gpu
end

function reset_graph(n)
    init_graph(n)
end


function floyd_warshall_gpu!(n, graph)

    threads = 16
    threads_per_block = (threads, threads)
    blocks = (ceil(Int, n / threads), ceil(Int, n / threads))

    @cuda threads=threads_per_block blocks=blocks floyd_kernel(graph, n)
    CUDA.synchronize()
    return graph
end


function floyd_kernel(graph, n)
    i, j, k = threadIdx().x, threadIdx().y, 1:n
    offset_i = (blockIdx().x - 1) * blockDim().x
    offset_j = (blockIdx().y - 1) * blockDim().y
    if offset_i + i <= n && offset_j + j <= n
        for kk in k
            graph[offset_i + i, offset_j + j] = min(graph[offset_i + i, offset_j + j], graph[offset_i + i, kk] + graph[kk, offset_j + j])
        end
    end
    return
end

#Broadcast.broadcasted(f::typeof(floyd_warshall_gpu!), A::AbstractArray) = floyd_warshall_gpu!(A)


function create_testfile(graph, prefix)
    open("benchmarks/floyd-warshall/test_cases/$prefix.jls", "w") do io
        Serialization.serialize(io, graph)
    end
end


function assert_correctness(graph, prefix)
    graph_test = open("benchmarks/floyd-warshall/test_cases/$prefix.jls" ) do io
        Serialization.deserialize(io)
    end
    @assert isequal(graph, graph_test)
end

function main()

    # Dummy Run so that first run is ignored
    n = 500
    graph = init_graph(n)
    floyd_warshall_gpu!(n, graph)

    n = 200
    graph = init_graph(n)
    # CUDA.@time floyd_warshall_gpu!(n, graph)
    res = @dphpc_time(init_graph(n),floyd_warshall_gpu!(n, graph),"S")
    result_graph = CUDA.copyto!(Array{Int}(undef, n, n), graph)
    # print(result_graph[1:50, 1:50])
    if ASSERT # && res != nothing
        assert_correctness(result_graph, "S")
    end

    n = 400
    graph = init_graph(n)
    @dphpc_time(init_graph(n),floyd_warshall_gpu!(n, graph),"M")

    n = 800
    graph = init_graph(n)
    @dphpc_time(init_graph(n),floyd_warshall_gpu!(n, graph),"L")

    n = 2800
    graph = init_graph(n)
    @dphpc_time(init_graph(n),floyd_warshall_gpu!(n, graph),"paper")

end

main()
