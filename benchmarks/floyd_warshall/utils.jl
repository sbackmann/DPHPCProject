include("../../timing/dphpc_timing.jl")
using Serialization
using CUDA
using BenchmarkTools
# CUDA.set_runtime_version!(v"12.0")
# using CUDA
# CUDA.precompile_runtime()

function init_graph(n)
    tmp = 0
    graph = zeros(Int, n, n)
    for j in 0:n-1
        for i in 0:n-1
            graph[i+1, j+1] = i * j % 7 + 1
            tmp = (i + j) % 13 == 0 || (i + j) % 7 == 0 || (i + j) % 11 == 0 ? 1 : 0
            if tmp == 0
                graph[i+1, j+1] = 999
            end
        end
    end
    return graph
end


function reset_graph(graph)
    graph_gpu = CuArray(graph)
    CUDA.synchronize()
    return graph_gpu
end


function floyd_warshall_gpu_assert!(n, graph)

    threads = 16
    threads_per_block = (threads, threads)
    blocks = (ceil(Int, n / threads), ceil(Int, n / threads))

    CUDA.@sync blocking=true (@cuda threads=threads_per_block blocks=blocks floyd_kernel_assert(graph, n))
    return graph
end


function floyd_kernel_assert(graph, n)
    k = 1:n
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    if i <= n && j <= n
        for kk in k
            if graph[i, j] < graph[i, kk] + graph[kk, j]
                graph[i, j] = graph[i, j]
            else
                graph[i, j] = graph[i, kk] + graph[kk, j]
            end
        end
    end
    return
end

function assert_naive(res, n)
    graph = init_graph(n)
    graph_gpu = reset_graph(graph)
    floyd_warshall_gpu_assert!(n, graph_gpu)
    graph_cpu = CUDA.copyto!(Array{Int}(undef, n, n), graph_gpu)
    # display(result_graph[1:50, 1:50])

    return isequal(res, graph_cpu)
end


# function create_testfile(graph, prefix)
#     open("benchmarks/floyd-warshall/test_cases/$prefix.jls", "w") do io
#         Serialization.serialize(io, graph)
#     end
# end


# function assert_correctness(graph, prefix)
#     graph_test = open("benchmarks/floyd-warshall/test_cases/$prefix.jls" ) do io
#         Serialization.deserialize(io)
#     end
#     @assert isequal(graph, graph_test)
# end

function main()

    n = 200
    graph = init_graph(n)
    graph_gpu = reset_graph(graph)
    floyd_warshall_gpu!(n, graph_gpu)
    result_graph = CUDA.copyto!(Array{Int}(undef, n, n), graph_gpu)

    if assert_naive(result_graph, n)

        @dphpc_time(graph_gpu = reset_graph(graph), floyd_warshall_gpu!(n, graph_gpu)) # warmup

        benchmark_sizes = NPBenchManager.get_parameters("floyd_warshall")

        for (preset, sizes) in benchmark_sizes
            (n,) = collect(values(sizes))
            graph = init_graph(n)
            graph_gpu = reset_graph(graph)
            # CUDA.@time floyd_warshall_gpu!(n, graph)
            @dphpc_time(graph_gpu = reset_graph(graph), floyd_warshall_gpu!(n, graph_gpu), preset)
            

        end

    else
        println("VALIDATION FAILED")
    end



end