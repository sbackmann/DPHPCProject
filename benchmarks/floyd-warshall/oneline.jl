include("../../timing/dphpc_timing.jl")
using Serialization

ASSERT = true


init_graph(n) = init_graph(n, zeros(Int, n, n))

function init_graph(n, graph)
    tmp = 0
    for i in 0:n-1
        for j in 0:n-1
            graph[i+1, j+1] = i * j % 7 + 1
            tmp = (i + j) % 13 == 0 || (i + j) % 7 == 0 || (i + j) % 11 == 0 ? 1 : 0
            if tmp == 0
                graph[i+1, j+1] = 999
            end
        end
    end
    return graph
end

function reset_graph(n, graph)
    init_graph(n, graph)
end

function floyd_warshall(n, graph)

    for k in 1:n
        for i in 1:n
            graph[i, :] = min.(graph[i, k] .+ graph[k, :], graph[i, :])
        end
    end

    return graph
end

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

    n = 200
    graph = init_graph(n)
    @dphpc_time(init_graph(n, graph),floyd_warshall(n, graph),"S")
    if ASSERT
        assert_correctness(graph, "S")
    end

    n = 400
    graph = init_graph(n)
    @dphpc_time(init_graph(n, graph),floyd_warshall(n, graph),"M")

    n = 800
    graph = init_graph(n)
    @dphpc_time(init_graph(n, graph),floyd_warshall(n, graph),"L")

    n = 2800
    graph = init_graph(n)
    @dphpc_time(init_graph(n),floyd_warshall(n, graph),"paper")

end

main()