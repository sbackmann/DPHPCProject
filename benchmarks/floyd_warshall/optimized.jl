include("../../timing/dphpc_timing.jl")
using Serialization

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
    return graph
end


function floyd_warshall(n, graph)

    for k in 1:n
        for i in 1:n
            # @views graph[i, 1:n] .= min.(graph[i, 1:n], graph[i, k] .+ graph[k, 1:n])


            @views graph[1:k-1, i] .= min.(graph[1:k-1, k] .+ graph[k, i], graph[1:k-1, i])
            graph[k, i] = min(graph[k, i], graph[k, k] + graph[k, i])
            @views graph[k+1:n, i] .= min.(graph[k+1:n, k] .+ graph[k, i], graph[k+1:n, i])
        end
    end

    return graph
end

function create_testfile(graph, prefix)
    open("benchmarks/floyd_warshall/test_cases/$prefix.jls", "w") do io
        Serialization.serialize(io, graph)
    end
end


function assert_correctness(graph, prefix)
    graph_test = open(joinpath(@__DIR__, "test_cases/$prefix.jls") ) do io
        Serialization.deserialize(io)
    end
    @assert isequal(graph, graph_test)
end

function main()

    benchmarks = NPBenchManager.get_parameters("floyd_warshall")

    n = (benchmarks["S"] |> values |> collect)[1]
    graph = init_graph(n)
    res = @dphpc_time(graph = init_graph(n),floyd_warshall(n, graph),"S")
    if ASSERT && res != nothing
        assert_correctness(graph, "S")
    end

    n = (benchmarks["M"] |> values |> collect)[1]
    graph = init_graph(n)
    @dphpc_time(graph = init_graph(n),floyd_warshall(n, graph),"M")

    n = (benchmarks["L"] |> values |> collect)[1]
    graph = init_graph(n)
    @dphpc_time(graph = init_graph(n),floyd_warshall(n, graph),"L")

    n = (benchmarks["paper"] |> values |> collect)[1]
    graph = init_graph(n)
    @dphpc_time(graph = init_graph(n),floyd_warshall(n, graph),"paper")

end

main()