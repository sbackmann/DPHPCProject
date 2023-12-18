
include("../../timing/dphpc_timing.jl")
using Serialization
using LinearAlgebra

ASSERT = false


struct FWNumber{T <: Number}
    value::T
end

# floyd warshall is like matrix multiplication, only that instead of *, do + between elements, and instead of summation for reduction, do min()

Base.:(+)(a::FWNumber{T}, b::FWNumber{T}) where T = FWNumber(min(a.value, b.value))
Base.:(*)(a::FWNumber{T}, b::FWNumber{T}) where T = FWNumber(a.value + b.value)
Base.zero(::Type{FWNumber{T}}) where T = FWNumber(T(999))
Base.zero(::FWNumber{T}) where T = FWNumber(T(999))
Base.Int(a::FWNumber{T}) where T <: Integer = a.value

init_graph(n) = init_graph(n, zeros(FWNumber{Int}, n, n))

function init_graph(n, graph)
    tmp = 0
    for i in 0:n-1
        for j in 0:n-1
            graph[i+1, j+1] = FWNumber(i * j % 7 + 1)
            tmp = (i + j) % 13 == 0 || (i + j) % 7 == 0 || (i + j) % 11 == 0 ? 1 : 0
            if tmp == 0
                graph[i+1, j+1] = FWNumber(999)
            end
        end
    end
    return graph
end

function reset_graph(n, graph)
    init_graph(n, graph)
end

floyd_warshall(n, graph) = for it in (1:n, n:-1:1) for k in it graph[:, k] .+= graph * graph[:, k] end end # I know it's not pretty, but it works


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
    floyd_warshall(n, graph)
    if ASSERT
        assert_correctness(Int.(graph), "S")
    end

    n = 400
    graph = init_graph(n)
    @dphpc_time(init_graph(n, graph),floyd_warshall(n, graph),"M")

    n = 850
    graph = init_graph(n)
    @dphpc_time(init_graph(n, graph),floyd_warshall(n, graph),"L")

    n = 2800
    graph = init_graph(n)
    @dphpc_time(init_graph(n),floyd_warshall(n, graph),"paper")

end

main()