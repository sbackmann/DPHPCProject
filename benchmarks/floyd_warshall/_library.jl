
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

# function floyd_warshall(n, graph)

#     for k in 1:n
#         graph[1:k, k]     .+= graph[1:k, :]     * graph[:, k] 
#         graph[k+1:end, k] .+= graph[k+1:end, :] * graph[:, k] 
#     end 

# end


function floyd_warshall(n, graph)
    t = Vector{FWNumber{Int}}(undef, n)

    for k in 1:n
        @views mul!(t, graph, graph[:, k] )
        @view(graph[:, k]) .+= t
    end 

    for k in n:-1:1
        @views mul!(t, graph, graph[:, k] )
        @view(graph[:, k]) .+= t
    end 

end



function floyd_warshall_naive(n) # to validate the validation I guess
    graph = init_graph(n)

    for k in 1:n
        for i in 1:n
            for j in 1:n
                graph[i, j] += graph[i, k] * graph[k, j]
                # Optimization column major
                #graph[j, i] = min(graph[j, i], graph[j, k] + graph[k, i])
            end
        end
    end

    return graph
end

function assert_correctness(graph, n)
    graph_test = floyd_warshall_naive(n)
    @assert isequal(graph, graph_test)
end


function main()

    benchmarks = NPBenchManager.get_parameters("floyd_warshall")

    n = (benchmarks["S"] |> values |> collect)[1]
    graph = init_graph(n)
    floyd_warshall(n, graph)
    assert_correctness(graph, n)
    
    @dphpc_time(graph = init_graph(n),floyd_warshall(n, graph),"S")

    

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