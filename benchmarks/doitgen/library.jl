include("../../timing/dphpc_timing.jl")
using Serialization
using LinearAlgebra: mul!

ASSERT = true


function init_array(nr, nq, np, A=nothing, C4=nothing)
    if A == nothing
        A = zeros(Float64, nr, nq, np)
        C4 = zeros(Float64, np, np)
    end
    for i in 0:nr-1
        for j in 0:nq-1
            for k in 0:np-1
                A[i+1, j+1, k+1] = ((i * j + k) % np) / np
            end
        end
    end
    for i in 0:np-1
        for j in 0:np-1
            C4[i+1, j+1] = (i * j % np) / np
        end
    end
    sum = zeros(Float64, np)
    return A, C4, sum
end


function reset(nr, nq, np, A, C4)
    init_array(nr, nq, np, A, C4)
end



function doitgen(nr, nq, np, A, C4, sum)
    for r in 1:nr
        for q in 1:nq
            mul!(sum, C4', @view(A[r, q, :]))
            @view(A[r, q, :]) .= sum
        end
    end
    return A
end


function create_testfile(A, prefix)
    open("benchmarks/doitgen/test_cases/$prefix.jls", "w") do io
        Serialization.serialize(io, A)
    end
end


function assert_correctness(A, prefix)
    A_test = open("benchmarks/doitgen/test_cases/$prefix.jls" ) do io
        Serialization.deserialize(io)
    end
    @assert isequal(A, A_test)
end

function main()

    benchmark_sizes = NPBenchManager.get_parameters("doitgen")

    nr,nq,np = 60, 60, 128
    (A, C4, sum) = init_array(nr, nq, np)
    @dphpc_time((A, C4, sum) = init_array(nr, nq, np), doitgen(nr, nq, np, A, C4, sum)) #warmup

    for (preset, dims) in benchmark_sizes
        nr,nq,np = dims |> values |> collect
        (A, C4, sum) = init_array(nr, nq, np)

        @dphpc_time((A, C4, sum) = init_array(nr, nq, np), doitgen(nr, nq, np, A, C4, sum), preset)
    end

end

main()