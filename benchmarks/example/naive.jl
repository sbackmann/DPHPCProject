
include("_example.jl")

function example(A, B, n, out)
    for c = 1:n
        for r = 1:n
            for i = 1:n
                out[r, c] += A[r, i] * B[i, c]
            end
        end
    end
end

main()