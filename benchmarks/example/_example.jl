
include("../../timing/dphpc_timing.jl")

function init(n)
    A = zeros(Float64, n, n)
    B = zeros(Float64, n, n)
    out = zeros(Float64, n, n)
    for c = 1:n, r = 1:n
        A[r, c] = r*c*3 % n
        B[r, c] = r*c*7 % n
    end
    return A, B, out
end

function run_kernel(n, preset)
    @dphpc_time(
        (A, B, out) = init(n),
        example(A, B, n, out),
        preset
    )
end


function main()
    
    benchmarks = Dict(
        "S" => 100,
        "M" => 400, 
        "L" => 800,
        "paper" => 1600
    )

    for (preset, n) in benchmarks
        run_kernel(n, preset)
    end

end
