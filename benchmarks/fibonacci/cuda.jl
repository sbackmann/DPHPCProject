include("../../timing/dphpc_timing.jl")
using CUDA

function gpufib!(out, n)
    if n <= 2
        out[1] = 1
    else
        sₙ = 1
        sₙ₋₁ = 1
        for i=3:n
            sₙ₋₁, sₙ = sₙ, sₙ + sₙ₋₁ 
        end
        out[1] = sₙ
    end
    return nothing
end

run_kernel(n, out) = CUDA.@sync(@cuda threads=1 gpufib!(out, n))

function main() 
    out = CUDA.zeros(Int, 1)

    # println(out)

    @dphpc_time run_kernel(38, out)

    # println(out)

    @dphpc_time nothing run_kernel(1000, out) "S"
    @dphpc_time nothing run_kernel(10000, out) "M"
    @dphpc_time nothing run_kernel(100000, out) "L"
    

end

main()