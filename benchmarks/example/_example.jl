
include("../../timing/dphpc_timing.jl")
if !isdefined(Main, :NPBenchManager) include("../../timing/NPBenchManager.jl") end

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

function run_kernel_gpu(n, preset)
    matrices_h = init(n)
    @dphpc_time(
        (A, B, out) = CuArray.(matrices_h),
        example(A, B, n, out),
        preset
    )
end


function is_valid()
    n = 50
    A = ones(Int, (n, n))
    B = ones(Int, (n, n))
    out = zeros(Int, (n, n))
    example(A, B, n, out)
    return all(out .== n)
end

function is_valid_gpu()
    n = 50
    A = ones(Int, (n, n))
    B = ones(Int, (n, n))
    out = zeros(Int, (n, n))
    (A_d, B_d, out_d) = CuArray.((A, B, out))
    example(A_d, B_d, n, out_d)
    copyto!(out, out_d)
    return all(out .== n)
end


function main(;gpu=false)
    
    benchmarks = NPBenchManager.get_parameters("example")

    if gpu
        if is_valid_gpu()
            run_kernel_gpu(20, "missing") # warmup
            for (preset, sizes) in benchmarks
                (n,) = collect(values(sizes))
                run_kernel_gpu(n, preset)
            end
        end
    else
        if is_valid()
            run_kernel(20, "missing") # warmup
            for (preset, sizes) in benchmarks
                (n,) = collect(values(sizes))
                run_kernel(n, preset)
            end
        end
    end

end
