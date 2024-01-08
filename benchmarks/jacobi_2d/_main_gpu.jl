
function init_arrays(n)
    A = zeros(Float64, n, n)
    B = zeros(Float64, n, n)

    A = [((i - 1) * (j + 1) + 2) / n for i in 1:n, j in 1:n]
    B = [((i - 1) * (j + 2) + 3) / n for i in 1:n, j in 1:n]

    return A, B
end


function reset(A, B)
    return CuArray(A), CuArray(B)
end

function print_array(A)
    for i in 1:size(A, 1)
        for j in 1:size(A, 2)
            @printf("%.2f ", A[i, j])
        end
        println()
    end
end

function main()

    benchmarks = NPBenchManager.get_parameters("jacobi_2d")

    tsteps, n = 10, 30
    A_cpu, B_cpu = init_arrays(n)   # it doesn't write through to these
    A, B = reset(A_cpu, B_cpu)
    @dphpc_time((A,B)=reset(A_cpu, B_cpu), run_kernel_j2d(tsteps, n, A, B)) # warmup

    for (preset, sizes) in benchmarks
        tsteps, n = collect(values(sizes))
        A_cpu, B_cpu = init_arrays(n)   # it doesn't write through to these
        A, B = reset(A_cpu, B_cpu)
        @dphpc_time((A,B)=reset(A_cpu, B_cpu), run_kernel_j2d(tsteps, n, A, B), preset)
        # A_res = CUDA.copyto!(Array{Float64}(undef, n, n), A)
        # println("matrix A out:")
        # print_array(A_res)
    end

    RESULTS #<3
end 
