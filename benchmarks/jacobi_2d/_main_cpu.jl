
function main()
    benchmarks = NPBenchManager.get_parameters("jacobi_2d")

    for (preset, sizes) in benchmarks
        tsteps, n = collect(values(sizes))
        @dphpc_time((A,B)=init_arrays(n), kernel_j2d(tsteps, n, A, B), preset)
    end

    RESULTS #<3
end