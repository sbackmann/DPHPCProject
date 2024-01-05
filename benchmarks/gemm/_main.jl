
using LinearAlgebra

function reference(N, M, K, A, B, C)
    alpha = 1.5
    beta = 1.2
    result = alpha .* (A * B) .+ beta .* C
    CUDA.synchronize()
    return result
end

function is_valid(result) 

    N, M, K = 30, 40, 50
    (A, B, C) = init_matrices(N,M,K)
    ref = reference(N, M, K, A, B, C)
    #print("C from validation script")
    #println(c_cpu_)

    # println(norm(ref - result))
    # display(ref - result)
    return norm(ref - result) < 1e-8 

end 

function main()


    N, M, K = 30, 40, 50
    (A, B, C) = init_matrices(N,M,K)
    run_gemm_kernel(N,M,K,A,B,C)
    if is_valid(C)

        N, M, K = 100, 110, 120 # warmup
        @dphpc_time((A, B, C) = init_matrices(N,M,K), run_gemm_kernel(N, M, K, A, B, C))

        benchmarks = NPBenchManager.get_parameters("gemm")
        for (preset, sizes) in benchmarks
            N, M, K = collect(values(sizes))
            @dphpc_time((A, B, C) = init_matrices(N,M,K), run_gemm_kernel(N, M, K, A, B, C), preset)
        end

    else
        println("VALIDATION FAILED")
    end

end