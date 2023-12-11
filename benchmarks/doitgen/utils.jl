include("../../timing/dphpc_timing.jl")
using Serialization
using CUDA


function init_array(nr, nq, np)
    A = zeros(Float64, nr, nq, np)
    C4 = zeros(Float64, np, np)
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
    return A, C4
end


function reset(A, C4, nr, nq, np)
    A_gpu = CuArray(A)
    C4_gpu = CuArray(C4)

    # cannot be single dim array bc threads manipulate at the same time (think about optimmization)
    sum = CUDA.fill(0.0, nr, nq, np)
    CUDA.synchronize()
    return A_gpu, C4_gpu, sum
end


function doitgen_gpu_assert!(nr, nq, np, A, C4, sum)
    threads = 16
    threads_per_block = (threads, threads)
    n = max(nr, nq)
    blocks = (ceil(Int, n / threads), ceil(Int, n / threads))

    CUDA.@sync(@cuda threads=threads_per_block blocks=blocks doitgen_kernel_assert(nr, nq, np, A, C4, sum))    

end


function doitgen_kernel_assert(nr, nq, np, A, C4, sum)
    p, s = 1:np, 1:np
    r = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    q = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if r <= nr && q <= nq
        for pp in p
            sum[r, q, pp] = 0.0
            for ss in s
                sum[r, q, pp] += A[r, q, ss] * C4[ss, pp]
            end
        end

        for pp in p
            A[r, q, pp] = sum[r, q, pp]
        end
    end

    return
end


function assert_naive(res, nr, nq, np)
    A, C4 = init_array(nr, nq, np)
    A_gpu, C4_gpu, sum = reset(A, C4, nr, nq, np)
    doitgen_gpu_assert!(nr, nq, np, A_gpu, C4_gpu, sum)
    A_cpu = CUDA.copyto!(Array{Float64}(undef, nr, nq, np), A_gpu)
    #println(A_cpu[1:5, 1:5, 1:5])
    #println(res[1:5, 1:5, 1:5])
    @assert isequal(res, A_cpu)
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
    #println(A_test[1:5, 1:5, 1:5])
    #println(A[1:5, 1:5, 1:5])
    @assert isequal(A, A_test)
end

function main()
    
    nr = 60
    nq = 60
    np = 128
    A, C4 = init_array(nr, nq, np)
    A_gpu, C4_gpu, sum = reset(A, C4, nr, nq, np)
    res = @dphpc_time((A_gpu, C4_gpu, sum) = reset(A, C4, nr, nq, np), doitgen_gpu!(nr, nq, np, A_gpu, C4_gpu, sum), "S")
    if ASSERT && res != nothing
        result_A = CUDA.copyto!(Array{Float64}(undef, nr, nq, np), A_gpu)
        assert_naive(result_A, nr, nq, np)
    end
    

    nr = 110
    nq = 125
    np = 256
    A, C4 = init_array(nr, nq, np)
    A_gpu, C4_gpu, sum = reset(A, C4, nr, nq, np)
    res = @dphpc_time((A_gpu, C4_gpu, sum) = reset(A, C4, nr, nq, np), doitgen_gpu!(nr, nq, np, A_gpu, C4_gpu, sum), "M")
    if ASSERT && res != nothing
        result_A = CUDA.copyto!(Array{Float64}(undef, nr, nq, np), A_gpu)
        assert_naive(result_A, nr, nq, np)
    end


    nr = 220
    nq = 250
    np = 512
    A, C4 = init_array(nr, nq, np)
    A_gpu, C4_gpu, sum = reset(A, C4, nr, nq, np)
    res = @dphpc_time((A_gpu, C4_gpu, sum) = reset(A, C4, nr, nq, np), doitgen_gpu!(nr, nq, np, A_gpu, C4_gpu, sum), "L")
    if ASSERT && res != nothing
        result_A = CUDA.copyto!(Array{Float64}(undef, nr, nq, np), A_gpu)
        assert_naive(result_A, nr, nq, np)
    end


    nr = 220
    nq = 250
    np = 270
    A, C4 = init_array(nr, nq, np)
    A_gpu, C4_gpu, sum = reset(A, C4, nr, nq, np)
    res = @dphpc_time((A_gpu, C4_gpu, sum) = reset(A, C4, nr, nq, np), doitgen_gpu!(nr, nq, np, A_gpu, C4_gpu, sum), "paper")
    if ASSERT && res != nothing
        result_A = CUDA.copyto!(Array{Float64}(undef, nr, nq, np), A_gpu)
        assert_naive(result_A, nr, nq, np)
    end
    
end