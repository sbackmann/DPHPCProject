include("../../timing/dphpc_timing.jl")
using Serialization
using CUDA

ASSERT = true


function init_array(nr, nq, np, A=nothing, C4=nothing)
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
    A_gpu = CUDA.fill(-1.0, nr, nq, np)
    C4_gpu = CUDA.fill(-1.0, np, np)
    CUDA.copyto!(A_gpu, A)
    CUDA.copyto!(C4_gpu, C4)
    # cannot be single dim array bc threads manipulate at the same time (think about optimmization)
    sum = CUDA.fill(0.0, nr, nq, np)
    CUDA.synchronize()
    return A_gpu, C4_gpu, sum
end


function reset(nr, nq, np)
    init_array(nr, nq, np)
end


function doitgen_gpu!(nr, nq, np, A, C4, sum)
    threads = 16
    threads_per_block = (threads, threads)
    n = max(nr, nq)
    blocks = (ceil(Int, n / threads), ceil(Int, n / threads))

    CUDA.@sync(@cuda threads=threads_per_block blocks=blocks doitgen_kernel(nr, nq, np, A, C4, sum))    

end

function doitgen_kernel(nr, nq, np, A, C4, sum)
    p, s = 1:np, 1:np
    r = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    q = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if r <= nr && q <= nq
        for pp in p
            temp_sum = 0.0
            for ss in s
                temp_sum += A[r, q, ss] * C4[ss, pp]
            end
            sum[r, q, pp] = temp_sum
        end

        for pp in p
            A[r, q, pp] = sum[r, q, pp]
        end
    end

    return
end

function doitgen(nr, nq, np, A, C4, sum)
    for r in 1:nr
        for q in 1:nq
            for p in 1:np
                sum[p] = 0.0
                for s in 1:np
                    sum[p] += A[r, q, s] * C4[s, p]
                end
            end
            for p in 1:np
                A[r, q, p] = sum[p]
            end
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
    #println(A_test[1:5, 1:5, 1:5])
    @assert isequal(A, A_test)
end

function main()

    nr = 60
    nq = 60
    np = 128
    A, C4, sum = init_array(nr, nq, np)
    doitgen_gpu!(nr, nq, np, A, C4, sum)
    result_A = CUDA.copyto!(Array{Float64}(undef, nr, nq, np), A)
    # println(result_A[1:5, 1:5, 1:5])
    if ASSERT # && res != nothing
        assert_correctness(result_A, "S")
    end
    
    nr = 60
    nq = 60
    np = 128
    A, C4, sum = init_array(nr, nq, np)
    res = @dphpc_time(reset(nr, nq, np), doitgen_gpu!(nr, nq, np, A, C4, sum), "S")
    
    nr = 110
    nq = 125
    np = 256
    A, C4, sum = init_array(nr, nq, np)
    res = @dphpc_time(reset(nr, nq, np), doitgen_gpu!(nr, nq, np, A, C4, sum), "M")

    nr = 220
    nq = 250
    np = 512
    A, C4, sum = init_array(nr, nq, np)
    res = @dphpc_time(reset(nr, nq, np), doitgen_gpu!(nr, nq, np, A, C4, sum), "L")

    nr = 220
    nq = 250
    np = 270
    A, C4, sum = init_array(nr, nq, np)
    res = @dphpc_time(reset(nr, nq, np), doitgen_gpu!(nr, nq, np, A, C4, sum), "paper")

end

main()