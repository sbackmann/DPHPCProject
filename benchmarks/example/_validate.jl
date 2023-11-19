
# using LinearAlgebra
using Test
using CUDA

@testset begin
        
    n = 100
    A = rand(-10:10, n, n)
    B = rand(-10:10, n, n)
    out = zeros(Int, n, n)

    true_result = A * B

    display(true_result)



    include("naive.jl")
    out = zeros(n, n)
    example(A, B, n, out)
    println("testing naive.jl")
    @test out == true_result
    # display(out)



    include("blocked.jl")
    out = zeros(n, n)
    example(A, B, n, out)
    println("testing blocked.jl")
    @test out == true_result

    # display(out)


    function validate_gpu_version(n)
        A = rand(-10:10, n, n)
        B = rand(-10:10, n, n)
        out = zeros(Int, n, n)
    
        true_result = A * B

        dA, dB, dout = CuArray.((A, B, out))
        
        example(dA, dB, n, dout)
        copyto!(out, dout)
        # display(true_result)
        # display(Matrix(dout))
        
        @test out == true_result
    end

    println("testing cuda_oneline.jl")
    include("cuda_oneline.jl")
    validate_gpu_version(n)
    
    println("testing cuda.jl")
    include("cuda.jl")
    validate_gpu_version(n)

end


