include("../../timing/dphpc_timing.jl")
using CUDA 


function kernel_col(A, N, i)
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x + i

    if j <= N
        A[j, i] /= A[i, i]
    end

    nothing
end


function kernel_submat(A, N, i)
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x + i
    k = (blockIdx().y - 1) * blockDim().y + threadIdx().y + i

    if j <= N && k <= N
        A[j, k] -= (A[i, k] * A[j, i])
    end

    nothing
end


function run_lu_kernel(N, A)
    threadsPerBlock1D = 256
    threadsPerBlock2D = (16, 16)
    
    for i in 1:N
        blocks1D = div(N - i - 1, threadsPerBlock1D) + 1
        @cuda threads=threadsPerBlock1D blocks=blocks1D kernel_col(A, N, i)
        
        blocks2D = (div(N - i - 1, threadsPerBlock2D[1]) + 1, div(N - i- 1, threadsPerBlock2D[2]) + 1)
        @cuda threads=threadsPerBlock2D blocks=blocks2D kernel_submat(A, N, i)
    end
    
    CUDA.synchronize()
end


include("_main_gpu.jl")

function init_array(N)

    A = zeros(Float64,N, N)

    for i in 1:N
        for j in 1:i
            A[j, i] = ((-j-1) % N) / N + 1.0
        end
        for j in i+1:N
            A[j, i] = 0.0
        end
        A[i, i] = 1.0
    end

    B = zeros(Float64, N, N)
    
    for t in 1:N
        for r in 1:N
            for s in 1:N
                B[s, r] += A[t,r] * A[t,s]
            end
        end
    end

    A .= B

    return transpose(CuArray(A))
end

function is_valid(N, A)
    trueA = init_array(N)
    L = UnitLowerTriangular(CuArray(A))
    R = UpperTriangular(CuArray(A))
    difference = norm(L*R - trueA)
    println(difference)
    if difference < 1e-6
        return true
    else
        # display(A)
        println("VALIDATION FAILED")
        return false
    end

end

main()
