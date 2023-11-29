include("./utils.jl")

ASSERT = true


function floyd_warshall_gpu!(n, graph)

    threads = 16
    threads_per_block = (threads, threads)
    blocks = (ceil(Int, n / threads), ceil(Int, n / threads))

    CUDA.@sync(@cuda threads=threads_per_block blocks=blocks floyd_kernel(graph, n))
    return graph
end


function floyd_kernel(graph, n)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    if i <= n && j <= n
        
        # NOT WORKING
        row_i = view(graph, i, 1:n)
        column_j = view(graph, j, 1:n)
        
        #x = CuArray{eltype(row_i)}(undef, n)
        #y = CuArray{eltype(column_j)}(undef, n)

        #copyto!(x, row_i)
        #copyto!(y, column_j)
        #sum = x .+ y
    end
    return
end

main()