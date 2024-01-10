include("./utils.jl")

ASSERT = true
const BLOCK_SIZE = 32
const MAX_DIST = Int32(1000000)

function floyd_warshall_gpu!(n, graph)

    threads_per_block = (BLOCK_SIZE, BLOCK_SIZE)
    blocks1 = (1, 1)
    blocks2 = (ceil(Int, n / BLOCK_SIZE), 2)
    blocks3 = (ceil(Int, n / BLOCK_SIZE), ceil(Int, n / BLOCK_SIZE))
    totalBlocks = ceil(Int, n / BLOCK_SIZE)

    for blockID in 1:totalBlocks
        
        @cuda threads=threads_per_block blocks=blocks1 floyd_kernel_1(graph, n, blockID)

        @cuda threads=threads_per_block blocks=blocks2 floyd_kernel_2(graph, n, blockID)

        @cuda threads=threads_per_block blocks=blocks3 floyd_kernel_3(graph, n, blockID)
    end
    CUDA.synchronize()
    return graph
end


function floyd_kernel_1(graph, n, blockID)

    graphShared = @cuStaticSharedMem(Int32, (BLOCK_SIZE, BLOCK_SIZE))

    idy = threadIdx().x
    idx = threadIdx().y
    
    i = BLOCK_SIZE * (blockID - 1) + idy
    j = BLOCK_SIZE * (blockID - 1) + idx

    if i <= n && j <= n
        @inbounds graphShared[idy, idx] = graph[i, j]
    else
        @inbounds graphShared[idy, idx] = MAX_DIST
    end
    sync_threads()

    @inbounds for k in 1:BLOCK_SIZE
        tmp = graphShared[idy, k] + graphShared[k, idx]
        if graphShared[idy, idx] > tmp
            graphShared[idy, idx] = tmp
        end
        sync_threads()
    end
    if i <= n && j <= n
        @inbounds graph[i, j] = graphShared[idy, idx]
    end
    return
end

function floyd_kernel_2(graph, n, blockID)
    if blockIdx().x == blockID
        return
    end

    idy = threadIdx().x
    idx = threadIdx().y

    i = BLOCK_SIZE * (blockID - 1) + idy
    j = BLOCK_SIZE * (blockID - 1) + idx

    graphSharedTmp = @cuStaticSharedMem(Int32, (BLOCK_SIZE, BLOCK_SIZE))

    if i <= n && j <= n
        @inbounds graphSharedTmp[idy, idx] = graph[i, j]
    else
        @inbounds graphSharedTmp[idy, idx] = MAX_DIST
    end
    
    if blockIdx().y == 1
        j = BLOCK_SIZE * (blockIdx().x - 1) + idx
    else
        i = BLOCK_SIZE * (blockIdx().x - 1) + idy
    end

    bestRoute = MAX_DIST
    if i <= n && j <= n
        @inbounds bestRoute = graph[i, j]
    end
    graphShared = @cuStaticSharedMem(Int32, (BLOCK_SIZE, BLOCK_SIZE))
    @inbounds graphShared[idy, idx] = bestRoute
    sync_threads()
    
    if blockIdx().y == 1
        @inbounds for k in 1:BLOCK_SIZE
            tmp = graphSharedTmp[idy, k] + graphShared[k, idx]
            if bestRoute > tmp
                bestRoute = tmp
            end

            sync_threads()
            graphShared[idy, idx] = bestRoute
            sync_threads()
        end
    else
        @inbounds for k in 1:BLOCK_SIZE
            tmp = graphShared[idy, k] + graphSharedTmp[k, idx]
            if bestRoute > tmp
                bestRoute = tmp
            end

            sync_threads()
            graphShared[idy, idx] = bestRoute
            sync_threads()
        end
    end
    if i <= n && j <= n
        @inbounds graph[i, j] = bestRoute
    end
    return
end

function floyd_kernel_3(graph, n, blockID)
    if blockIdx().x == blockID || blockIdx().y == blockID
        return
    end

    graphSharedTmpRow = @cuStaticSharedMem(Int32, (BLOCK_SIZE, BLOCK_SIZE))
    graphSharedTmpCol = @cuStaticSharedMem(Int32, (BLOCK_SIZE, BLOCK_SIZE))

    idy = threadIdx().x
    idx = threadIdx().y

    i = blockDim().y * (blockIdx().y - 1) + idy
    j = blockDim().x * (blockIdx().x - 1) + idx

    iRow = BLOCK_SIZE * (blockID - 1) + idy
    jCol = BLOCK_SIZE * (blockID - 1) + idx

    if i <= n && jCol <= n
        @inbounds graphSharedTmpCol[idy, idx] = graph[i, jCol]
    else
        @inbounds graphSharedTmpCol[idy, idx] = MAX_DIST
    end

    if iRow <= n && j <= n
        @inbounds graphSharedTmpRow[idy, idx] = graph[iRow, j]
    else
        @inbounds graphSharedTmpRow[idy, idx] = MAX_DIST
    end
    
    sync_threads()

    if i <= n && j <= n
        @inbounds bestRoute = graph[i, j]
        
        @inbounds for k in 1:BLOCK_SIZE
            tmp = graphSharedTmpCol[idy, k] + graphSharedTmpRow[k, idx]
            if bestRoute > tmp
                bestRoute = tmp
            end
        end
        @inbounds graph[i, j] = bestRoute
        
    end
    return
end

main()