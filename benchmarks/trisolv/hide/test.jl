using CUDA

function double_array_kernel(arr)
    t = threadIdx().x
    b = blockIdx().x
    g = gridDim().x

    @cuprintln("block: $(b) threadIdx: $(t), stride: $(stride), gridDim: $(g)")

    elems_per_block = ceil(Int64, length(arr) / gridDim().x)
    elems_per_thread = ceil(Int64, elems_per_block / blockDim().x)
    start_idx = elems_per_block * (blockIdx().x-1) + elems_per_thread * threadIdx().x - 1
    # @cuprintln("elems_per_block: $(elems_per_block), elems_per_thread: $(elems_per_thread), start_idx: $(start_idx)")

    for i = start_idx:min(start_idx + elems_per_thread - 1, length(arr))
        arr[i] = blockIdx().x * 100 + t 
        # arr[i] *= 2
    end

    return
end

function main()
    N = 10
    host_array = collect(1:N)

    # Copy the array to the GPU
    device_array = CUDA.fill(0, N)
    CUDA.copyto!(device_array, host_array)

    # Launch the CUDA kernel
    t = 3
    # @cuda threads=t blocks=div(N + t - 1, t) double_array_kernel(device_array)
    @cuda threads=t blocks=2 double_array_kernel(device_array)

    # Copy the modified array back to the host
    CUDA.copyto!(host_array, device_array)

    println("Original array: ", collect(1:N))
    println("Doubled array: ", host_array)
end

main()
