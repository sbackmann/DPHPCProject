
include("_example.jl")

function example(A, B, n, out)
    block_size1 = 16
    block_size2 = 2
    nr_blocks1 = (n-1) ÷ 16 + 1
    nr_blocks2 = block_size1 / block_size2 |> Int
    
    for i1 in 1:nr_blocks1
        for j1 in 1:nr_blocks1
            for k1 in 1:nr_blocks1  
                
                for i2 in 1:nr_blocks2
                    i_out = (i1-1)*block_size1 + (i2-1)*block_size2
                    if i_out >= n break end
                    for j2 in 1:nr_blocks2
                        j_out = (j1-1)*block_size1 + (j2-1)*block_size2
                        if j_out >= n break end
                        for k2 in 1:nr_blocks2
                            k_out = (k1-1)*block_size1 + (k2-1)*block_size2
                            if k_out >= n break end
                            
                            for i3 in 1:block_size2
                                i = i_out + i3
                                if i > n break end
                                for j3 in 1:block_size2
                                    j = j_out + j3
                                    if j > n break end
                                    for k3 in 1:block_size2
                                        k = k_out + k3
                                        if k > n break end
                                        
                                        out[j, i] = out[j, i] + A[j, k] * B[k, i]
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end
end

main()




