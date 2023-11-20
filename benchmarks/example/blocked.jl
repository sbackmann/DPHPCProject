
include("_example.jl")

function example(A, B, n, out)
    BS = 32 # blocksize
    nb = n ÷ BS # nr_blocks
    l = nb*BS+1
    for cc = 0:nb-1
        for rr = 0:nb-1
            for ii = 0:nb-1
                for c = cc*BS+1:(cc+1)*BS
                    for r = rr*BS+1:(rr+1)*BS
                        for i = ii*BS+1:(ii+1)*BS
                            out[r, c] += A[r, i] * B[i, c]
                        end
                    end
                end
            end
            for c = cc*BS+1:(cc+1)*BS
                for r = rr*BS+1:(rr+1)*BS
                    for i = l:n
                        out[r, c] += A[r, i] * B[i, c]
                    end
                end
            end
        end
    end
    
    for c = 1:l-1
        for r = l:n
            for i = 1:n
                out[r, c] += A[r, i] * B[i, c]
            end
        end
    end
    for c = l:n
        for r = 1:l-1
            for i = 1:n
                out[r, c] += A[r, i] * B[i, c]
            end
        end
    end
    for c = l:n
        for r = l:n
            for i = 1:n
                out[r, c] += A[r, i] * B[i, c]
            end
        end
    end
end

main()