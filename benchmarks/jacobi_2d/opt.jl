include("../../timing/dphpc_timing.jl")
using Printf


function init_arrays(n)
    A = zeros(Float64, n, n)
    B = zeros(Float64, n, n)

    A = [((i - 1) * (j + 1) + 2) / n for i in 1:n, j in 1:n]
    B = [((i - 1) * (j + 2) + 3) / n for i in 1:n, j in 1:n]

    return A, B
end


function kernel_j2d(tsteps, n, A, B)
    local r, k
    for t in 1:tsteps
        # COL MAJOR & BETTER LOCALITY
        # for j in 2:3
        #     for i in 2:(n-1)
        #         B[i, j] = 0.2 * (A[i, j] + A[i, j-1] + A[i, j+1] + A[i+1, j] + A[i-1, j])
        #     end
        # end
        # for j in 4:(n-1)
        #     for i in 2:(n-1)
        #         B[i, j] = 0.2 * (A[i, j] + A[i, j-1] + A[i, j+1] + A[i+1, j] + A[i-1, j])
        #         A[i, j-2] = 0.2 * (B[i, j-2] + B[i, j-3] + B[i, j-1] + B[i+1, j-2] + B[i-1, j-2])
        #     end
        # end
        # for j in (n-2):(n-1)
        #     for i in 2:(n-1)
        #         A[i, j] = 0.2 * (B[i, j] + B[i, j-1] + B[i, j+1] + B[i+1, j] + B[i-1, j])
        #     end
        # end
        
        # UNROLL INNER 2X
        # for j=2:3
        #     for i=2:(n-1)
        #         B[i, j] = 0.2 * (A[i, j] + A[i, j-1] + A[i, j+1] + A[i+1, j] + A[i-1, j])
        #     end
        # end
        # for j=4:(n-1)
        #     for outer r=2:2:(n-2)
        #         B[r, j] = 0.2 * (A[r, j] + A[r, j-1] + A[r, j+1] + A[r+1, j] + A[r-1, j])
        #         B[r+1, j] = 0.2 * (A[r+1, j] + A[r+1, j-1] + A[r+1, j+1] + A[r+2, j] + A[r, j])
                
        #         A[r, j-2] = 0.2 * (B[r, j-2] + B[r, j-3] + B[r, j-1] + B[r+1, j-2] + B[r-1, j-2])
        #         A[r+1, j-2] = 0.2 * (B[r+1, j-2] + B[r+1, j-3] + B[r+1, j-1] + B[r+2, j-2] + B[r, j-2])
        #     end
        #     for i=r:(n-1)
        #         B[i, j] = 0.2 * (A[i, j] + A[i, j-1] + A[i, j+1] + A[i+1, j] + A[i-1, j])
        #         A[i, j-2] = 0.2 * (B[i, j-2] + B[i, j-3] + B[i, j-1] + B[i+1, j-2] + B[i-1, j-2])
        #     end
        # end
        # for j=(n-2):(n-1)
        #     for i=2:(n-1)
        #         A[i, j] = 0.2 * (B[i, j] + B[i, j-1] + B[i, j+1] + B[i+1, j] + B[i-1, j])
        #     end
        # end
        
        # UNROLL INNER 2X & SKIP BOUNDARY CHECKS
        # for j=2:3
        #     for i=2:(n-1)
        #         @inbounds B[i, j] = 0.2 * (A[i, j] + A[i, j-1] + A[i, j+1] + A[i+1, j] + A[i-1, j])
        #     end
        # end
        # for j=4:(n-1)
        #     for outer r=2:2:(n-2)
        #         @inbounds B[r, j] = 0.2 * (A[r, j] + A[r, j-1] + A[r, j+1] + A[r+1, j] + A[r-1, j])
        #         @inbounds B[r+1, j] = 0.2 * (A[r+1, j] + A[r+1, j-1] + A[r+1, j+1] + A[r+2, j] + A[r, j])
                
        #         @inbounds A[r, j-2] = 0.2 * (B[r, j-2] + B[r, j-3] + B[r, j-1] + B[r+1, j-2] + B[r-1, j-2])
        #         @inbounds A[r+1, j-2] = 0.2 * (B[r+1, j-2] + B[r+1, j-3] + B[r+1, j-1] + B[r+2, j-2] + B[r, j-2])
        #     end
        #     for i=r:(n-1)
        #         @inbounds B[i, j] = 0.2 * (A[i, j] + A[i, j-1] + A[i, j+1] + A[i+1, j] + A[i-1, j])
        #         @inbounds A[i, j-2] = 0.2 * (B[i, j-2] + B[i, j-3] + B[i, j-1] + B[i+1, j-2] + B[i-1, j-2])
        #     end
        # end
        # for j=(n-2):(n-1)
        #     for i=2:(n-1)
        #         @inbounds A[i, j] = 0.2 * (B[i, j] + B[i, j-1] + B[i, j+1] + B[i+1, j] + B[i-1, j])
        #     end
        # end
        
        # UNROLL INNER 4X
        # for j=2:3
        #     for i=2:(n-1)
        #         @inbounds B[i, j] = 0.2 * (A[i, j] + A[i, j-1] + A[i, j+1] + A[i+1, j] + A[i-1, j])
        #     end
        # end
        # for j=4:(n-1)
        #     for outer r=2:4:(n-4)
        #         @inbounds B[r, j] = 0.2 * (A[r, j] + A[r, j-1] + A[r, j+1] + A[r+1, j] + A[r-1, j])
        #         @inbounds B[r+1, j] = 0.2 * (A[r+1, j] + A[r+1, j-1] + A[r+1, j+1] + A[r+2, j] + A[r, j])
        #         @inbounds B[r+2, j] = 0.2 * (A[r+2, j] + A[r+2, j-1] + A[r+2, j+1] + A[r+3, j] + A[r+1, j])
        #         @inbounds B[r+3, j] = 0.2 * (A[r+3, j] + A[r+3, j-1] + A[r+3, j+1] + A[r+4, j] + A[r+2, j])
                
        #         @inbounds A[r, j-2] = 0.2 * (B[r, j-2] + B[r, j-3] + B[r, j-1] + B[r+1, j-2] + B[r-1, j-2])
        #         @inbounds A[r+1, j-2] = 0.2 * (B[r+1, j-2] + B[r+1, j-3] + B[r+1, j-1] + B[r+2, j-2] + B[r, j-2])
        #         @inbounds A[r+2, j-2] = 0.2 * (B[r+2, j-2] + B[r+2, j-3] + B[r+2, j-1] + B[r+3, j-2] + B[r+1, j-2])
        #         @inbounds A[r+3, j-2] = 0.2 * (B[r+3, j-2] + B[r+3, j-3] + B[r+3, j-1] + B[r+4, j-2] + B[r+2, j-2])
        #     end
        #     for i=r:(n-1)
        #         @inbounds B[i, j] = 0.2 * (A[i, j] + A[i, j-1] + A[i, j+1] + A[i+1, j] + A[i-1, j])
        #         @inbounds A[i, j-2] = 0.2 * (B[i, j-2] + B[i, j-3] + B[i, j-1] + B[i+1, j-2] + B[i-1, j-2])
        #     end
        # end
        # for j=(n-2):(n-1)
        #     for i=2:(n-1)
        #         @inbounds A[i, j] = 0.2 * (B[i, j] + B[i, j-1] + B[i, j+1] + B[i+1, j] + B[i-1, j])
        #     end
        # end
        
        # CORRECT STRIDING
        # for j=2:3
        #     for i=2:(n-1)
        #         @inbounds B[i, j] = 0.2 * (A[i, j] + A[i, j-1] + A[i, j+1] + A[i+1, j] + A[i-1, j])
        #     end
        # end
        # for outer k=4:2:(n-2)
        #     for outer r=2:2:(n-2)
        #         @inbounds B[r, k] = 0.2 * (A[r, k] + A[r, k-1] + A[r, k+1] + A[r+1, k] + A[r-1, k])     # 1
        #         @inbounds B[r+1, k] = 0.2 * (A[r+1, k] + A[r+1, k-1] + A[r+1, k+1] + A[r+2, k] + A[r, k])   # 2
                
        #         @inbounds A[r, k-2] = 0.2 * (B[r, k-2] + B[r, k-3] + B[r, k-1] + B[r+1, k-2] + B[r-1, k-2])
        #         @inbounds A[r+1, k-2] = 0.2 * (B[r+1, k-2] + B[r+1, k-3] + B[r+1, k-1] + B[r+2, k-2] + B[r, k-2])
        #     end
        #     for i=(r+2):(n-1)
        #         @inbounds B[i, k] = 0.2 * (A[i, k] + A[i, k-1] + A[i, k+1] + A[i+1, k] + A[i-1, k]) # 1'
                
        #         @inbounds A[i, k-2] = 0.2 * (B[i, k-2] + B[i, k-3] + B[i, k-1] + B[i+1, k-2] + B[i-1, k-2])
        #     end
            
        #     for outer r=2:2:(n-2)
        #         @inbounds B[r, k+1] = 0.2 * (A[r, k+1] + A[r, k] + A[r, k+2] + A[r+1, k+1] + A[r-1, k+1])   # far away: next col
        #         @inbounds B[r+1, k+1] = 0.2 * (A[r+1, k+1] + A[r+1, k] + A[r+1, k+2] + A[r+2, k+1] + A[r, k+1])
                
        #         @inbounds A[r, k-1] = 0.2 * (B[r, k-1] + B[r, k-2] + B[r, k] + B[r+1, k-1] + B[r-1, k-1])   # dep on 1
        #         @inbounds A[r+1, k-1] = 0.2 * (B[r+1, k-1] + B[r+1, k-2] + B[r+1, k] + B[r+2, k-1] + B[r, k-1])     # dep on 2
        #     end
        #     for i=(r+2):(n-1)
        #         @inbounds B[i, k+1] = 0.2 * (A[i, k+1] + A[i, k] + A[i, k+2] + A[i+1, k+1] + A[i-1, k+1])
        #         @inbounds A[i, k-1] = 0.2 * (B[i, k-1] + B[i, k-2] + B[i, k] + B[i+1, k-1] + B[i-1, k-1])   # dep on 1'
        #     end
        # end
        # for j=(k+2):(n-1)
        #     for i=2:(n-1)
        #         @inbounds B[i, j] = 0.2 * (A[i, j] + A[i, j-1] + A[i, j+1] + A[i+1, j] + A[i-1, j])
        #         @inbounds A[i, j-2] = 0.2 * (B[i, j-2] + B[i, j-3] + B[i, j-1] + B[i+1, j-2] + B[i-1, j-2])
        #     end
        # end
        # for j=(n-2):(n-1)
        #     for i=2:(n-1)
        #         @inbounds A[i, j] = 0.2 * (B[i, j] + B[i, j-1] + B[i, j+1] + B[i+1, j] + B[i-1, j])
        #     end
        # end
        
        # UNROLL OUTER 2X (fused loops), INNER 2X
        # for j=2:3
        #     for i=2:(n-1)
        #         @inbounds B[i, j] = 0.2 * (A[i, j] + A[i, j-1] + A[i, j+1] + A[i+1, j] + A[i-1, j])
        #     end
        # end
        # for outer k=4:2:(n-2)
        #     for outer r=2:2:(n-2)
        #         @inbounds B[r, k] = 0.2 * (A[r, k] + A[r, k-1] + A[r, k+1] + A[r+1, k] + A[r-1, k])     # 1
        #         @inbounds B[r+1, k] = 0.2 * (A[r+1, k] + A[r+1, k-1] + A[r+1, k+1] + A[r+2, k] + A[r, k])   # 2
        #         @inbounds B[r, k+1] = 0.2 * (A[r, k+1] + A[r, k] + A[r, k+2] + A[r+1, k+1] + A[r-1, k+1])   # far away: next col
        #         @inbounds B[r+1, k+1] = 0.2 * (A[r+1, k+1] + A[r+1, k] + A[r+1, k+2] + A[r+2, k+1] + A[r, k+1])
                
        #         @inbounds A[r, k-2] = 0.2 * (B[r, k-2] + B[r, k-3] + B[r, k-1] + B[r+1, k-2] + B[r-1, k-2])
        #         @inbounds A[r+1, k-2] = 0.2 * (B[r+1, k-2] + B[r+1, k-3] + B[r+1, k-1] + B[r+2, k-2] + B[r, k-2])
        #         @inbounds A[r, k-1] = 0.2 * (B[r, k-1] + B[r, k-2] + B[r, k] + B[r+1, k-1] + B[r-1, k-1])   # dep on 1
        #         @inbounds A[r+1, k-1] = 0.2 * (B[r+1, k-1] + B[r+1, k-2] + B[r+1, k] + B[r+2, k-1] + B[r, k-1])     # dep on 2
        #     end
        #     for i=(r+2):(n-1)
        #         @inbounds B[i, k] = 0.2 * (A[i, k] + A[i, k-1] + A[i, k+1] + A[i+1, k] + A[i-1, k]) # 1'
        #         @inbounds B[i, k+1] = 0.2 * (A[i, k+1] + A[i, k] + A[i, k+2] + A[i+1, k+1] + A[i-1, k+1])
                
        #         @inbounds A[i, k-2] = 0.2 * (B[i, k-2] + B[i, k-3] + B[i, k-1] + B[i+1, k-2] + B[i-1, k-2])
        #         @inbounds A[i, k-1] = 0.2 * (B[i, k-1] + B[i, k-2] + B[i, k] + B[i+1, k-1] + B[i-1, k-1])   # dep on 1'
        #     end
        # end
        # for j=(k+2):(n-1)
        #     for i=2:(n-1)
        #         @inbounds B[i, j] = 0.2 * (A[i, j] + A[i, j-1] + A[i, j+1] + A[i+1, j] + A[i-1, j])
        #         @inbounds A[i, j-2] = 0.2 * (B[i, j-2] + B[i, j-3] + B[i, j-1] + B[i+1, j-2] + B[i-1, j-2])
        #     end
        # end
        # for j=(n-2):(n-1)
        #     for i=2:(n-1)
        #         @inbounds A[i, j] = 0.2 * (B[i, j] + B[i, j-1] + B[i, j+1] + B[i+1, j] + B[i-1, j])
        #     end
        # end
        
        # UNROLL OUTER 2X (fused loops), INNER 4X
        # for j=2:3
        #     for i=2:(n-1)
        #         @inbounds B[i, j] = 0.2 * (A[i, j] + A[i, j-1] + A[i, j+1] + A[i+1, j] + A[i-1, j])
        #     end
        # end
        # for outer k=4:2:(n-2)
        #     for outer r=2:4:(n-4)
        #         @inbounds B[r, k] = 0.2 * (A[r, k] + A[r, k-1] + A[r, k+1] + A[r+1, k] + A[r-1, k])     # 1
        #         @inbounds B[r+1, k] = 0.2 * (A[r+1, k] + A[r+1, k-1] + A[r+1, k+1] + A[r+2, k] + A[r, k])   # 2
        #         @inbounds B[r+2, k] = 0.2 * (A[r+2, k] + A[r+2, k-1] + A[r+2, k+1] + A[r+3, k] + A[r+1, k]) # 3
        #         @inbounds B[r+3, k] = 0.2 * (A[r+3, k] + A[r+3, k-1] + A[r+3, k+1] + A[r+4, k] + A[r+2, k]) # 4
        #         @inbounds B[r, k+1] = 0.2 * (A[r, k+1] + A[r, k] + A[r, k+2] + A[r+1, k+1] + A[r-1, k+1])   # far away: next col
        #         @inbounds B[r+1, k+1] = 0.2 * (A[r+1, k+1] + A[r+1, k] + A[r+1, k+2] + A[r+2, k+1] + A[r, k+1])
        #         @inbounds B[r+2, k+1] = 0.2 * (A[r+2, k+1] + A[r+2, k] + A[r+2, k+2] + A[r+3, k+1] + A[r+1, k+1])
        #         @inbounds B[r+3, k+1] = 0.2 * (A[r+3, k+1] + A[r+3, k] + A[r+3, k+2] + A[r+4, k+1] + A[r+2, k+1])
                
        #         @inbounds A[r, k-2] = 0.2 * (B[r, k-2] + B[r, k-3] + B[r, k-1] + B[r+1, k-2] + B[r-1, k-2])
        #         @inbounds A[r+1, k-2] = 0.2 * (B[r+1, k-2] + B[r+1, k-3] + B[r+1, k-1] + B[r+2, k-2] + B[r, k-2])
        #         @inbounds A[r+2, k-2] = 0.2 * (B[r+2, k-2] + B[r+2, k-3] + B[r+2, k-1] + B[r+3, k-2] + B[r+1, k-2])
        #         @inbounds A[r+3, k-2] = 0.2 * (B[r+3, k-2] + B[r+3, k-3] + B[r+3, k-1] + B[r+4, k-2] + B[r+2, k-2])
        #         @inbounds A[r, k-1] = 0.2 * (B[r, k-1] + B[r, k-2] + B[r, k] + B[r+1, k-1] + B[r-1, k-1])   # dep on 1
        #         @inbounds A[r+1, k-1] = 0.2 * (B[r+1, k-1] + B[r+1, k-2] + B[r+1, k] + B[r+2, k-1] + B[r, k-1])     # dep on 2
        #         @inbounds A[r+2, k-1] = 0.2 * (B[r+2, k-1] + B[r+2, k-2] + B[r+2, k] + B[r+3, k-1] + B[r+1, k-1])   # dep on 3
        #         @inbounds A[r+3, k-1] = 0.2 * (B[r+3, k-1] + B[r+3, k-2] + B[r+3, k] + B[r+4, k-1] + B[r+2, k-1])   # dep on 4
        #     end
        #     for i=(r+4):(n-1)
        #         @inbounds B[i, k] = 0.2 * (A[i, k] + A[i, k-1] + A[i, k+1] + A[i+1, k] + A[i-1, k]) # 1'
        #         @inbounds B[i, k+1] = 0.2 * (A[i, k+1] + A[i, k] + A[i, k+2] + A[i+1, k+1] + A[i-1, k+1])
                
        #         @inbounds A[i, k-2] = 0.2 * (B[i, k-2] + B[i, k-3] + B[i, k-1] + B[i+1, k-2] + B[i-1, k-2])
        #         @inbounds A[i, k-1] = 0.2 * (B[i, k-1] + B[i, k-2] + B[i, k] + B[i+1, k-1] + B[i-1, k-1])   # dep on 1'
        #     end
        # end
        # for j=(k+2):(n-1)
        #     for i=2:(n-1)
        #         @inbounds B[i, j] = 0.2 * (A[i, j] + A[i, j-1] + A[i, j+1] + A[i+1, j] + A[i-1, j])
        #         @inbounds A[i, j-2] = 0.2 * (B[i, j-2] + B[i, j-3] + B[i, j-1] + B[i+1, j-2] + B[i-1, j-2])
        #     end
        # end
        # for j=(n-2):(n-1)
        #     for i=2:(n-1)
        #         @inbounds A[i, j] = 0.2 * (B[i, j] + B[i, j-1] + B[i, j+1] + B[i+1, j] + B[i-1, j])
        #     end
        # end
        
        # UNROLL OUTER 2X, INNER 8X
        for j=2:3
            for i=2:(n-1)
                @inbounds B[i, j] = 0.2 * (A[i, j] + A[i, j-1] + A[i, j+1] + A[i+1, j] + A[i-1, j])
            end
        end
        for outer k=4:2:(n-2)
            for outer r=2:8:(n-8)
                @inbounds B[r, k] = 0.2 * (A[r, k] + A[r, k-1] + A[r, k+1] + A[r+1, k] + A[r-1, k])     # 1
                @inbounds B[r+1, k] = 0.2 * (A[r+1, k] + A[r+1, k-1] + A[r+1, k+1] + A[r+2, k] + A[r, k])   # 2
                @inbounds B[r+2, k] = 0.2 * (A[r+2, k] + A[r+2, k-1] + A[r+2, k+1] + A[r+3, k] + A[r+1, k]) # 3
                @inbounds B[r+3, k] = 0.2 * (A[r+3, k] + A[r+3, k-1] + A[r+3, k+1] + A[r+4, k] + A[r+2, k]) # 4
                @inbounds B[r+4, k] = 0.2 * (A[r+4, k] + A[r+4, k-1] + A[r+4, k+1] + A[r+5, k] + A[r+3, k]) # 5
                @inbounds B[r+5, k] = 0.2 * (A[r+5, k] + A[r+5, k-1] + A[r+5, k+1] + A[r+6, k] + A[r+4, k]) # 6
                @inbounds B[r+6, k] = 0.2 * (A[r+6, k] + A[r+6, k-1] + A[r+6, k+1] + A[r+7, k] + A[r+5, k]) # 7
                @inbounds B[r+7, k] = 0.2 * (A[r+7, k] + A[r+7, k-1] + A[r+7, k+1] + A[r+8, k] + A[r+6, k]) # 8
                @inbounds B[r, k+1] = 0.2 * (A[r, k+1] + A[r, k] + A[r, k+2] + A[r+1, k+1] + A[r-1, k+1])   # far away: next col
                @inbounds B[r+1, k+1] = 0.2 * (A[r+1, k+1] + A[r+1, k] + A[r+1, k+2] + A[r+2, k+1] + A[r, k+1])
                @inbounds B[r+2, k+1] = 0.2 * (A[r+2, k+1] + A[r+2, k] + A[r+2, k+2] + A[r+3, k+1] + A[r+1, k+1])
                @inbounds B[r+3, k+1] = 0.2 * (A[r+3, k+1] + A[r+3, k] + A[r+3, k+2] + A[r+4, k+1] + A[r+2, k+1])
                @inbounds B[r+4, k+1] = 0.2 * (A[r+4, k+1] + A[r+4, k] + A[r+4, k+2] + A[r+5, k+1] + A[r+3, k+1])
                @inbounds B[r+5, k+1] = 0.2 * (A[r+5, k+1] + A[r+5, k] + A[r+5, k+2] + A[r+6, k+1] + A[r+4, k+1])
                @inbounds B[r+6, k+1] = 0.2 * (A[r+6, k+1] + A[r+6, k] + A[r+6, k+2] + A[r+7, k+1] + A[r+5, k+1])
                @inbounds B[r+7, k+1] = 0.2 * (A[r+7, k+1] + A[r+7, k] + A[r+7, k+2] + A[r+8, k+1] + A[r+6, k+1])
                
                @inbounds A[r, k-2] = 0.2 * (B[r, k-2] + B[r, k-3] + B[r, k-1] + B[r+1, k-2] + B[r-1, k-2])
                @inbounds A[r+1, k-2] = 0.2 * (B[r+1, k-2] + B[r+1, k-3] + B[r+1, k-1] + B[r+2, k-2] + B[r, k-2])
                @inbounds A[r+2, k-2] = 0.2 * (B[r+2, k-2] + B[r+2, k-3] + B[r+2, k-1] + B[r+3, k-2] + B[r+1, k-2])
                @inbounds A[r+3, k-2] = 0.2 * (B[r+3, k-2] + B[r+3, k-3] + B[r+3, k-1] + B[r+4, k-2] + B[r+2, k-2])
                @inbounds A[r+4, k-2] = 0.2 * (B[r+4, k-2] + B[r+4, k-3] + B[r+4, k-1] + B[r+5, k-2] + B[r+3, k-2])
                @inbounds A[r+5, k-2] = 0.2 * (B[r+5, k-2] + B[r+5, k-3] + B[r+5, k-1] + B[r+6, k-2] + B[r+4, k-2])
                @inbounds A[r+6, k-2] = 0.2 * (B[r+6, k-2] + B[r+6, k-3] + B[r+6, k-1] + B[r+7, k-2] + B[r+5, k-2])
                @inbounds A[r+7, k-2] = 0.2 * (B[r+7, k-2] + B[r+7, k-3] + B[r+7, k-1] + B[r+8, k-2] + B[r+6, k-2])
                @inbounds A[r, k-1] = 0.2 * (B[r, k-1] + B[r, k-2] + B[r, k] + B[r+1, k-1] + B[r-1, k-1])   # dep on 1
                @inbounds A[r+1, k-1] = 0.2 * (B[r+1, k-1] + B[r+1, k-2] + B[r+1, k] + B[r+2, k-1] + B[r, k-1])     # dep on 2
                @inbounds A[r+2, k-1] = 0.2 * (B[r+2, k-1] + B[r+2, k-2] + B[r+2, k] + B[r+3, k-1] + B[r+1, k-1])   # dep on 3
                @inbounds A[r+3, k-1] = 0.2 * (B[r+3, k-1] + B[r+3, k-2] + B[r+3, k] + B[r+4, k-1] + B[r+2, k-1])   # dep on 4
                @inbounds A[r+4, k-1] = 0.2 * (B[r+4, k-1] + B[r+4, k-2] + B[r+4, k] + B[r+5, k-1] + B[r+3, k-1])   # dep on 5
                @inbounds A[r+5, k-1] = 0.2 * (B[r+5, k-1] + B[r+5, k-2] + B[r+5, k] + B[r+6, k-1] + B[r+4, k-1])   # dep on 6
                @inbounds A[r+6, k-1] = 0.2 * (B[r+6, k-1] + B[r+6, k-2] + B[r+6, k] + B[r+7, k-1] + B[r+5, k-1])   # dep on 7
                @inbounds A[r+7, k-1] = 0.2 * (B[r+7, k-1] + B[r+7, k-2] + B[r+7, k] + B[r+8, k-1] + B[r+6, k-1])   # dep on 8
            end
            for i=(r+8):(n-1)
                @inbounds B[i, k] = 0.2 * (A[i, k] + A[i, k-1] + A[i, k+1] + A[i+1, k] + A[i-1, k]) # 1'
                @inbounds B[i, k+1] = 0.2 * (A[i, k+1] + A[i, k] + A[i, k+2] + A[i+1, k+1] + A[i-1, k+1])
                
                @inbounds A[i, k-2] = 0.2 * (B[i, k-2] + B[i, k-3] + B[i, k-1] + B[i+1, k-2] + B[i-1, k-2])
                @inbounds A[i, k-1] = 0.2 * (B[i, k-1] + B[i, k-2] + B[i, k] + B[i+1, k-1] + B[i-1, k-1])   # dep on 1'
            end
        end
        for j=(k+2):(n-1)
            for i=2:(n-1)
                @inbounds B[i, j] = 0.2 * (A[i, j] + A[i, j-1] + A[i, j+1] + A[i+1, j] + A[i-1, j])
                @inbounds A[i, j-2] = 0.2 * (B[i, j-2] + B[i, j-3] + B[i, j-1] + B[i+1, j-2] + B[i-1, j-2])
            end
        end
        for j=(n-2):(n-1)
            for i=2:(n-1)
                @inbounds A[i, j] = 0.2 * (B[i, j] + B[i, j-1] + B[i, j+1] + B[i+1, j] + B[i-1, j])
            end
        end
        
    end
end


function print_array(A)
    for i in 1:size(A, 1)
        for j in 1:size(A, 2)
            @printf("%.2f ", A[i, j])
        end
        println()
    end
end


function main()
    tsteps, n = 50, 150
    A, B = init_arrays(n)
    #println("matrix A in:")
    #print_array(A)
    res = @dphpc_time((A,B)=init_arrays(n), kernel_j2d(tsteps, n, A, B), "S")
    #println("matrix A out:")
    #print_array(A)
    println(res)

    tsteps, n = 80, 350
    A, B = init_arrays(n)
    res = @dphpc_time((A,B)=init_arrays(n), kernel_j2d(tsteps, n, A, B), "M")
    println(res)

    tsteps, n = 200, 700
    A, B = init_arrays(n)
    res = @dphpc_time((A,B)=init_arrays(n), kernel_j2d(tsteps, n, A, B), "L")
    println(res)

    tsteps, n = 1000, 2800
    # tsteps, n = 500, 1400 # in-between for testing
    A, B = init_arrays(n)
    res = @dphpc_time((A,B)=init_arrays(n), kernel_j2d(tsteps, n, A, B), "paper")
    println(res)
end

main()