
function init(M, N, datatype=Float64; cuda=false)
    data = [datatype((i-1) * (j-1)) / M for i in 1:N, j in 1:M]
    return cuda ? CuArray(data) : data
end

function transpose(data, datat, M, N)
    for i in 1:M
        for j in 1:N
            datat[i, j] = data[j, i]
        end
    end
    return
end

data = init(3, 4)
println(data)

datat = zeros(3, 4)
transpose(data, datat, 3, 4)
println(datat)