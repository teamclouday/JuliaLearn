# A module that collects all necessary helper functions
module JuTools
export compute_accuracy, scale_data, scale_data!, shuffle_data, shuffle_data!, split_data
export data_generate_linear_2d

import Statistics
import Random

"""
Compute accuracy given prediction and real values
"""
function compute_accuracy(y_pred::Array{T} where T<:Number, y_real::Array{T} where T<:Number)::AbstractFloat
    @assert ndims(y_pred) == ndims(y_real) == 1
    @assert size(y_pred) == size(y_real)
    acc = Statistics.mean(float.(y_pred) .== float.(y_real))
    return acc
end

"""
Scale a 2d data, to unit variance
"""
function scale_data(X::Array{T} where T<:Number)::Array
    @assert ndims(X) == 2
    u = Statistics.mean(X, dims=1)
    s = Statistics.std(X, dims=1)
    res = (X .- u) ./ s
    return res
end

"""
Scale a 2d data in place, to unit variance
"""
function scale_data!(X::Array{T} where T<:Number)
    @assert ndims(X) == 2
    u = Statistics.mean(X, dims=1)
    s = Statistics.std(X, dims=1)
    X .= (X .- u) ./ s
    return nothing
end

"""
Shuffle X and y data\\
`X` is 2 dim, `y` is 1 dim\\
Returns shuffled `X` and `y` in Tuple
"""
function shuffle_data(X::Array{T} where T<:Number, y::Array{T} where T<:Number)::Tuple{Array,Array}
    @assert ndims(X) == 2
    @assert ndims(y) == 1
    @assert size(X)[1] == size(y)[1]
    data = hcat(X, y)
    data .= data[Random.randperm(size(data)[1]), :]
    X_shuffled = data[:, 1:end-1]
    y_shuffled = data[:, end]
    return (X_shuffled, y_shuffled)
end

"""
Shuffle X and y data in place\\
`X` is 2 dim, `y` is 1 dim\\
Returns shuffled `X` and `y` in Tuple
"""
function shuffle_data!(X::Array{T} where T<:Number, y::Array{T} where T<:Number)
    @assert ndims(X) == 2
    @assert ndims(y) == 1
    @assert size(X)[1] == size(y)[1]
    data = hcat(X, y)
    data .= data[Random.randperm(size(data)[1]), :]
    X .= data[:, 1:end-1]
    y .= data[:, end]
    return nothing
end

"""
Split X and y data to training set and test set\\
`X` is 2 dim, `y` is 1 dim\\
`ratio` should be in range (0, 1), defines the test data percentage\\
Returns `X_train`, `X_test`, `y_train`, `y_test`
"""
function split_data(X::Array{T} where T<:Number, y::Array{T} where T<:Number;
        shuffle::Bool=true, ratio::AbstractFloat=0.3)::Tuple{Array,Array,Array,Array}
    @assert ndims(X) == 2
    @assert ndims(y) == 1
    @assert size(X)[1] == size(y)[1]
    @assert 0.0 < ratio < 1.0
    if shuffle
        X, y = shuffle_data(X, y)
    end
    m = size(X)[1]
    n = floor(Int, float(m) * ratio)
    # make sure at least one value in the arrays
    n = min(n, m-1)
    n = max(1, n)
    X_test = X[1:n, :]
    y_test = y[1:n]
    X_train = X[n+1:end, :]
    y_train = y[n+1:end]
    return (X_train, X_test, y_train, y_test)
end




# functions for data generation

"""
Generate 2 dimensional data for testing\\
`linear_func` should define a linear function\\
e.g. x1 -> x1 * 2 + 1\\
Returns a Tuple for X and y
"""
function data_generate_linear_2d(;linear_func::Function=k -> k * 2 + 1, data_size::Integer=1000,
        range_min::AbstractFloat=0.0, range_max::AbstractFloat=100.0, range_step::AbstractFloat=0.1,
        random_scale::AbstractFloat=20.0)::Tuple{Array, Array}
    @assert data_size > 0
    @assert range_max > range_min
    @assert range_step > 0
    @assert range_step < (range_max - range_min)
    X_data = Random.rand(range_min:range_step:range_max, (data_size, 2))
    Y_data = Array{Int64}(undef, data_size)
    for i = 1:size(X_data)[1]
        if X_data[i, 2] > linear_func(X_data[i, 1]) + Random.randn() * random_scale
            Y_data[i] = 0
        else
            Y_data[i] = 1
        end
    end
    return (X_data, Y_data)
end

end