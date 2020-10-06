# A module that collects all necessary helper functions
module JuTools
export compute_accuracy, scale_data, scale_data!, shuffle_data, shuffle_data!, split_data
export data_generate_linear_2d

import Statistics
import Random

"""
Compute accuracy given prediction and real values
"""
function compute_accuracy(y_pred::Array, y_real::Array)::AbstractFloat
    @assert ndims(y_pred) == ndims(y_real) == 1
    @assert size(y_pred) == size(y_real)
    acc = Statistics.mean(y_pred .== y_real)
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
    shuffle_data!(X_data, Y_data)
    return (X_data, Y_data)
end

"""
Generate 2 dimensional data for testing\\
`pos1` and `pos2` define the cluster center of 2 targets\\
`radius1` and `radius2` define the cluster radius\\
`random_scale` defines the scale of randomness\\
Returns a Tuple for X and y, where 2 targets are balanced
"""
function data_generate_cluster_2d(;pos1::Tuple{AbstractFloat, AbstractFloat}=(10.0, 10.0), pos2::Tuple{AbstractFloat, AbstractFloat}=(30.0, 30.0),
        radius1::AbstractFloat=5.0, radius2::AbstractFloat=5.0, random_scale::AbstractFloat=10.0, data_size::Integer=1000)::Tuple{Array, Array}
    @assert radius1 > 0
    @assert radius2 > 0
    @assert data_size >= 2
    data_size_1 = div(data_size, 2)
    data_size_2 = data_size - data_size_1
    step_scale = min(radius1 / 10.0, radius2 / 10.0)
    step_scale = 10.0 ^ (trunc(Int32, step_scale) - 2)
    X_data_1 = Random.rand(-radius1:step_scale:radius1, (data_size_1, 2))
    X_data_2 = Random.rand(-radius2:step_scale:radius2, (data_size_2, 2))
    X_data_1[:, 1] .= X_data_1[:, 1] .+ pos1[1]
    X_data_1[:, 2] .= X_data_1[:, 2] .+ pos1[2]
    X_data_2[:, 1] .= X_data_2[:, 1] .+ pos2[1]
    X_data_2[:, 2] .= X_data_2[:, 2] .+ pos2[2]
    X_data_1 .= X_data_1 .+ (Random.randn(size(X_data_1)) .* random_scale)
    X_data_2 .= X_data_2 .+ (Random.randn(size(X_data_2)) .* random_scale)
    Y_data_1 = ones(data_size_1)
    Y_data_2 = zeros(data_size_2)
    X_data = cat(X_data_1, X_data_2, dims=1)
    Y_data = cat(Y_data_1, Y_data_2, dims=1)
    shuffle_data!(X_data, Y_data)
    return (X_data, Y_data)
end

end