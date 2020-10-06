# Naive Bayes Module

module NB
export NaiveBayes, train, predict

"""
Data structure for Naive Bayes\\
`pY` is probability for each target\\
`mean` is the corresponding mean of Xi for each taregt\\
`var` is the corresponding variance of Xi for each target
"""
mutable struct NaiveBayes
    n_features::Integer
    pY::Dict{Number,AbstractFloat}
    mean::Dict{Number,Array{AbstractFloat}}
    var::Dict{Number,Array{AbstractFloat}}
end

"""
Training function for Naive Bayes\\
`X_data` has shape (M,N)\\
`Y_data` has shape (M,)\\
Returns the computed data structure `NaiveBayes`
"""
function train(X_data::Array{T} where T<:Number, Y_data::Array{K} where K<:Number)::NaiveBayes
    @assert ndims(X_data) == ndims(Y_data) + 1
    @assert size(X_data)[1] == size(Y_data)[1]
    n_features = size(X_data)[2]
    # compute probability of Y
    pY = Dict{Number,AbstractFloat}()
    for Y_val in Y_data
        if haskey(pY, Y_val)
            pY[Y_val] += 1.0
        else
            pY[Y_val] = 1.0
        end
    end
    for Y_val in keys(pY)
        pY[Y_val] /= size(Y_data)[1]
    end
    # compute Xi mean and var for each Y
    mean = Dict{Number,Array{AbstractFloat}}()
    var = Dict{Number,Array{AbstractFloat}}()
    for Y_val in keys(pY)
        X_part = X_data[Y_data .== Y_val, :]
        mean[Y_val] = vec(sum(X_part, dims=1) ./ size(X_part)[1])
        var[Y_val] = vec(sum((X_part .- reshape(mean[Y_val], 1, :)).^2, dims=1) ./ size(X_part)[1])
    end
    return NaiveBayes(n_features, pY, mean, var)
end

"""
Vectorized guassian function\\
Returns the product of P(xi|y)
"""
function guassian(X_vec::Array, mean::Array, var::Array)::AbstractFloat
    @assert ndims(X_vec) == ndims(mean) == ndims(var) == 1
    @assert length(X_vec) == length(mean) == length(var)
    left = 1.0 ./ sqrt.((2.0 * pi) .* var)
    right = exp.(-(X_vec .- mean).^2 ./ (2.0 .* var))
    p = left .* right
    return prod(p)
end

"""
Predict function for Naive Bayes\\
`X_data` has ndims of 1 or 2
"""
function predict(X_data::Array{T} where T<:Number, data::NaiveBayes)::Array
    if ndims(X_data) == 1
        X_data = reshape(X_data, 1, :)
    end
    @assert ndims(X_data) == 2
    @assert size(X_data)[2] == data.n_features
    prediction = Array{Number}(undef, size(X_data)[1])
    for i in 1:size(X_data)[1]
        X_vec = X_data[i, :]
        Y_dict = Dict{Number,AbstractFloat}()
        for Y_val in keys(data.pY)
            Y_dict[Y_val] = data.pY[Y_val] * guassian(X_vec, data.mean[Y_val], data.var[Y_val])
        end
        prediction[i] = sort(collect(Y_dict), by=m->m[2], rev=true)[1][1]
    end
    return prediction
end

end