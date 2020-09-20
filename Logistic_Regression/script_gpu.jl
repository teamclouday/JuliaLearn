# Logistic Regression Module with CUDA support
# Logistic Regression Module
module LogRegGPU
export train, predict, predict_proba, scale, cost, accuracy

using CUDA
import Random
import Statistics

CUDA.allowscalar(false)

"""
Helper function to scale input `X` to unit variance\\
`X` is required to have dimension 2
"""
function scale(X::CuArray)::CuArray
    @assert ndims(X) == 2
    u = Statistics.mean(X, dims=1)
    s = Statistics.std(X, dims=1)
    res = (X .- u) ./ s
    return res
end

"""
Sigmoid function on array `Z`
"""
function sigmoid(Z::CuArray)::CuArray
    denom = 1 .+ (float(MathConstants.e) .^ (-Z))
    return 1 ./ denom
end

"""
Cost function for logistic Regression\\
If `X` is shape (M, N)\\
`y` should be (M,)\\
`beta` should be (N+1,)
------
May not be used
"""
function cost(X::CuArray, y::CuArray, beta::CuArray)::AbstractFloat
    @assert ndims(X) == 2
    @assert ndims(y) == 1
    @assert ndims(beta) == 1
    @assert size(X) == (size(y)[1], size(beta)[1]-1)
    m = size(X)[1]
    X_extended = hcat(X, ones(size(X)[1]))
    X_combined = X_extended * reshape(beta, (size(beta)[1], 1))
    prob = sigmoid(X_combined)
    y_prep = reshape(y, (size(y)[1], 1))
    vec = y_prep .* (log.(prob)) .+ (1 .- y_prep) .* (log.(1 .- prob))
    cost = -1 / float(m) * sum(vec)
    return cost
end

"""
Predict function for Logistic Regression\\
If `X` is shape (M, N)\\
`beta` should be (N+1,)\\
Returns 1d array of real probabilities
"""
function predict_proba(X::CuArray, beta::CuArray)::CuArray
    @assert ndims(X) == 2
    @assert ndims(beta) == 1
    @assert size(X)[2] == size(beta)[1]-1
    X_extended = hcat(X, ones(size(X)[1]))
    X_combined = X_extended * reshape(beta, (size(beta)[1], 1))
    prob = sigmoid(X_combined)
    prob = reshape(prob, (size(prob)[1]*size(prob)[2],)) # flatten to 1d array
    return prob
end

"""
Predict function for Logistic Regression\\
If `X` is shape (M, N)\\
`beta` should be (N+1,)\\
Returns 1d array of 0,1
"""
function predict(X::CuArray, beta::CuArray)::CuArray
    @assert ndims(X) == 2
    @assert ndims(beta) == 1
    @assert size(X)[2] == size(beta)[1]-1
    X_extended = hcat(X, ones(size(X)[1]))
    X_combined = X_extended * reshape(beta, (size(beta)[1], 1))
    prob = sigmoid(X_combined)
    prob = reshape(prob, (size(prob)[1]*size(prob)[2],)) # flatten to 1d array
    real_prob::CuArray{Integer} = map(m -> m >= 0.5 ? 1.0 : 0.0, prob)
    return real_prob
end

"""
Learning function for Gradient Descent\\
If `X` is shape (M, N)\\
`y` should be (M,)\\
`beta` should be (N+1,)
------
Returns `nothing`
------
Note: `beta` will be updated inplace
"""
function learn!(X::CuArray, y::CuArray, beta::CuArray, alpha::AbstractFloat)
    @assert ndims(X) == 2
    @assert ndims(y) == 1
    @assert ndims(beta) == 1
    @assert size(X) == (size(y)[1], size(beta)[1]-1)
    predictions = predict_proba(X, beta)
    offset = predictions .- y
    offset = reshape(offset, (size(offset)[1], 1))
    X_extended = hcat(X, ones(size(X)[1]))
    gradients = X_extended' * offset
    gradients = gradients ./ size(X)[1]
    gradients = gradients .* alpha
    beta .= beta .- reshape(gradients, (size(gradients)[1]*size(gradients)[2],))
    return nothing
end

"""
Learning function for Gradient Descent\\
If `X` is shape (M, N)\\
`y` should be (M,)\\
`beta` should be (N+1,)
------
Returns updated `beta`
"""
function learn(X::CuArray, y::CuArray, beta::CuArray, alpha::AbstractFloat)::CuArray
    @assert ndims(X) == 2
    @assert ndims(y) == 1
    @assert ndims(beta) == 1
    @assert size(X) == (size(y)[1], size(beta)[1]-1)
    predictions = predict_proba(X, beta)
    offset = predictions .- y
    offset = reshape(offset, (size(offset)[1], 1))
    X_extended = hcat(X, ones(size(X)[1]))
    gradients = X_extended' * offset
    gradients = gradients ./ size(X)[1]
    gradients = gradients .* alpha
    beta = beta .- reshape(gradients, (size(gradients)[1]*size(gradients)[2],))
    return beta
end

"""
Training function for Logisitic Regression\\
Implemented using Gradient Descent\\
If `X` is shape (M, N)\\
`y` should be (M,)\\
`max_iter` should be >= 0
------
if `return_all`, then return all `beta`(weights) for each iteration\\
else, return the final `beta`(weight) after iterations
"""
function train(X::CuArray, y::CuArray; learning_rate::AbstractFloat=0.01, max_iter::Integer=100, return_all::Bool=false)::CuArray
    @assert ndims(X) == 2
    @assert ndims(y) == 1
    @assert size(X)[1] == size(y)[1]
    @assert max_iter >= 0
    beta = CuArray(Random.randn(size(X)[2]+1))
    res = nothing
    if return_all
        res = reshape(beta, (1, size(beta)[1]))
    else
        res = beta
    end
    for i = 1:max_iter
        if return_all
            beta = learn(X, y, beta, learning_rate)
            res = cat(res, reshape(beta, (1, size(beta)[1])), dims=1)
        else
            learn!(X, y, beta, learning_rate)
            res .= beta
        end
    end
    return res
end

"""
Accuracy function for Logisitic Regression\\
`y_pred` and `y_real` should have same shape, 1d array
------
Returns accuracy as float
"""
function accuracy(y_pred::CuArray, y_real::CuArray)::AbstractFloat
    @assert ndims(y_pred) == ndims(y_real) == 1
    @assert size(y_pred) == size(y_real)
    sum = 0
    for (m, n) in zip(y_pred, y_real)
        if m == n
            sum += 1
        end
    end
    return sum / size(y_pred)[1]
end

end