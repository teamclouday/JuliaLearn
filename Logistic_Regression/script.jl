# Logistic Regression Module
module LogReg
export train, predict, predict_proba, scale, cost, accuracy

import Random
import Statistics

include("../tools.jl")
import .JuTools

"""
Helper function to scale input `X` to unit variance\\
`X` is required to have dimension 2
"""
function scale(X::Array)::Array
    @assert ndims(X) == 2
    u = Statistics.mean(X, dims=1)
    s = Statistics.std(X, dims=1)
    res = (X .- u) ./ s
    return res
end

"""
Sigmoid function on array `Z`
"""
function sigmoid(Z::Array)::Array
    denom = 1 .+ (Float64(MathConstants.e) .^ (-Z))
    return 1 ./ denom
end

"""
Cost function for logistic Regression\\
If `X` is shape (M, N)\\
`y` should be (M,)\\
`beta` can be (N,) or (N+1,)
------
May not be used
"""
function cost(X::Array, y::Array, beta::Array)::AbstractFloat
    @assert ndims(X) == 2
    @assert ndims(y) == 1
    @assert ndims(beta) == 1
    @assert size(X)[1] == size(y)[1]
    @assert size(X)[2] == size(beta)[1] || size(X)[2] == size(beta)[1] - 1
    if size(X)[2] < size(beta)[1]
        X = hcat(X, ones(size(X)[1]))
    end
    m = size(X)[1]
    X_combined = X * beta
    prob = sigmoid(X_combined)
    vec = y .* (log.(prob)) .+ (1 .- y) .* (log.(1 .- prob))
    cost = -1 / float(m) * sum(vec)
    return cost
end

"""
Predict function for Logistic Regression\\
If `X` is shape (M, N)\\
`beta` can be (N,) or (N+1,)\\
Returns 1d array of real probabilities
"""
function predict_proba(X::Array, beta::Array)::Array
    @assert ndims(X) == 2
    @assert ndims(beta) == 1
    @assert size(X)[2] == size(beta)[1] || size(X)[2] == size(beta)[1] - 1
    if size(X)[2] < size(beta)[1]
        X = hcat(X, ones(size(X)[1]))
    end
    X_combined = X * beta
    prob = sigmoid(X_combined)
    return prob
end

"""
Predict function for Logistic Regression\\
If `X` is shape (M, N)\\
`beta` can be (N,) or (N+1,)\\
Returns 1d array of 0,1
"""
function predict(X::Array, beta::Array)::Array
    @assert ndims(X) == 2
    @assert ndims(beta) == 1
    @assert size(X)[2] == size(beta)[1] || size(X)[2] == size(beta)[1] - 1
    if size(X)[2] < size(beta)[1]
        X = hcat(X, ones(size(X)[1]))
    end
    X_combined = X * beta
    prob = sigmoid(X_combined)
    real_prob::Array = map(m -> m >= 0.5 ? 1.0 : 0.0, prob)
    return real_prob
end

"""
Learning function for Gradient Descent\\
If `X` is shape (M, N)\\
`y` should be (M,)\\
`beta` should be (N,)
------
Returns `nothing`
------
Note: `beta` will be updated inplace
"""
function learn!(X::Array, y::Array, beta::Array, momentum::Array, alpha::AbstractFloat)
    @assert ndims(X) == 2
    @assert ndims(y) == 1
    @assert ndims(beta) == 1
    @assert size(X) == (size(y)[1], size(beta)[1])
    predictions = predict_proba(X, beta)
    offset = predictions .- y
    gradients = X' * offset
    gradients .= gradients ./ size(X)[1]
    gradients .= gradients .* alpha
    momentum .= gradients .+ (0.9 .* momentum)
    beta .= beta .- momentum
    return nothing
end

"""
Training function for Logisitic Regression\\
Implemented using Gradient Descent\\
If `X` is shape (M, N)\\
`y` should be (M,)\\
`max_iter` should be >= 0
------
Returns `beta` in shape (N+1,)

------
Set `early_stop` to `false`, to force run maximum iteractions\\
Set `random_weights` to `false` to initialize weights of 0.0
"""
function train(X::Array, y::Array; learning_rate::AbstractFloat=0.1, max_iter::Integer=1000,
        n_iter_no_change::Integer=5, tol::AbstractFloat=0.001, verbose::Bool=false,
        shuffle::Bool=true, early_stop::Bool=true, random_weights::Bool=true)::Array
    @assert ndims(X) == 2
    @assert ndims(y) == 1
    @assert size(X)[1] == size(y)[1]
    @assert max_iter >= 0
    @assert n_iter_no_change >= 0
    @assert tol >= 0.0
    X = hcat(X, ones(size(X)[1])) # for constant multiplication
    X .= Float64.(X)
    y = Float64.(y)
    if shuffle
        JuTools.shuffle_data!(X, y)
    end
    beta = nothing
    if random_weights
        beta = Random.randn(size(X)[2])
    else
        beta = zeros(size(X)[2])
    end
    momentum = zeros(size(X)[2])
    best_cost = nothing
    n_cost_no_change = n_iter_no_change
    for i = 1:max_iter
        if n_cost_no_change <= 0 && early_stop
            break
        end
        learn!(X, y, beta, momentum, learning_rate)
        new_cost = cost(X, y, beta)
        if verbose
            acc = accuracy(predict(X, beta), y)
            println("Iter: ", i)
            println("Cost = ", new_cost)
            println("Accuracy = ", acc)
            println()
        end
        if early_stop
            if best_cost === nothing || isnan(best_cost)
                best_cost = new_cost
            else
                if new_cost > best_cost - tol
                    n_cost_no_change -= 1
                else
                    best_cost = min(new_cost, best_cost)
                    n_cost_no_change = n_iter_no_change
                end
            end
        end
    end
    return beta
end

"""
Accuracy function for Logisitic Regression\\
`y_pred` and `y_real` should have same shape, 1d array
------
Returns accuracy as float
"""
function accuracy(y_pred::Array, y_real::Array)::AbstractFloat
    @assert ndims(y_pred) == ndims(y_real) == 1
    @assert size(y_pred) == size(y_real)
    acc = Statistics.mean(float.(y_pred) .== float.(y_real))
    return acc
end

end