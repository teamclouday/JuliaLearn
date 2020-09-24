# Support Vector Machines Module
module SVM
export WeightsLinearSVM, train_linear
export cost, predict, predict_proba

import Random

include("../tools.jl")
import .JuTools

"""
The weight data struct for training linear SVM
"""
mutable struct WeightsLinearSVM
    C::AbstractFloat
    w::Array{T} where T<:AbstractFloat
    b::AbstractFloat
end

"""
Cost function for linear SVM\\
If `X_data` has shape (M,N)\\
`Y_data` has shape (M,)\\
`weights.w` has shape (N,)\\
Returns the computed loss
"""
function cost(X_data::Array{T} where T<:Number, Y_data::Array{T} where T<:Number, weights::WeightsLinearSVM)::AbstractFloat
    @assert ndims(Y_data) == ndims(weights.w) == 1
    @assert size(X_data) == (size(Y_data)[1], size(weights.w)[1])
    loss_w = 0.5 * (weights.w' * weights.w)
    loss_inner = 1.0 .- Y_data .* vec(X_data * weights.w .+ weights.b)
    loss_inner .= map(m->max(0.0,m), loss_inner)
    loss = loss_w + weights.C * sum(loss_inner) / size(X_data)[1]
    return loss
end

"""
Learning function for linear SVM, using Gradient Descent\\
If `X_data` has shape (M,N)\\
`Y_data` has shape (M,)\\
`weights.w` has shape (N,)\\
Update the weights in place
"""
function learn!(X_data::Array{T} where T<:Number, Y_data::Array{T} where T<:Number, weights::WeightsLinearSVM, alpha::AbstractFloat)
    @assert ndims(Y_data) == ndims(weights.w) == 1
    @assert size(X_data) == (size(Y_data)[1], size(weights.w)[1])
    # compute deciding feature
    decide = (Y_data .* (X_data * weights.w .+ weights.b)) .< 1 # (? < 1) will be 1, otherwise 0
    # update w
    gradient_w = weights.w .+ (weights.C / size(X_data)[1]) .* vec(-(Y_data .* decide)' * X_data)
    gradient_w .= gradient_w .* alpha
    weights.w .= weights.w .- gradient_w
    # update b
    gradient_b = (weights.C / size(X_data)[1]) * sum(-(Y_data .* decide))
    gradient_b *= alpha
    weights.b = weights.b - gradient_b
    return nothing
end

"""
Prediction function for linear SVM\\
If `X_data` has shape (M,N)\\
`weights.w` has shape (N,)\\
Returns the actual computed values
"""
function predict_proba(X_predict::Array{T} where T<:Number, weights::WeightsLinearSVM)::Array
    @assert 1 <= ndims(X_predict) <= 2
    if ndims(X_predict) == 1
        X_predict = reshape(X_predict, (1, size(X_predict)[1]))
    end
    @assert size(X_predict)[2] == size(weights.w)[1]
    prediction = vec(X_predict * weights.w .+ weights.b)
    return prediction
end

"""
Prediction function for linear SVM\\
If `X_data` has shape (M,N)\\
`weights.w` has shape (N,)\\
Returns the converted predictions, in {-1, 1}
"""
function predict(X_predict::Array{T} where T<:Number, weights::WeightsLinearSVM)::Array
    @assert 1 <= ndims(X_predict) <= 2
    if ndims(X_predict) == 1
        X_predict = reshape(X_predict, (1, size(X_predict)[1]))
    end
    @assert size(X_predict)[2] == size(weights.w)[1]
    prediction = vec(X_predict * weights.w .+ weights.b)
    prediction .= map(m -> m >= 0 ? 1.0 : -1.0, prediction)
    return prediction
end

"""
Training function for linear SVM\\
If `X_data` has shape (M,N)\\
`Y_data` has shape (M,)\\
`C` is the constraint for SVM decision margin\\
`max_iter` should be >= 0
------
Returns trained weights as `WeightsLinearSVM` object

------
Set `early_stop` to `false`, to force run maximum iteractions\\
Set `random_weights` to `false` to initialize weights of 0.0
"""
function train_linear(X_data::Array{T} where T<:Number, Y_data::Array{T} where T<:Number, C::AbstractFloat;
        learning_rate::AbstractFloat=0.1, max_iter::Integer=1000, n_iter_no_change::Integer=5, tol::AbstractFloat=0.001,
        verbose::Bool=false, shuffle::Bool=true, early_stop::Bool=true, random_weights::Bool=true)::WeightsLinearSVM
    @assert ndims(X_data) == ndims(Y_data) + 1 == 2
    @assert size(X_data)[1] == size(Y_data)[1]
    @assert max_iter >= 0
    @assert n_iter_no_change >= 0
    @assert tol >= 0
    X_data = Float64.(X_data)
    Y_data = Float64.(Y_data)
    if shuffle
        JuTools.shuffle_data!(X_data, Y_data)
    end
    # is it better to use zero weights than normal weights ?
    weights = nothing
    if random_weights
        weights = WeightsLinearSVM(C, Random.randn(size(X_data)[2]), Random.randn())
    else
        weights = WeightsLinearSVM(C, zeros(size(X_data)[2]), 0.0)
    end
    best_cost = nothing
    n_cost_no_change = n_iter_no_change
    for i in 1:max_iter
        if n_cost_no_change <= 0 && early_stop
            break
        end
        learn!(X_data, Y_data, weights, learning_rate)
        new_cost = cost(X_data, Y_data, weights)
        if verbose
            acc = JuTools.compute_accuracy(predict(X_data, weights), Y_data)
            println("Iter: $i")
            println("Cost = $new_cost")
            println("Accuracy = $acc")
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
    return weights
end

end