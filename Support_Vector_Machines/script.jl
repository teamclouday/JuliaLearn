# Support Vector Machines Module
module SVM
export WeightsLinearSVM, train_linear
export WeightsSVM, train
export cost, predict, predict_proba

import Random
import Statistics

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
function learn!(X_data::Array{T} where T<:Number, Y_data::Array{T} where T<:Number, weights::WeightsLinearSVM, momentum::WeightsLinearSVM, alpha::AbstractFloat)
    @assert ndims(Y_data) == ndims(weights.w) == 1
    @assert size(X_data) == (size(Y_data)[1], size(weights.w)[1])
    # compute deciding feature
    decide = (Y_data .* (X_data * weights.w .+ weights.b)) .< 1 # (? < 1) will be 1, otherwise 0
    # update w
    gradient_w = weights.w .+ (weights.C / size(X_data)[1]) .* vec(-(Y_data .* decide)' * X_data)
    gradient_w .= gradient_w .* alpha
    momentum.w .= gradient_w .+ (0.9 .* momentum.w)
    weights.w .= weights.w .- momentum.w
    # update b
    gradient_b = (weights.C / size(X_data)[1]) * sum(-(Y_data .* decide))
    gradient_b *= alpha
    momentum.b = gradient_b + (0.9 * momentum.b)
    weights.b = weights.b - momentum.b
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
    momentum = WeightsLinearSVM(C, zeros(size(X_data)[2]), 0.0)
    best_cost = nothing
    n_cost_no_change = n_iter_no_change
    for i in 1:max_iter
        if n_cost_no_change <= 0 && early_stop
            break
        end
        learn!(X_data, Y_data, weights, momentum, learning_rate)
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


"""
The weight data struct for training common SVM (with kernels)
"""
mutable struct WeightsSVM
    C::AbstractFloat                # constraint
    b::AbstractFloat                # threshold
    gamma::AbstractFloat            # parameter used for polynomial, rbf, and sigmoid kernels
    r::AbstractFloat                # parameter used for polynomial, and sigmoid kernels
    d::AbstractFloat                # parameter used for polynomial kernel
    tol_alpha::AbstractFloat        # tolerance for alpha
    tol_error::AbstractFloat        # tolerance for error
    alpha::Array{T} where T<:Number # alpha array
    error::Array{T} where T<:Number # array for error cache
    kernel::String                  # kernel function name
end

"""
Kernel Function (linear)
"""
function kernel_linear(X1::Array{T} where T<:Number, X2::Array{T} where T<:Number)::Array
    @assert ndims(X1) == ndims(X2) == 2
    @assert size(X1)[2] == size(X2)[2]
    result = X1 * X2'
    return result
end
"""
Kernel Function (polynomial)\\
`d` is `degree` parameter in training function\\
`r` is `coef` parameter in training function\\
`gamma` is computed from `gamma` parameter in training function
"""
function kernel_polynomial(X1::Array{T} where T<:Number, X2::Array{T} where T<:Number;
        d::AbstractFloat=1.0, r::AbstractFloat=0.0, gamma::AbstractFloat=1.0)::Array
    @assert ndims(X1) == ndims(X2) == 2
    @assert size(X1)[2] == size(X2)[2]
    result = (gamma .* (X1 * X2') .+ r) .^ d
    return result
end
"""
Kernel Function (rbf)\\
`gamma` is computed from `gamma` parameter in training function
"""
function kernel_rbf(X1::Array{T} where T<:Number, X2::Array{T} where T<:Number; gamma::AbstractFloat=1.0)::Array
    @assert ndims(X1) == ndims(X2) == 2
    @assert size(X1)[2] == size(X2)[2]
    result = (sum(X1 .^ 2, dims=2) * ones(size(X2)[1])') .+ (ones(size(X1)[1]) * sum(X2 .^ 2, dims=2)') .- 2.0 .* (X1 * X2')
    result .= broadcast(m->max(0.0, m), result) # ignore very small negative outputs, due to precision
    result .= sqrt.(result)
    result .= (-gamma) .* result
    result .= exp.(result)
    return result
end
"""
Kernel Function (sigmoid)\\
`r` is `coef` parameter in training function\\
`gamma` is computed from `gamma` parameter in training function
"""
function kernel_sigmoid(X1::Array{T} where T<:Number, X2::Array{T} where T<:Number; gamma::AbstractFloat=1.0, r::AbstractFloat=0.0)::Array
    @assert ndims(X1) == ndims(X2) == 2
    @assert size(X1)[2] == size(X2)[2]
    result = gamma .* (X1 * X2') .+ r
    result .= tanh.(result)
    return result
end

"""
Cost function for SVM\\
If `X_data` has shape (M,N)\\
`weights.alpha` has shape (M,)\\
Returns the computed cost\\
__Note__:\\
Cost is the opposite of objective value
"""
function cost(X_data::Array{T} where T<:Number, weights::WeightsSVM)::AbstractFloat
    @assert ndims(X_data) == ndims(weights.alpha) + 1 == 2
    @assert size(X_data)[1] == size(weights.alpha)[1]
    result = nothing
    if weights.kernel == "linear"
        result = kernel_linear(X_data, X_data)
    elseif weights.kernel == "polynomial"
        result = kernel_polynomial(X_data, X_data, d=weights.d, r=weights.r, gamma=weights.gamma)
    elseif weights.kernel == "rbf"
        result = kernel_rbf(X_data, X_data, gamma=weights.gamma)
    elseif weights.kernel == "sigmoid"
        result = kernel_sigmoid(X_data, X_data, gamma=weights.gamma, r=weights.r)
    else
        throw(ArgumentError("Error: kernel function $weights.kernel is not recognized"))
    end
    result = 0.5 * (weights.alpha' * result * weights.alpha) - sum(weights.alpha)
    return result
end

"""
Step function for learning SVM\\
If `X_data` has shape (M,N)\\
`Y_data` has shape (M,)\\
`weights.alpha` has shape (M,)\\
`weights.error` has shape (M,)\\
`id1` and `id2` are index in `weights.alpha`\\
Tries to update `weights.alpha` in place\\
Returns `1` if step further, else `0`
"""
function learn_step!(X_data::Array{T} where T<:Number, Y_data::Array{T} where T<:Number,
        weights::WeightsSVM, id1::Integer, id2::Integer)::Integer
    @assert ndims(X_data) == ndims(Y_data) + 1 == 2
    @assert size(X_data)[1] == size(Y_data)[1]
    @assert size(X_data)[1] == size(weights.alpha)[1]
    @assert size(weights.error) == size(weights.alpha)
    @assert id1 >= 1
    @assert id2 >= 1
    # if choosing same alpha, skip
    if id1 == id2
        return 0
    end
    # prepare data
    alpha1 = weights.alpha[id1]
    alpha2 = weights.alpha[id2]
    Y1 = Y_data[id1]
    Y2 = Y_data[id2]
    error1 = weights.error[id1]
    error2 = weights.error[id2]
    # compute L & H
    L = nothing
    H = nothing
    if Y1 != Y2
        L = max(0.0, alpha2 - alpha1)
        H = min(weights.C, weights.C + alpha2 - alpha1)
    else
        L = max(0.0, alpha1 + alpha2 - weights.C)
        H = min(weights.C, alpha1 + alpha2)
    end
    if L == H
        return 0
    end
    # compute kernel results and 2nd derivative eta
    k11 = nothing
    k12 = nothing
    k22 = nothing
    n_features = size(X_data)[2]
    X_id1 = reshape(X_data[id1, :], (1, n_features))
    X_id2 = reshape(X_data[id2, :], (1, n_features))
    if weights.kernel == "linear"
        k11 = kernel_linear(X_id1, X_id1)[1]
        k12 = kernel_linear(X_id1, X_id2)[1]
        k22 = kernel_linear(X_id2, X_id2)[1]
    elseif weights.kernel == "polynomial"
        k11 = kernel_polynomial(X_id1, X_id1, d=weights.d, r=weights.r, gamma=weights.gamma)[1]
        k12 = kernel_polynomial(X_id1, X_id2, d=weights.d, r=weights.r, gamma=weights.gamma)[1]
        k22 = kernel_polynomial(X_id2, X_id2, d=weights.d, r=weights.r, gamma=weights.gamma)[1]
    elseif weights.kernel == "rbf"
        k11 = kernel_rbf(X_id1, X_id1, gamma=weights.gamma)[1]
        k12 = kernel_rbf(X_id1, X_id2, gamma=weights.gamma)[1]
        k22 = kernel_rbf(X_id2, X_id2, gamma=weights.gamma)[1]
    elseif weights.kernel == "sigmoid"
        k11 = kernel_sigmoid(X_id1, X_id1, gamma=weights.gamma, r=weights.r)[1]
        k12 = kernel_sigmoid(X_id1, X_id2, gamma=weights.gamma, r=weights.r)[1]
        k22 = kernel_sigmoid(X_id2, X_id2, gamma=weights.gamma, r=weights.r)[1]
    else
        throw(ArgumentError("Error: kernel function $weights.kernel is not recognized"))
    end
    eta = 2.0 * k12 - k11 - k22
    # compute new alpha2 (a2)
    a2 = nothing
    if eta < 0.0
        a2 = alpha2 - Y2 * (error1 - error2) / eta
        a2 = min(a2, H)
        a2 = max(a2, L)
    else
        c1 = eta / 2.0
        c2 = Y2 * (error1 - error2) - eta * alpha2
        Lobj = c1 * L * L + c2 * L
        Hobj = c1 * H * H + c2 * H
        if Lobj > (Hobj + weights.tol_alpha)
            a2 = L
        elseif Lobj < (Hobj - weights.tol_alpha)
            a2 = H
        else
            a2 = alpha2
        end
    end
    # push to 0 or C
    if a2 < 1e-8
        a2 = 0.0
    elseif a2 > (weights.C - 1e-8)
        a2 = weights.C
    end
    # skip if cannot be optimized
    if abs(a2 - alpha2) < weights.tol_alpha * (a2 + alpha2 + weights.tol_alpha)
        return 0
    end
    # compute new alpha1 (a1)
    a1 = alpha1 + (Y1 * Y2) * (alpha2 - a2)
    if a1 < 0.0
        a2 += (Y1 * Y2) * a1
        a1 = 0.0
    elseif a1 > weights.C
        a2 += (Y1 * Y2) * (a1 - weights.C)
        a1 = weights.C
    end
    # update threshold
    b1 = error1 + Y1 * (a1 - alpha1) * k11 + Y2 * (a2 - alpha2) * k12 + weights.b
    b2 = error2 + Y1 * (a1 - alpha1) * k12 + Y2 * (a2 - alpha2) * k22 + weights.b
    b_new = nothing
    if 0 < a1 < weights.C
        b_new = b1
    elseif 0 < a2 < weights.C
        b_new = b2
    else
        b_new = (b1 + b2) * 0.5
    end
    # update error cache
    non_optimized_ids = [i for i in 1:size(X_data)[1] if (0 < weights.alpha[i] < weights.C)]
    kerr1 = nothing
    kerr2 = nothing
    if weights.kernel == "linear"
        kerr1 = vec(kernel_linear(X_id1, X_data[non_optimized_ids, :]))
        kerr2 = vec(kernel_linear(X_id2, X_data[non_optimized_ids, :]))
    elseif weights.kernel == "polynomial"
        kerr1 = vec(kernel_polynomial(X_id1, X_data[non_optimized_ids, :], d=weights.d, r=weights.r, gamma=weights.gamma))
        kerr2 = vec(kernel_polynomial(X_id2, X_data[non_optimized_ids, :], d=weights.d, r=weights.r, gamma=weights.gamma))
    elseif weights.kernel == "rbf"
        kerr1 = vec(kernel_rbf(X_id1, X_data[non_optimized_ids, :], gamma=weights.gamma))
        kerr2 = vec(kernel_rbf(X_id2, X_data[non_optimized_ids, :], gamma=weights.gamma))
    elseif weights.kernel == "sigmoid"
        kerr1 = vec(kernel_sigmoid(X_id1, X_data[non_optimized_ids, :], gamma=weights.gamma, r=weights.r))
        kerr2 = vec(kernel_sigmoid(X_id2, X_data[non_optimized_ids, :], gamma=weights.gamma, r=weights.r))
    end
    weights.error[non_optimized_ids] .= weights.error[non_optimized_ids] .+
        ((Y1*(a1-alpha1)) .* kerr1) .+ ((Y2*(a2-alpha2)) .* kerr2) .- (b_new - weights.b)
    weights.error[id1] = 0.0
    weights.error[id2] = 0.0
    # update alpha and b
    weights.b = b_new
    weights.alpha[id1] = a1
    weights.alpha[id2] = a2
    return 1
end

"""
Learning function for SVM, using SMO\\
If `X_data` has shape (M,N)\\
`Y_data` has shape (M,)\\
`weights.alpha` has shape (M,)\\
Tries to update `weights.alpha` with step function\\
Returns `1` if step further, else `0`
"""
function learn!(X_data::Array{T} where T<:Number, Y_data::Array{T} where T<:Number,
        weights::WeightsSVM, id::Integer; verbose::Bool=false)::Integer
    @assert ndims(X_data) == ndims(Y_data) + 1 == 2
    @assert size(X_data)[1] == size(Y_data)[1]
    @assert size(X_data)[1] == size(weights.alpha)[1]
    @assert size(weights.error) == size(weights.alpha)
    @assert id >= 1
    Y = Y_data[id]
    alpha = weights.alpha[id]
    error = weights.error[id]
    r = error * Y
    if ((r < -weights.tol_error) && (alpha < weights.C)) || ((r > weights.tol_error) && (alpha > 0))
        alpha_target = [i for (i, m) in enumerate(weights.alpha) if (0.0 < m < weights.C)]
        # try argmax E1 - E2
        if verbose
            println("Trying argmax(abs(E1 - E2))")
        end
        new_id = 0
        tmax = 0
        for i in alpha_target
            tmp = abs(error - weights.error[i])
            if(tmp > tmax)
                tmax = tmp
                new_id = i
            end
        end
        if new_id >= 1
            step = learn_step!(X_data, Y_data, weights, new_id, id)
            if step > 0
                return step
            end
        end
        # loop non-bound alphas, randomly
        if verbose
            println("Trying random non-bound alphas")
        end
        for new_id in alpha_target[Random.randperm(length(alpha_target))]
            step = learn_step!(X_data, Y_data, weights, new_id, id)
            if step > 0
                return step
            end
        end
        # else loop all alphas, randomly
        if verbose
            println("Trying random remaining alphas")
        end
        for new_id in Random.randperm(length(weights.alpha))
            if new_id in alpha_target
                continue # skip the alpha ids that already looked at
            end
            step = learn_step!(X_data, Y_data, weights, new_id, id)
            if step > 0
                return step
            end
        end
    end
    return 0
end

"""
Prediction function for SVM\\
If `X_data` has shape (M,N)\\
`Y_data` has shape (M,)\\
`weights.alpha` has shape (M,)\\
`X_predict` has shape (k,N)\\
Returns the actual predictions
"""
function predict_proba(X_predict::Array{T} where T<:Number, X_data::Array{T} where T<:Number,
        Y_data::Array{T} where T<:Number, weights::WeightsSVM)::Array
    if ndims(X_predict) == 1
        X_predict = reshape(X_predict, (1, size(X_predict)[1]))
    end
    @assert ndims(X_predict) == ndims(X_data) == ndims(Y_data) + 1 == 2
    @assert size(X_predict)[2] == size(X_data)[2]
    result = nothing
    if weights.kernel == "linear"
        result = kernel_linear(X_data, X_predict)
    elseif weights.kernel == "polynomial"
        result = kernel_polynomial(X_data, X_predict, d=weights.d, r=weights.r, gamma=weights.gamma)
    elseif weights.kernel == "rbf"
        result = kernel_rbf(X_data, X_predict, gamma=weights.gamma)
    elseif weights.kernel == "sigmoid"
        result = kernel_sigmoid(X_data, X_predict, gamma=weights.gamma, r=weights.r)
    else
        throw(ArgumentError("Error: kernel function $weights.kernel is not recognized"))
    end
    prediction = vec((weights.alpha .* Y_data)' * result)
    return prediction
end

"""
Prediction function for SVM\\
If `X_data` has shape (M,N)\\
`Y_data` has shape (M,)\\
`weights.alpha` has shape (M,)\\
`X_predict` has shape (k,N)\\
Returns the converted predictions, in {-1, 1}
"""
function predict(X_predict::Array{T} where T<:Number, X_data::Array{T} where T<:Number,
        Y_data::Array{T} where T<:Number, weights::WeightsSVM)::Array
    if ndims(X_predict) == 1
        X_predict = reshape(X_predict, (1, size(X_predict)[1]))
    end
    @assert ndims(X_predict) == ndims(X_data) == ndims(Y_data) + 1 == 2
    @assert size(X_predict)[2] == size(X_data)[2]
    result = nothing
    if weights.kernel == "linear"
        result = kernel_linear(X_data, X_predict)
    elseif weights.kernel == "polynomial"
        result = kernel_polynomial(X_data, X_predict, d=weights.d, r=weights.r, gamma=weights.gamma)
    elseif weights.kernel == "rbf"
        result = kernel_rbf(X_data, X_predict, gamma=weights.gamma)
    elseif weights.kernel == "sigmoid"
        result = kernel_sigmoid(X_data, X_predict, gamma=weights.gamma, r=weights.r)
    else
        throw(ArgumentError("Error: kernel function $weights.kernel is not recognized"))
    end
    prediction = vec((weights.alpha .* Y_data)' * result)
    prediction .= map(m -> m >= 0 ? 1.0 : -1.0, prediction)
    return prediction
end

"""
Training function for linear SVM\\
If `X_data` has shape (M,N)\\
`Y_data` has shape (M,)\\
`C` is the constraint for SVM decision margin\\
`tol_alpha` sets tolerance for alpha array\\
`tol_error` sets tolerance for error cache\\
`kernel` selects a kernel function, possible values are:
* "rbf"
* "linear"
* "sigmoid"
* "polynomial"

`gamma` selects a way to compute gamma, possible values are:
* "scale"
* "auto"

`degree` defines the degree in `polynomial` kernel function\\
`coef` defines the `r` in `polynomial` and `sigmoid` kernel functions
------
Returns trained alphas as `WeightsSVM` object
"""
function train(X_data::Array{T} where T<:Number, Y_data::Array{T} where T<:Number, C::AbstractFloat;
        tol_alpha::AbstractFloat=0.01, tol_error::AbstractFloat=0.01, kernel::String="rbf", gamma::String="scale",
        degree::AbstractFloat=1.0, coef::AbstractFloat=0.0, verbose::Bool=false)::WeightsSVM
    @assert ndims(X_data) == ndims(Y_data) + 1 == 2
    @assert size(X_data)[1] == size(Y_data)[1]
    X_data = Float64.(X_data)
    Y_data = Float64.(Y_data)
    # gamma is computed the same way sklearn does
    gamma_num = nothing
    if gamma == "scale"
        gamma_num = 1.0 / (size(X_data)[2] * Statistics.var(X_data))
    elseif gamma == "auto"
        gamma_num = 1.0 / size(X_data)[2]
    else
        throw(ArgumentError("Error: gamma $gamma is not recognized, possible values are 'scale' and 'auto'"))
    end
    weights = WeightsSVM(C, 0.0, gamma_num, coef, degree, tol_alpha, tol_error, Float64.(zeros(size(X_data)[1])), -copy(Y_data), kernel)
    num_changed = 0
    examine_all = true
    total_steps = 0
    while (num_changed > 0) || examine_all
        num_changed = 0
        if examine_all
            if verbose
                println("Scanning all training data")
            end
            for i in 1:size(X_data)[1]
                step = learn!(X_data, Y_data, weights, i, verbose=verbose)
                num_changed += step
                if step > 0 && verbose
                    obj = -cost(X_data, weights)
                    println("1 step further, objective = $obj")
                end
            end
        else
            if verbose
                println("Scanning data whose alpha is not at limit")
            end
            alpha_target = [i for (i, m) in enumerate(weights.alpha) if (m != 0.0 && m != weights.C)]
            for i in alpha_target
                step = learn!(X_data, Y_data, weights, i, verbose=verbose)
                num_changed += step
                if step > 0 && verbose
                    obj = -cost(X_data, weights)
                    println("1 step further, objective = $obj")
                end
            end
        end
        if examine_all
            examine_all = false
        elseif num_changed <= 0
            examine_all = true
        end
        total_steps += num_changed
    end
    if verbose
        println("Training Complete\nTotal steps: $total_steps")
    end
    return weights
end

end