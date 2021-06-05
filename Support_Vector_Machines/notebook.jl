### A Pluto.jl notebook ###
# v0.14.7

using Markdown
using InteractiveUtils

# ╔═╡ 6a41e9d0-c58c-11eb-078e-c36bf22a38d2
md"""
# Support Vector Machines

[Reference 1](https://scikit-learn.org/stable/modules/svm.html)\
[Reference 2](https://en.wikipedia.org/wiki/Support_vector_machine)\
[Reference 3](http://www.robots.ox.ac.uk/~az/lectures/ml/lect3.pdf)
"""

# ╔═╡ 9cfc2328-0e4f-4a86-944c-31c0c133fe9a
md"""
The SVM classification problem (directly taken from sklearn website):\


Given training vectors $x_i\in \mathbb{R}^P$, $i=1,...,n$, and a vector $y\in\{1,-1\}^n$\
The goal is to find $w\in \mathbb{R}^P$ and $b\in \mathbb{R}$ such that prediction given by $\text{sign}(w^T\phi(x)+b)$ is accurate for most samples\


Solve the following __primal problem__:\
$$\begin{align}
\min_{w,b,\zeta}\frac{1}{2}w^Tw+C\sum^n_{i=1}\zeta_i\\
\text{subject to }y_i(w^T\phi(x_i)+b)\geq1-\zeta_i,\\
\zeta_i\geq0,i=1,...,n
\end{align}$$\


Intuition is we are maximizing the margin (by minimizing $\|w\|^2=w^Tw$), while penalize when a sample is misclassified\


The __dual problem__ to primal is:\
$$\begin{align}
\min_\alpha\frac{1}{2}\alpha^TQ\alpha-e^T\alpha\\
\text{subject to }y^T\alpha=0,\\
0\leq\alpha_i\leq C,i=1,...,n
\end{align}$$\


 $e$ is vector of all ones, $Q$ is nxn positive semidefinite matrix, $Q_{ij}\equiv y_iy_jK(x_i,x_j)$, and $K(x_i,x_j)=\phi(x_i)^T\phi(x_j)$ is the kernel  
$\alpha_i$ are dual coefficients, upper-bounded by $C$\
Highlighting the fact that training vectors are implicitly mapped to a higher (or infinite) dimensional space by function $\phi$\


After optimization problem solved, the __output__ for a given sample $x$ is:
$$\sum_{i\in SV}y_i\alpha_iK(x_i,x)+b$$\


The problem solved by `liblinear` for `LinearSVC` is a equivant form of primal problem:\
$$\min_{w,b}\frac{1}{2}w^Tw+C\sum_{i=1}\max(0,y_i(w^T\phi(x_i)+b))$$\
which does not involve inner products between samples, and therefore cannot apply kernel tricks\


$$C=\frac{1}{\text{alpha}}$$
"""

# ╔═╡ 30975066-eb17-4682-bda1-fd77f9d44734
function ingredients(path::String)
	# this is from the Julia source code (evalfile in base/loading.jl)
	# but with the modification that it returns the module instead of the last object
	name = Symbol(basename(path))
	m = Module(name)
	Core.eval(m,
        Expr(:toplevel,
             :(eval(x) = $(Expr(:core, :eval))($name, x)),
             :(include(x) = $(Expr(:top, :include))($name, x)),
             :(include(mapexpr::Function, x) = $(Expr(:top, :include))(mapexpr, $name, x)),
             :(include($path))))
	m
end

# ╔═╡ 9088c5d0-f07c-404f-8e08-2c8585f86a2f
import Random

# ╔═╡ fc9d6f80-70c1-4518-b9db-29c6e3e2d280
import Statistics

# ╔═╡ 6a0019bf-ab70-4095-ac52-7606e401d9a5
import LinearAlgebra

# ╔═╡ b8447232-6b1d-4b3a-8d11-a9a2da1cf31b
JuTools = ingredients("../tools.jl").JuTools

# ╔═╡ e4380307-4abe-4a20-9127-e608d1a5fa6e
md"""
### Linear SVM Implementation (Hinge Loss)

[Reference](https://stackoverflow.com/questions/48804198/soft-margin-in-linear-support-vector-machine-using-python)\


Cost function is:\
$$J=\frac{1}{2}w^Tw+\frac{C}{N}\sum^N_{i=1}\max\Big(0,1-y_i(w^T\phi(x_i)+b)\Big)$$


Gradient function for $w$ is:\
$$\frac{\partial J}{\partial w}=w + \frac{C}{N}\sum^N_{i=1}\begin{cases}
0 & y_i(w^T\phi(x_i)+b) \geq 1\\
-y_i\phi(x_i) & \text{otherwise}
\end{cases}$$


Gradient function for $b$ is:\
$$\frac{\partial J}{\partial b}=\frac{C}{N}\sum^N_{i=1}\begin{cases}
0 & y_i(w^T\phi(x_i)+b) \geq 1\\
-y_i & \text{otherwise}
\end{cases}$$
"""

# ╔═╡ 6841f5fe-45ad-4f71-84d5-368d196a40ab
# define a struct to store weights
# this should be returned by a training function
# alpha should be treated as constant
mutable struct WeightsLinearSVM
    C::AbstractFloat
    w::Array{T} where T<:AbstractFloat
    b::AbstractFloat
end

# ╔═╡ a093820c-20a0-4af1-98b3-ea92ad37bc77
# define cost function for linear SVM
# assum Y_data is {-1, 1}
function cost(X_data::Array{T} where T<:Number, Y_data::Array{T} where T<:Number, weights::WeightsLinearSVM)::AbstractFloat
    @assert ndims(Y_data) == ndims(weights.w) == 1
    @assert size(X_data) == (size(Y_data)[1], size(weights.w)[1])
    loss_w = 0.5 * (weights.w' * weights.w)
    loss_inner = 1.0 .- Y_data .* vec(X_data * weights.w .+ weights.b)
    loss_inner .= map(m->max(0.0,m), loss_inner)
    loss = loss_w + weights.C * sum(loss_inner) / size(X_data)[1]
    return loss
end

# ╔═╡ 8c3fa961-d88d-4e28-98e9-18b8804c568e
X_data, Y_data = JuTools.data_generate_linear_2d()

# ╔═╡ 6341ae48-9cda-4019-bfe6-443caf8788aa
Y_data .= Y_data .* 2.0 .- 1.0 # convert from {0,1} to {-1,1}

# ╔═╡ 500703d2-e315-44c7-8d6c-c36f40b815df
X_train, X_test, Y_train, Y_test = JuTools.split_data(X_data, Y_data)

# ╔═╡ 7531e45c-95fd-4664-8c37-3364aae7020b
@show size(X_train)

# ╔═╡ 747fabf5-fd7b-40cc-b5da-b15f6925cff0
@show size(X_test)

# ╔═╡ 9504574b-2281-4f8a-83d5-3eea38acec53
@show size(Y_train)

# ╔═╡ c496efb2-5a50-4131-bd65-e9025a3273a4
@show size(Y_test)

# ╔═╡ 46e9b187-d445-4efe-b4f2-ad824e5ec832
weight_test = WeightsLinearSVM(1.0, Random.randn(size(X_data)[2]), Random.randn())

# ╔═╡ 8adbc614-9440-42d3-9672-f3b816e114a4
# define the learning function (gradient descent)
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

# ╔═╡ 633040d0-651b-427f-a5a8-83124bbea43b
# define prediction function
function predict_proba(X_predict::Array{T} where T<:Number, weights::WeightsLinearSVM)::Array
    @assert ndims(X_predict) == 2
    @assert size(X_predict)[2] == size(weights.w)[1]
    prediction = vec(X_predict * weights.w .+ weights.b)
    return prediction
end

# ╔═╡ ea07e163-d2a0-4447-9239-64d1aee8b3a0
# output prediction is in {-1, 1}
function predict(X_predict::Array{T} where T<:Number, weights::WeightsLinearSVM)::Array
    @assert ndims(X_predict) == 2
    @assert size(X_predict)[2] == size(weights.w)[1]
    prediction = vec(X_predict * weights.w .+ weights.b)
    prediction .= map(m -> m >= 0 ? 1.0 : -1.0, prediction)
    return prediction
end

# ╔═╡ 77bd6c88-335f-46a9-a46b-3d3596d54bf9
md"""
### SVM (with various kernels)

[Reference](https://en.wikipedia.org/wiki/Quadratic_programming)\


For solving SVM minimization problem:\
$$\begin{align}
\min_\alpha\frac{1}{2}\alpha^TQ\alpha-e^T\alpha\\
\text{subject to }y^T\alpha=0,\\
0\leq\alpha_i\leq C,i=1,...,n
\end{align}$$\


where $e$ is vector of all ones, $Q$ is nxn positive semidefinite matrix, $Q_{ij}\equiv y_iy_jK(x_i,x_j)$, and $K(x_i,x_j)=\phi(x_i)^T\phi(x_j)$ is the kernel  
$\alpha_i$ are dual coefficients, upper-bounded by $C$\


We'll be using a Quatratic Programming technique: [Sequential Minimal Optimization](https://en.wikipedia.org/wiki/Sequential_minimal_optimization)(SMO)\


My implementation is inspired from [this blog](https://jonchar.net/notebooks/SVM/), whose code is originally from [this paper](https://www.researchgate.net/publication/234786663_Fast_Training_of_Support_Vector_Machines_Using_Sequential_Minimal_Optimization)\


The problem can also be written into (objective function):\
$$\begin{align}
\max_\alpha e^T\alpha-\frac{1}{2}\alpha^TQ\alpha\\
\text{subject to }y^T\alpha=0,\\
0\leq\alpha_i\leq C,i=1,...,n
\end{align}$$\


The kernel functions that I'm going to implement are (using sklearn names):
* Linear Kernel: $\langle x_i,x_j \rangle$  
* Polynomial Kernel: $(\gamma\langle x_i,x_j \rangle + r)^d$ ($d$ is degree, and $r$ is coeficient, $\gamma$ is a parameter)  
* Rbf Kernel: $\exp(-\gamma\|x_i-x_j\|^2)$, where $\gamma$ is a parameter  
* Sigmoid Kernel: $\tanh(\gamma\langle x_i,x_j \rangle + r)$, where $r$ and $\gamma$ are parameters  
"""

# ╔═╡ 111304b2-bfbb-4875-838a-93c153d452be
# define a struct to store information
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

# ╔═╡ 3f6728b3-fb75-4783-ab6e-96ae6301f7cb
# linear kernel
function kernel_linear(X1::Array{T} where T<:Number, X2::Array{T} where T<:Number)::Array
    @assert ndims(X1) == ndims(X2) == 2
    @assert size(X1)[2] == size(X2)[2]
    result = X1 * X2'
    return result
end

# ╔═╡ e3b0cd51-75bc-41aa-96eb-251b376a183d
# polynomial kernel
function kernel_polynomial(X1::Array{T} where T<:Number, X2::Array{T} where T<:Number;
        d::AbstractFloat=1.0, r::AbstractFloat=0.0, gamma::AbstractFloat=1.0)::Array
    @assert ndims(X1) == ndims(X2) == 2
    @assert size(X1)[2] == size(X2)[2]
    result = (gamma .* (X1 * X2') .+ r) .^ d
    return result
end

# ╔═╡ 4a0ee69a-6151-423b-9227-65fa4e9f2b5e
# rbf kernel
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

# ╔═╡ 85939fbf-de18-46d8-9708-dec8a85f49dd
# sigmoid kernel
function kernel_sigmoid(X1::Array{T} where T<:Number, X2::Array{T} where T<:Number; gamma::AbstractFloat=1.0, r::AbstractFloat=0.0)::Array
    @assert ndims(X1) == ndims(X2) == 2
    @assert size(X1)[2] == size(X2)[2]
    result = gamma .* (X1 * X2') .+ r
    result .= tanh.(gamma)
    return result
end

# ╔═╡ 99d0a23e-e06b-4c13-89a7-62aeebf7c559
# because we call it cost function, we will use the original formula
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

# ╔═╡ 50e8d679-d4fc-4ed8-9868-b59cc4151d2c
cost(X_data, Y_data, weight_test)

# ╔═╡ ece26477-6e53-4536-a6a8-61947cb2815e
# define learning each step function
# update weights in place, and return num steps
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
    eta = 2 * k12 - k11 - k22
    # compute new alpha2 (a2)
    a2 = nothing
    if eta < 0.0
        a2 = alpha2 - Y2 * (error1 - error2) / eta
        a2 = min(a2, H)
        a2 = max(a2, L)
    else
        weights.alpha[id2] = L
        Lobj = -cost(X_data, weights)
        weights.alpha[id2] = H
        Hobj = -cost(X_data, weights)
        weights.alpha[id2] = alpha2
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
    non_optimized_ids = [i for i in 1:size(X_data)[1] if (i != id1 && i != id2 && (0 < weights.alpha[i] < weights.C))]
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
        ((Y1*(a1-alpha1)) .* kerr1) .+ ((Y2*(a2-alpha2)) .* kerr2) .+ (b_new - weights.b)
    weights.error[id1] = 0.0
    weights.error[id2] = 0.0
    # update alpha and b
    weights.b = b_new
    weights.alpha[id1] = a1
    weights.alpha[id2] = a2
    return 1
end

# ╔═╡ 59c861e3-8c4d-450d-b727-c59cdf17a84f
# now define the learning function
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
        new_id = -1
        tmax = 0
        if verbose
            println("Trying argmax(abs(E1 - E2))")
        end
        for i in alpha_target
            tmp = abs(error - weights.error[i])
            if(tmp > tmax)
                tmax = tmp
                new_id = i
            end
        end
        if new_id >= 1
            step = learn_step!(X_data, Y_data, weights, id, new_id)
            if step > 0
                return step
            end
        end
        # loop non-bound alphas, randomly
        if verbose
            println("Trying random non-bound alphas")
        end
        for new_id in alpha_target[Random.randperm(length(alpha_target))]
            step = learn_step!(X_data, Y_data, weights, id, new_id)
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
            step = learn_step!(X_data, Y_data, weights, id, new_id)
            if step > 0
                return step
            end
        end
    end
    return 0
end

# ╔═╡ 882b95f5-6261-42d4-91b2-16a7ce9c1aa4
# implement predict functions
function predict_proba(X_predict::Array{T} where T<:Number, X_data::Array{T} where T<:Number,
        Y_data::Array{T} where T<:Number, weights::WeightsSVM)::Array
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

# ╔═╡ dde248cd-6fac-4c59-877d-58e5b8f06685
function predict(X_predict::Array{T} where T<:Number, X_data::Array{T} where T<:Number,
        Y_data::Array{T} where T<:Number, weights::WeightsSVM)::Array
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

# ╔═╡ aafac217-757c-442a-88b1-17e8895f703a
# training function for linear SVM
# assume Y_data is in {-1, 1}
# this function is similar to the training function for Logistic Regression (Both are gradient descent)
function train_linear(X_data::Array{T} where T<:Number, Y_data::Array{T} where T<:Number, C::AbstractFloat;
        learning_rate::AbstractFloat=0.1, max_iter::Integer=1000, n_iter_no_change::Integer=5, tol::AbstractFloat=0.001,
        verbose::Bool=false, shuffle::Bool=true, early_stop::Bool=true)::WeightsLinearSVM
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
    weights = WeightsLinearSVM(C, Random.randn(size(X_data)[2]), Random.randn())
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

# ╔═╡ ec256a57-ea44-4f73-af5b-f5e09ba2fec1
weights = train_linear(X_train, Y_train, 1.0, learning_rate=0.005, max_iter=20, tol=0.001, verbose=true)

# ╔═╡ fb710b22-79b0-477c-b2a6-f85fe8b4271a
JuTools.compute_accuracy(predict(X_test, weights), Y_test)

# ╔═╡ 47846484-1824-41b3-b26d-e4e3953fd079
# finally implement the training function
function train(X_data::Array{T} where T<:Number, Y_data::Array{T} where T<:Number, C::AbstractFloat;
        tol_alpha::AbstractFloat=0.001, tol_error::AbstractFloat=0.001, kernel::String="rbf", gamma::String="scale",
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

# ╔═╡ fdd56685-18d1-47b5-9b9c-27090422f250
weights_svm = train(X_train, Y_train, 10.0, kernel="rbf", verbose=false, gamma="auto", coef=0.0, degree=2.0)

# ╔═╡ 1b11d952-a35f-4778-8a7f-779eb72f0436
JuTools.compute_accuracy(predict(X_test, X_train, Y_train, weights_svm), Y_test)

# ╔═╡ Cell order:
# ╟─6a41e9d0-c58c-11eb-078e-c36bf22a38d2
# ╟─9cfc2328-0e4f-4a86-944c-31c0c133fe9a
# ╟─30975066-eb17-4682-bda1-fd77f9d44734
# ╠═9088c5d0-f07c-404f-8e08-2c8585f86a2f
# ╠═fc9d6f80-70c1-4518-b9db-29c6e3e2d280
# ╠═6a0019bf-ab70-4095-ac52-7606e401d9a5
# ╠═b8447232-6b1d-4b3a-8d11-a9a2da1cf31b
# ╟─e4380307-4abe-4a20-9127-e608d1a5fa6e
# ╠═6841f5fe-45ad-4f71-84d5-368d196a40ab
# ╠═a093820c-20a0-4af1-98b3-ea92ad37bc77
# ╠═8c3fa961-d88d-4e28-98e9-18b8804c568e
# ╠═6341ae48-9cda-4019-bfe6-443caf8788aa
# ╠═500703d2-e315-44c7-8d6c-c36f40b815df
# ╠═7531e45c-95fd-4664-8c37-3364aae7020b
# ╠═747fabf5-fd7b-40cc-b5da-b15f6925cff0
# ╠═9504574b-2281-4f8a-83d5-3eea38acec53
# ╠═c496efb2-5a50-4131-bd65-e9025a3273a4
# ╠═46e9b187-d445-4efe-b4f2-ad824e5ec832
# ╠═50e8d679-d4fc-4ed8-9868-b59cc4151d2c
# ╠═8adbc614-9440-42d3-9672-f3b816e114a4
# ╠═633040d0-651b-427f-a5a8-83124bbea43b
# ╠═ea07e163-d2a0-4447-9239-64d1aee8b3a0
# ╠═aafac217-757c-442a-88b1-17e8895f703a
# ╠═ec256a57-ea44-4f73-af5b-f5e09ba2fec1
# ╠═fb710b22-79b0-477c-b2a6-f85fe8b4271a
# ╟─77bd6c88-335f-46a9-a46b-3d3596d54bf9
# ╠═111304b2-bfbb-4875-838a-93c153d452be
# ╠═3f6728b3-fb75-4783-ab6e-96ae6301f7cb
# ╠═e3b0cd51-75bc-41aa-96eb-251b376a183d
# ╠═4a0ee69a-6151-423b-9227-65fa4e9f2b5e
# ╠═85939fbf-de18-46d8-9708-dec8a85f49dd
# ╠═99d0a23e-e06b-4c13-89a7-62aeebf7c559
# ╠═ece26477-6e53-4536-a6a8-61947cb2815e
# ╠═59c861e3-8c4d-450d-b727-c59cdf17a84f
# ╠═882b95f5-6261-42d4-91b2-16a7ce9c1aa4
# ╠═dde248cd-6fac-4c59-877d-58e5b8f06685
# ╠═47846484-1824-41b3-b26d-e4e3953fd079
# ╠═fdd56685-18d1-47b5-9b9c-27090422f250
# ╠═1b11d952-a35f-4778-8a7f-779eb72f0436
