### A Pluto.jl notebook ###
# v0.14.7

using Markdown
using InteractiveUtils

# ╔═╡ 2fdecb39-bff4-4f3c-b3f5-35438541a29d
# import necessary libraries
using Random

# ╔═╡ 29802e3a-8dfe-41c7-a25e-2f0961949854
using Statistics

# ╔═╡ cba2ccb0-c5a2-11eb-11be-41eb0278cb72
md"""
### Logistic Regression

[Reference 1](https://towardsdatascience.com/introduction-to-logistic-regression-66248243c148)\
[Reference 2](https://towardsdatascience.com/building-a-logistic-regression-in-python-301d27367c24)\
[ML Glossary](https://ml-cheatsheet.readthedocs.io/en/latest/logistic_regression.html)  

#### Sigmoid Function

$$f(x)=\frac{1}{1+e^{-(x)}}$$

which maps predicted values to probabilities

#### Hypothesis Representation

$$Z=\beta_0+\beta_1X$$

$$h\Theta(x)=\text{sigmoid}(Z)=\frac{1}{1+e^{-(\beta_0+\beta_1X)}}$$

#### Cost Function

$$\text{Cost}(h_\theta(x),y)=
    \begin{cases}
    -\log(h_\theta(x)) & \quad \text{if } y = 1\\
    -\log(1-h_\theta(x)) & \quad \text{if } y = 0
    \end{cases}$$

which is also

$$J(\theta)=-\frac{1}{m}\sum \Big[ y^{(i)}\log(h\theta(x^{(i)})) + (1-y^{(i)})\log(1-h\theta(x^{(i)})) \Big]$$

which should be minimized

#### Gradient Descent

$$\theta_j := \theta_j - \alpha\frac{\partial}{\partial\theta_j}J(\theta)$$

In this case:

$$\theta_j := \theta_j - \alpha\sum^m_{i=1}(h_\theta(x^{(i)})-y^{(i)})x_j^{(i)}$$
"""

# ╔═╡ 3761501d-f02e-41cb-9003-74cb68283d74
begin
X_data = rand(Float32, (100, 5))
Y_data = rand(0:1, 100)
size(X_data), size(Y_data)
end

# ╔═╡ ad8d7761-f843-46f4-a7c6-4a5442d1e17a
function sigmoid(Z::Number)::Float32
    k = 1 + MathConstants.e ^ (-Float32(Z))
    return 1 / k
end

# ╔═╡ d457cb21-4e23-4527-a2e4-9654579f714a
typeof(X_data)

# ╔═╡ fcf0ea6c-72c4-40b9-bca9-d1ad40b67dd2
function scale(X::Array)::Array
    @assert ndims(X) == 2
    u = mean(X, dims=1) # compute mean
    s = std(X, dims=1)  # compute standard deviation
    res = (X .- u) ./ s
    return res
end

# ╔═╡ c7a7f2af-fa56-4bbf-afa1-a64b1af5b78d
X_data_scaled = scale(X_data);

# ╔═╡ 1a17435a-484c-4ea8-809b-1d3b0a548288
mean(X_data_scaled, dims=1)

# ╔═╡ 5a87d92e-643f-4032-a6fb-226ce7cab2af
std(X_data_scaled, dims=1)

# ╔═╡ 0890963a-806a-456a-89cf-ad68b11b41c3
# update sigmoid function
function sigmoid(Z::Array)::Array
    denom = 1 .+ (MathConstants.e .^ (-Z))
    return 1 ./ denom
end

# ╔═╡ bf7c00f8-1a1b-4010-b2a9-c02f33351bd5
# test sigmoid function
sigmoid(1234)

# ╔═╡ 4c1f5e23-a985-4f8f-81f5-6b780412a4ad
sigmoid(0)

# ╔═╡ 7fa3c5e2-8431-49d7-94c2-d3885a6a79f6
sigmoid(0.123)

# ╔═╡ ed47d544-9ccd-47ff-bf2a-01a9b2b365b0
# test it
sigmoid(X_data_scaled)

# ╔═╡ 8eaa7843-aa2b-4af3-9e32-629629d0fc36
begin
beta = randn(size(X_data)[2]+1)
size(beta)
end

# ╔═╡ e4d93713-c75d-4cbf-a31c-a9194f25280d
function cost(X::Array, y::Array, beta::Array)::AbstractFloat
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

# ╔═╡ 4ae75dea-6fc1-4d10-8c63-176a01193caf
# try it
cost(X_data_scaled, Y_data, beta)

# ╔═╡ 83a969a4-2704-421e-ba02-70875e5b22fc
# prediction function
function predict_proba(X::Array, beta::Array)::Array
    @assert ndims(X) == 2
    @assert ndims(beta) == 1
    @assert size(X)[2] == size(beta)[1]-1
    X_extended = hcat(X, ones(size(X)[1]))
    X_combined = X_extended * reshape(beta, (size(beta)[1], 1))
    prob = sigmoid(X_combined)
    prob = [(prob...)...] # flatten to 1d array
    return prob
end

# ╔═╡ a04e19f3-8c20-47b4-b30a-ccc0696bf54b
predict_proba(X_data_scaled, beta)

# ╔═╡ 7d94d627-941c-41fb-8fce-cc0da6972f74
function predict(X::Array, beta::Array)::Array
    @assert ndims(X) == 2
    @assert ndims(beta) == 1
    @assert size(X)[2] == size(beta)[1]-1
    X_extended = hcat(X, ones(size(X)[1]))
    X_combined = X_extended * reshape(beta, (size(beta)[1], 1))
    prob = sigmoid(X_combined)
    prob = [(prob...)...] # flatten to 1d array
    prob = map(m -> m >= 0.5 ? 1 : 0, prob)
    return prob
end

# ╔═╡ 6418946f-eff9-43fd-8654-415fc41579b5
predict(X_data_scaled, beta)

# ╔═╡ 4c6ee096-98f5-48b8-9977-f2485365759c
# inplace learning function (Gradient Descent)
function learn!(X::Array, y::Array, beta::Array, alpha::AbstractFloat)
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
    beta .= beta .- [(gradients...)...]
    return nothing
end

# ╔═╡ addad88e-317d-4086-a5e4-a9682f80f54a
beta

# ╔═╡ 2c3b2fed-e9aa-44b5-a12d-40281ae487d9
learn!(X_data_scaled, Y_data, beta, 0.01)

# ╔═╡ 63ab48c3-8c92-4935-a4ce-36a7049e5c86
beta

# ╔═╡ c0f88234-87ec-41ca-8a90-df7d89997090
# define the logistic regression function
function train(X::Array, y::Array; learning_rate::AbstractFloat=0.01, max_iter::Integer=10, return_all::Bool=false)::Array
    @assert ndims(X) == 2
    @assert ndims(y) == 1
    @assert size(X)[1] == size(y)[1]
    @assert max_iter >= 0
    beta = Random.randn(size(X)[2]+1)
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

# ╔═╡ addd77a3-6d60-4387-8bad-65c55fc0171e
# accuracy function
function accuracy(y_pred::Array, y_real::Array)::AbstractFloat
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

# ╔═╡ eb2362a6-8d78-4f38-8b53-a1b9a1f54820
accuracy(predict(X_data_scaled, beta), Y_data)

# ╔═╡ acfefff5-53c3-421c-84c8-441007cbba2e
accuracy(predict(X_data_scaled, train(X_data_scaled, Y_data, max_iter=100, learning_rate=0.5)), Y_data)

# ╔═╡ 8d99210c-eabe-4fe7-a529-4e83be5e6cf8
md"""
Since it is linear model, we should not expect it to have a high accuracy on a completely randomly generated dataset\
But we can see the improvements
"""

# ╔═╡ Cell order:
# ╟─cba2ccb0-c5a2-11eb-11be-41eb0278cb72
# ╠═2fdecb39-bff4-4f3c-b3f5-35438541a29d
# ╠═3761501d-f02e-41cb-9003-74cb68283d74
# ╠═ad8d7761-f843-46f4-a7c6-4a5442d1e17a
# ╠═bf7c00f8-1a1b-4010-b2a9-c02f33351bd5
# ╠═4c1f5e23-a985-4f8f-81f5-6b780412a4ad
# ╠═7fa3c5e2-8431-49d7-94c2-d3885a6a79f6
# ╠═d457cb21-4e23-4527-a2e4-9654579f714a
# ╠═29802e3a-8dfe-41c7-a25e-2f0961949854
# ╠═fcf0ea6c-72c4-40b9-bca9-d1ad40b67dd2
# ╠═c7a7f2af-fa56-4bbf-afa1-a64b1af5b78d
# ╠═1a17435a-484c-4ea8-809b-1d3b0a548288
# ╠═5a87d92e-643f-4032-a6fb-226ce7cab2af
# ╠═0890963a-806a-456a-89cf-ad68b11b41c3
# ╠═ed47d544-9ccd-47ff-bf2a-01a9b2b365b0
# ╠═8eaa7843-aa2b-4af3-9e32-629629d0fc36
# ╠═e4d93713-c75d-4cbf-a31c-a9194f25280d
# ╠═4ae75dea-6fc1-4d10-8c63-176a01193caf
# ╠═83a969a4-2704-421e-ba02-70875e5b22fc
# ╠═a04e19f3-8c20-47b4-b30a-ccc0696bf54b
# ╠═7d94d627-941c-41fb-8fce-cc0da6972f74
# ╠═6418946f-eff9-43fd-8654-415fc41579b5
# ╠═4c6ee096-98f5-48b8-9977-f2485365759c
# ╠═addad88e-317d-4086-a5e4-a9682f80f54a
# ╠═2c3b2fed-e9aa-44b5-a12d-40281ae487d9
# ╠═63ab48c3-8c92-4935-a4ce-36a7049e5c86
# ╠═c0f88234-87ec-41ca-8a90-df7d89997090
# ╠═addd77a3-6d60-4387-8bad-65c55fc0171e
# ╠═eb2362a6-8d78-4f38-8b53-a1b9a1f54820
# ╠═acfefff5-53c3-421c-84c8-441007cbba2e
# ╟─8d99210c-eabe-4fe7-a529-4e83be5e6cf8
