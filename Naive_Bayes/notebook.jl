### A Pluto.jl notebook ###
# v0.14.7

using Markdown
using InteractiveUtils

# ╔═╡ d0c378ce-c5a1-11eb-0732-7bd1ff6303d2
md"""
# Naive Bayes

[Reference 1](https://en.wikipedia.org/wiki/Naive_Bayes_classifier)\
[Reference 2](https://scikit-learn.org/stable/modules/naive_bayes.html)
"""

# ╔═╡ c084dbc5-6f8f-47e2-9382-4df9234be5d2
md"""
Bayes' theorem:

$$P(y|x_1,...,x_n)=\frac{P(y)P(x_1,...,x_n|y)}{P(x_1,...,x_n)}$$


Classification:

$$\hat{y}=\text{arg}\max_yP(y)\prod^n_{i=1}P(x_i|y)$$


Gaussian Likelihood:

$$P(x_i|y)=\frac{1}{\sqrt{2\pi\sigma^2_{x_i,y}}}\exp\Bigg(-\frac{(x_i-\mu_{x_i,y})^2}{2\sigma^2_{x_i,y}}\Bigg)$$

where $\sigma_y$ and $\mu_y$ are estimated with maximum likelihood
"""

# ╔═╡ df15724c-422f-44d0-a921-1f0ef1c564fe
module tools include("../tools.jl") end

# ╔═╡ 92dca0e5-e95a-4060-bbd4-e464629473a9
JuTools = tools.JuTools

# ╔═╡ 7274c675-66c5-43a7-9c39-f7083de2619f
begin
X_data, Y_data = JuTools.data_generate_cluster_2d(pos1=(30.0, 80.0), pos2=(80.0, 30.0),
    radius1=5.0, radius2=10.0, random_scale=8.0, data_size=1000)
size(X_data), size(Y_data)
end

# ╔═╡ db6250c4-7e83-4a87-acf7-a2fc7051711f
mutable struct NaiveBayes
    n_features::Integer
    pY::Dict{Number,AbstractFloat}
    mean::Dict{Number,Array{AbstractFloat}}
    var::Dict{Number,Array{AbstractFloat}}
end

# ╔═╡ 2896ce68-e48e-4dc2-9317-18d6dfcd1365
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

# ╔═╡ 8935105e-72f1-46b6-8fe8-56f9afa6fa5f
function guassian(X_vec::Array, mean::Array, var::Array)::AbstractFloat
    @assert ndims(X_vec) == ndims(mean) == ndims(var) == 1
    @assert length(X_vec) == length(mean) == length(var)
    left = 1.0 ./ sqrt.((2.0 * pi) .* var)
    right = exp.(-(X_vec .- mean).^2 ./ (2.0 .* var))
    p = left .* right
    return prod(p)
end

# ╔═╡ 6eaa3999-30cd-4ab7-94cc-fab320f075d1
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

# ╔═╡ b820e684-4cde-4353-b539-744bfa2a375b
trained = train(X_data, Y_data);

# ╔═╡ 4cdc095c-0d9e-44d0-8708-b3a3c880d153
prediction = predict(X_data, trained)

# ╔═╡ 8a60639b-6db4-4353-b69e-8e72c6961982
JuTools.compute_accuracy(prediction, Y_data)

# ╔═╡ Cell order:
# ╟─d0c378ce-c5a1-11eb-0732-7bd1ff6303d2
# ╟─c084dbc5-6f8f-47e2-9382-4df9234be5d2
# ╠═df15724c-422f-44d0-a921-1f0ef1c564fe
# ╠═92dca0e5-e95a-4060-bbd4-e464629473a9
# ╠═7274c675-66c5-43a7-9c39-f7083de2619f
# ╠═db6250c4-7e83-4a87-acf7-a2fc7051711f
# ╠═2896ce68-e48e-4dc2-9317-18d6dfcd1365
# ╠═8935105e-72f1-46b6-8fe8-56f9afa6fa5f
# ╠═6eaa3999-30cd-4ab7-94cc-fab320f075d1
# ╠═b820e684-4cde-4353-b539-744bfa2a375b
# ╠═4cdc095c-0d9e-44d0-8708-b3a3c880d153
# ╠═8a60639b-6db4-4353-b69e-8e72c6961982
