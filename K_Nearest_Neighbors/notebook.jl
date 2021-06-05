### A Pluto.jl notebook ###
# v0.14.7

using Markdown
using InteractiveUtils

# ╔═╡ 5057c590-c5a4-11eb-2621-4b67f12ca95d
md"""
# K Nearest Neighbors
"""

# ╔═╡ 74f4afa1-dd5a-4931-ab48-a1d4fc77bbe5
md"""
[Reference 1](https://scikit-learn.org/stable/modules/neighbors.html)\
[Reference 2](https://en.wikipedia.org/wiki/Nearest_neighbor_search)\
[Reference 3](https://booking.ai/k-nearest-neighbours-from-slow-to-fast-thanks-to-maths-bec682357ccd)

### Naive Approach

No transforming on original dataset (No training)\
For prediction, iter through the original dataset, and find the nearest K data, by a given metrics


Can use euclidean distance:

$$\text{dist}(X_1, X_2) = \|X_1 - X_2\|$$

A better approach is cosine similarity:

$$\text{sim}(X_1, X_2) = \frac{X_1 \cdot X_2}{\|X_1\| \|X_2\|}$$

which computes the cos value between two vectors\
1 for 0 degree, and less than 1 for $(0, \pi]$
"""

# ╔═╡ cb1dd9ac-ca78-4169-b073-943f64160dbc
module tools include("../tools.jl") end

# ╔═╡ f1fa76da-f656-4f12-97d7-a8b847ca0733
JuTools = tools.JuTools

# ╔═╡ c3ad6021-8080-4870-8350-1337d548f23a
import Statistics

# ╔═╡ 83864e6e-4b62-45d5-962e-39e36bbcc43d
import Random

# ╔═╡ 0d7f31a9-7ae2-4be5-81b2-313ac58af62d
import LinearAlgebra

# ╔═╡ 73b4e7e0-7294-46dc-9545-9728277df652
function cosine_sim(X1::Array{T} where T<:Number, X2::Array{T} where T<:Number)::AbstractFloat
    @assert size(X1) == size(X2)
    @assert ndims(X1) == ndims(X2) == 1
    product = LinearAlgebra.dot(X1, X2)
    X1_norm = LinearAlgebra.norm(X1, 2)
    X2_norm = LinearAlgebra.norm(X2, 2)
    return product / (X1_norm * X2_norm)
end

# ╔═╡ 19b1c000-6fa3-42b7-9fc7-983c7caf44b9
begin
X_data, Y_data = JuTools.data_generate_linear_2d()
size(X_data), size(Y_data)
end

# ╔═╡ 063389c7-07b6-4050-9241-ea953c4f4338
X_data[1:2, :]

# ╔═╡ cf291226-c0a1-4206-a8cb-53bedb67b139
cosine_sim(X_data[1, :], X_data[2, :])

# ╔═╡ fbf7a012-dcf8-498f-ad29-ec8c4854406c
cosine_sim(X_data[1, :], X_data[3, :])

# ╔═╡ 2cf4e990-f788-44e3-94cc-9e210e47c762
# define majority vote function
function majority_vote(y::Array{T} where T<:Number)::Number
    @assert ndims(y) == 1
    unique_votes = Dict{Number, Integer}()
    for y_val in y
        if !haskey(unique_votes, y_val)
            push!(unique_votes, y_val => 1)
        else
            unique_votes[y_val] += 1
        end
    end
    result = sort(collect(unique_votes), by=m->m[2])
    return result[end][1]
end

# ╔═╡ c9d27318-5e3d-4c9c-9ba5-556c649baaf9
majority_vote([1,1,0])

# ╔═╡ ba24b3c5-f350-4774-a018-9acf1f4597ad
md"""
Output ordering is affected by input ordering
"""

# ╔═╡ 5e564f3b-1bda-41b1-8b22-bdf6d91314d2
majority_vote([1,1,0,0])

# ╔═╡ ed575b52-7509-4140-b5e7-295920f73e3d
majority_vote([1,1,0,0,0])

# ╔═╡ 8406faec-66f4-46aa-9582-964ebdbecfba
begin
X_train, X_test, Y_train, Y_test = JuTools.split_data(X_data, Y_data, shuffle=true, ratio=0.3)
size(X_train), size(X_test), size(Y_train), size(Y_test)
end

# ╔═╡ 3001da4a-3288-4168-b514-c0b12f552fdd
# define predict function, naive approach
function predict_naive(X_predict::Array{T} where T<:Number, K::Integer, X_data::Array{T} where T<:Number, Y_data::Array{T} where T<:Number)::Array
    @assert ndims(X_data) == 2
    @assert ndims(Y_data) == 1
    @assert size(X_data)[1] == size(Y_data)[1]
    @assert 0 < ndims(X_predict) <= 2
    @assert 0 < K < size(X_data)[1]
    if ndims(X_predict) < 2
        X_predict = reshape(X_predict, (1, size(X_predict)[1]))
    end
    @assert size(X_predict)[2] == size(X_data)[2]
    result = Array{Number}(undef, size(X_predict)[1])
    sim = Array{Tuple{Integer, AbstractFloat}}(undef, size(X_data)[1])
    for i in 1:size(X_predict)[1]
        vec_predict = X_predict[i, :]
        for j in 1:size(X_data)[1]
            vec_data = X_data[j, :]
            vec_similarity = cosine_sim(vec_predict, vec_data)
            sim[j] = (j, vec_similarity)
        end
        sort!(sim, by=m->m[2], rev=true)
        K_nearest_votes = Y_data[[m[1] for m in sim[1:K]]]
        result[i] = majority_vote(K_nearest_votes)
    end
    return result
end

# ╔═╡ 30320c6e-7ada-41b6-a938-f1fd19cc120a
Y_predict = predict_naive(X_test, 5, X_train, Y_train)

# ╔═╡ 27b26afb-a91c-47c4-94ef-1d64761f291d
JuTools.compute_accuracy(Y_predict, Y_test)

# ╔═╡ 0e1b5c16-f444-41dd-9e44-999969ba647c
# what about dist similarity?
function dist_sim(X1::Array{T} where T<:Number, X2::Array{T} where T<:Number)::AbstractFloat
    @assert size(X1) == size(X2)
    @assert ndims(X1) == ndims(X2) == 1
    return sqrt(sum((X1 .- X2).^2))
end

# ╔═╡ c42f2b8a-a88d-4e8c-a1c0-1a371ae6a669
dist_sim(X_data[1, :], X_data[2, :])

# ╔═╡ 158f77bf-2eb8-490a-a062-6364a6ec70e9
dist_sim(X_data[2, :], X_data[3, :])

# ╔═╡ 323e10a0-c68f-433a-afc5-0ac2514a90d1
md"""
It's greatly affected by the scale of data!
"""

# ╔═╡ 52010e3f-2e4f-4778-b53e-4fad04dd75bf
function predict_naive_fun(X_predict::Array{T} where T<:Number, K::Integer, X_data::Array{T} where T<:Number, Y_data::Array{T} where T<:Number)::Array
    @assert ndims(X_data) == 2
    @assert ndims(Y_data) == 1
    @assert size(X_data)[1] == size(Y_data)[1]
    @assert 0 < ndims(X_predict) <= 2
    @assert 0 < K < size(X_data)[1]
    if ndims(X_predict) < 2
        X_predict = reshape(X_predict, (1, size(X_predict)[1]))
    end
    @assert size(X_predict)[2] == size(X_data)[2]
    result = Array{Number}(undef, size(X_predict)[1])
    sim = Array{Tuple{Integer, AbstractFloat}}(undef, size(X_data)[1])
    for i in 1:size(X_predict)[1]
        vec_predict = X_predict[i, :]
        for j in 1:size(X_data)[1]
            vec_data = X_data[j, :]
            vec_similarity = dist_sim(vec_predict, vec_data)
            sim[j] = (j, vec_similarity)
        end
        sort!(sim, by=m->m[2])
        K_nearest_votes = Y_data[[m[1] for m in sim[1:K]]]
        result[i] = majority_vote(K_nearest_votes)
    end
    return result
end

# ╔═╡ 36ceae2a-4c04-4f67-9a6d-19a44cb2fa5b
JuTools.compute_accuracy(predict_naive_fun(X_test, 5, X_train, Y_train), Y_test)

# ╔═╡ 2c7f7263-5ba0-4786-8b68-8ca8fb3ffc26
md"""
Although it (`predict_naive`) may be slow on large dataset, it is easy to implement and it works as expected
"""

# ╔═╡ b2aac2a9-cf86-44f0-ae97-b2a52327b0f9
md"""
### K-Dimensional Tree (K-d tree) Approach

A space partitioning technique\
Treat each data row as a point in `k`-dimensional space\
[Wikipedia](https://en.wikipedia.org/wiki/K-d_tree)
"""

# ╔═╡ b127297e-cb0c-46f5-8551-3ba2c4dae4c6
mutable struct KdTree
    X_data::Array{T} where T<:Number # 1d vector
    Y_data::Number                   # number
    child_l::Union{KdTree,Nothing}
    child_r::Union{KdTree,Nothing}
end

# ╔═╡ c3c4fe2c-6e92-4c98-bce3-e77f394c405d
# K-d tree generator function
function create_kdtree(X_data::Array{T} where T<:Number, Y_data::Array{T} where T<:Number)::KdTree
    @assert ndims(X_data) == 2
    @assert ndims(Y_data) == 1
    @assert size(X_data)[1] == size(Y_data)[1]
    function kdtree_recursive_generate(X_data::Array, Y_data::Array, depth::Integer, n_axes::Integer)::KdTree
        curr_axis = mod(depth, n_axes) + 1 # array starts from 1
        data_combined = hcat(X_data, Y_data)
        data_combined = sortslices(data_combined, by=m->m[curr_axis], dims=1)
        X_data = data_combined[:, 1:end-1]
        Y_data = data_combined[:, end]
        i_mid = div(size(X_data)[1], 2) + 1
        node_X_data = X_data[i_mid, :]
        node_Y_data = Y_data[i_mid]
        node = KdTree(node_X_data, node_Y_data, nothing, nothing)
        if i_mid > 1
            node.child_l = kdtree_recursive_generate(X_data[1:i_mid-1,:], Y_data[1:i_mid-1], depth+1, n_axes)
        end
        if i_mid < size(X_data)[1]
            node.child_r = kdtree_recursive_generate(X_data[i_mid+1:end,:], Y_data[i_mid+1:end], depth+1, n_axes)
        end
        return node
    end
    return kdtree_recursive_generate(X_data, Y_data, 0, size(X_data)[2])
end

# ╔═╡ a76dcf16-060e-4ece-9b8d-76c45deb38c4
kdtree = create_kdtree(X_data, Y_data)

# ╔═╡ 4c3444dd-f2cb-4954-8dad-d2e9466252e8
kdtree_test = create_kdtree(reshape([30,5,10,70,50,35], (6, 1)), [1,1,1,1,1,1])

# ╔═╡ 94c43601-cff7-4c7f-a116-8a82aefed5b0
# inspired from https://stackoverflow.com/questions/1627305/nearest-neighbor-k-d-tree-wikipedia-proof/37107030#37107030
# note that for kdtree search, we use euclidean distance
function predict_kdtree(X_predict::Array{T} where T<:Number, kdtree::KdTree; K::Integer=5)::Array
    @assert K > 0
    @assert 0 < ndims(X_predict) <= 2
    if ndims(X_predict) == 1
        X_predict = reshape(X_predict, (1, size(X_predict)[1]))
    end
    @assert size(X_predict)[2] == size(kdtree.X_data)[1]

    function kdtree_closest_max(kdtree_closest::Array{Union{KdTree, Nothing}},
                kdtree_closest_val::Array{AbstractFloat})::Tuple{Integer, AbstractFloat}
        default = (0, 0.0)
        for i in 1:size(kdtree_closest)[1]
            if kdtree_closest[i] === nothing
                break
            elseif default[1] == 0 || (kdtree_closest_val[i] > default[2])
                default = (i, kdtree_closest_val[i])
            end
        end
        return default
    end
    
    function kdtree_update_nearest!(X_vec::Array, kdtree::KdTree, kdtree_closest::Array{Union{KdTree, Nothing}},
                kdtree_closest_val::Array{AbstractFloat})
        @assert size(kdtree_closest) == size(kdtree_closest_val)
        distance = dist_sim(kdtree.X_data, X_vec)
        if nothing in kdtree_closest
            for i in 1:size(kdtree_closest)[1]
                if kdtree_closest[i] === nothing
                    kdtree_closest[i] = KdTree(kdtree.X_data, kdtree.Y_data, nothing, nothing)
                    kdtree_closest_val[i] = distance
                    break
                end
            end
        else
            curr_max = kdtree_closest_max(kdtree_closest, kdtree_closest_val)
            if distance < curr_max[2]
                kdtree_closest[curr_max[1]] = KdTree(kdtree.X_data, kdtree.Y_data, nothing, nothing)
                kdtree_closest_val[curr_max[1]] = distance
            end
        end
    end
    
    function kdtree_recursive_search!(X_vec::Array, kdtree::KdTree, depth::Integer, n_axes::Integer, 
                kdtree_closest::Array{Union{KdTree, Nothing}}, kdtree_closest_val::Array{AbstractFloat})
        @assert size(kdtree_closest) == size(kdtree_closest_val)
        # check current node
        kdtree_update_nearest!(X_vec, kdtree, kdtree_closest, kdtree_closest_val)
        # run on children
        curr_axis = mod(depth, n_axes) + 1 # array starts from 1
        if X_vec[curr_axis] < kdtree.X_data[curr_axis]
            if kdtree.child_l !== nothing
                kdtree_recursive_search!(X_vec, kdtree.child_l, depth+1, n_axes, kdtree_closest, kdtree_closest_val)
            end
            if (X_vec[curr_axis] + kdtree_closest_max(kdtree_closest, kdtree_closest_val)[2] >= kdtree.X_data[curr_axis]) && kdtree.child_r !== nothing
                kdtree_recursive_search!(X_vec, kdtree.child_r, depth+1, n_axes, kdtree_closest, kdtree_closest_val)
            end
        else
            if kdtree.child_r !== nothing
                kdtree_recursive_search!(X_vec, kdtree.child_r, depth+1, n_axes, kdtree_closest, kdtree_closest_val)
            end
            if (X_vec[curr_axis] - kdtree_closest_max(kdtree_closest, kdtree_closest_val)[2] <= kdtree.X_data[curr_axis]) && kdtree.child_l !== nothing
                kdtree_recursive_search!(X_vec, kdtree.child_l, depth+1, n_axes, kdtree_closest, kdtree_closest_val)
            end
        end
    end
    
    result = Array{Number}(undef, size(X_predict)[1])
    for i in 1:size(X_predict)[1]
        kdtree_closest = Array{Union{KdTree, Nothing}}(nothing, K)
        kdtree_closest_val = Array{AbstractFloat}(undef, K)
        kdtree_recursive_search!(X_predict[i, :], kdtree, 0, size(X_predict)[2], kdtree_closest, kdtree_closest_val)
        K_nearest_votes = Number[]
        for i in 1:K
            if kdtree_closest[i] === nothing
                break
            else
                push!(K_nearest_votes, kdtree_closest[i].Y_data)
            end
        end
        result[i] = majority_vote(K_nearest_votes)
    end
    return result
end

# ╔═╡ 569da607-dea8-4994-9ffe-05959071a855
kdtree_train = create_kdtree(X_train, Y_train)

# ╔═╡ 0ea4cb86-6be5-49f1-b9b8-56f6a8cc8ed0
JuTools.compute_accuracy(predict_kdtree(X_test, kdtree_train, K=10), Y_test)

# ╔═╡ 98be952f-0591-45c6-be83-3c0f6cc5d26b
md"""
### Ball Tree Approach

A better space partition approach\
More efficient than K-d Tree when searching\
[Wikipedia](https://en.wikipedia.org/wiki/Ball_tree)
"""

# ╔═╡ dc218b5e-73f5-4dc8-b523-62bed53a71e7
mutable struct BallTree
    X_data::Array{T} where T<:Number
    Y_data::Number
    pivot::Union{Array{T},Nothing} where T<:Number # defines pivot point of hypersphere
    radius::AbstractFloat                            # defines radius of hypersphere
    child_l::Union{BallTree,Nothing}
    child_r::Union{BallTree,Nothing}
end

# ╔═╡ d7cc4b1c-f953-4f64-95bb-09e1d51a38f5
# inspired from https://gist.github.com/jakevdp/5216193
# ball tree generator function
function create_balltree(X_data::Array{T} where T<:Number, Y_data::Array{T} where T<:Number)::BallTree
    @assert ndims(X_data) == 2
    @assert ndims(Y_data) == 1
    @assert size(X_data)[1] == size(Y_data)[1]
    balltree = nothing
    if size(X_data)[1] == 1
        balltree = BallTree(X_data[1, :], Y_data[1], nothing, 0.0, nothing, nothing)
    else
        # find pivot
        pivot = vec(sum(X_data, dims=1)) ./ size(X_data)[1]
        # find radius
        radius = 0.0
        for i in 1:size(X_data)[1]
            X_vec = X_data[i, :]
            dist = dist_sim(pivot, X_vec)
            if dist > radius
                radius = dist
            end
        end
        # find greatest spread dimension
        d_greatest_spread = 1
        n_spread = 0.0
        for i in 1:size(X_data)[2]
            X_vec = X_data[:, i]
            current_spread = abs(maximum(X_vec) - minimum(X_vec))
            if current_spread > n_spread
                d_greatest_spread = i
                n_spread = current_spread
            end
        end
        data_combined = hcat(X_data, Y_data)
        data_combined = sortslices(data_combined, by=m->m[d_greatest_spread], dims=1)
        X_data = data_combined[:, 1:end-1]
        Y_data = data_combined[:, end]
        i_mid = div(size(X_data)[1], 2) + 1
        node_X_data = X_data[i_mid, :]
        node_Y_data = Y_data[i_mid]
        balltree = BallTree(node_X_data, node_Y_data, pivot, radius, nothing, nothing)
        if i_mid > 1
            balltree.child_l = create_balltree(X_data[1:i_mid-1,:], Y_data[1:i_mid-1])
        end
        if i_mid < size(X_data)[1]
            balltree.child_r = create_balltree(X_data[i_mid+1:end,:], Y_data[i_mid+1:end])
        end
    end
    return balltree
end

# ╔═╡ ffb15055-c887-47a9-88d7-71a33a3c2dba
balltree = create_balltree(X_data, Y_data)

# ╔═╡ 77ed169b-d426-4f3f-8362-809cb0c8dd22
balltree.X_data

# ╔═╡ 79ddb157-8ee4-4c44-b8bf-b4f36f88acce
balltree.Y_data

# ╔═╡ 6c589940-e00a-40c9-b2d9-c10384e4ac62
balltree.pivot

# ╔═╡ 5b5c5f78-75c2-4a85-b918-e7485f492d9e
balltree.radius

# ╔═╡ 75368c67-8774-4f67-9163-d31475f4e289
balltree_test = create_balltree(reshape([30,5,10,70,50,35], (6, 1)), [1,1,1,1,1,1])

# ╔═╡ a14bd64d-4188-4682-8a47-c3999afb0d09
# search function for ball tree
# similar to kdtree search
function predict_balltree(X_predict::Array{T} where T<:Number, balltree::BallTree; K::Integer=5)::Array
    @assert K > 0
    @assert 0 < ndims(X_predict) <= 2
    if ndims(X_predict) == 1
        X_predict = reshape(X_predict, (1, size(X_predict)[1]))
    end
    @assert size(X_predict)[2] == size(balltree.X_data)[1]
    
    function balltree_closest_max(balltree_closest::Array{Union{BallTree, Nothing}},
                balltree_closest_val::Array{AbstractFloat})::Tuple{Integer, AbstractFloat}
        # find the maximum value in balltree_closest_val
        # return its index and value
        default = (0, 0.0)
        for i in 1:size(balltree_closest)[1]
            if balltree_closest[i] === nothing
                break
            elseif default[1] == 0 || (balltree_closest_val[i] > default[2])
                default = (i, balltree_closest_val[i])
            end
        end
        return default
    end
    
    function balltree_update_nearest!(X_vec::Array, balltree::BallTree, balltree_closest::Array{Union{BallTree, Nothing}},
                balltree_closest_val::Array{AbstractFloat})
        # update current node by distance
        @assert size(balltree_closest) == size(balltree_closest_val)
        distance = dist_sim(balltree.X_data, X_vec)
        if nothing in balltree_closest
            for i in 1:size(balltree_closest)[1]
                if balltree_closest[i] === nothing
                    balltree_closest[i] = balltree
                    balltree_closest_val[i] = distance
                    break
                end
            end
        else
            curr_max = balltree_closest_max(balltree_closest, balltree_closest_val)
            if distance < curr_max[2]
                balltree_closest[curr_max[1]] = balltree
                balltree_closest_val[curr_max[1]] = distance
            end
        end
    end
    
    function balltree_recursive_search!(X_vec::Array, balltree::BallTree, balltree_closest::Array{Union{BallTree, Nothing}},
            balltree_closest_val::Array{AbstractFloat})
        # recursively search a balltree for K nearest neighbors
        @assert size(balltree_closest) == size(balltree_closest_val)
        if (!(nothing in balltree_closest) && (balltree.pivot !== nothing)
                && (dist_sim(X_vec, balltree.pivot) - balltree.radius >= 
                    balltree_closest_max(balltree_closest, balltree_closest_val)[2]))
            return nothing
        end
        # check current node
        balltree_update_nearest!(X_vec, balltree, balltree_closest, balltree_closest_val)
        # run on children
        if balltree.child_l === nothing || balltree.child_r === nothing
            if balltree.child_l !== nothing
                balltree_recursive_search!(X_vec, balltree.child_l, balltree_closest, balltree_closest_val)
            end
            if balltree.child_r !== nothing
                balltree_recursive_search!(X_vec, balltree.child_r, balltree_closest, balltree_closest_val)
            end
        else
            dist_left = dist_sim(X_vec, balltree.child_l.X_data)
            dist_right = dist_sim(X_vec, balltree.child_r.X_data)
            if dist_left < dist_right
                balltree_recursive_search!(X_vec, balltree.child_l, balltree_closest, balltree_closest_val)
                balltree_recursive_search!(X_vec, balltree.child_r, balltree_closest, balltree_closest_val)
            else
                balltree_recursive_search!(X_vec, balltree.child_r, balltree_closest, balltree_closest_val)
                balltree_recursive_search!(X_vec, balltree.child_l, balltree_closest, balltree_closest_val)
            end
        end
    end
    
    result = Array{Number}(undef, size(X_predict)[1])
    for i in 1:size(X_predict)[1]
        balltree_closest = Array{Union{BallTree, Nothing}}(nothing, K)
        balltree_closest_val = Array{AbstractFloat}(undef, K)
        balltree_recursive_search!(X_predict[i, :], balltree, balltree_closest, balltree_closest_val)
        K_nearest_votes = Number[]
        for i in 1:K
            if balltree_closest[i] === nothing
                break
            else
                push!(K_nearest_votes, balltree_closest[i].Y_data)
            end
        end
        result[i] = majority_vote(K_nearest_votes)
    end
    return result
end

# ╔═╡ 06e6d38d-b08e-4583-863a-e6e06726143b
balltree_train = create_balltree(X_train, Y_train)

# ╔═╡ 41e118e2-2172-4910-ba52-12e199737c6a
JuTools.compute_accuracy(predict_balltree(X_test, balltree_train, K=20), Y_test)

# ╔═╡ Cell order:
# ╟─5057c590-c5a4-11eb-2621-4b67f12ca95d
# ╟─74f4afa1-dd5a-4931-ab48-a1d4fc77bbe5
# ╠═cb1dd9ac-ca78-4169-b073-943f64160dbc
# ╠═f1fa76da-f656-4f12-97d7-a8b847ca0733
# ╠═c3ad6021-8080-4870-8350-1337d548f23a
# ╠═83864e6e-4b62-45d5-962e-39e36bbcc43d
# ╠═0d7f31a9-7ae2-4be5-81b2-313ac58af62d
# ╠═73b4e7e0-7294-46dc-9545-9728277df652
# ╠═19b1c000-6fa3-42b7-9fc7-983c7caf44b9
# ╠═063389c7-07b6-4050-9241-ea953c4f4338
# ╠═cf291226-c0a1-4206-a8cb-53bedb67b139
# ╠═fbf7a012-dcf8-498f-ad29-ec8c4854406c
# ╠═2cf4e990-f788-44e3-94cc-9e210e47c762
# ╠═c9d27318-5e3d-4c9c-9ba5-556c649baaf9
# ╟─ba24b3c5-f350-4774-a018-9acf1f4597ad
# ╠═5e564f3b-1bda-41b1-8b22-bdf6d91314d2
# ╠═ed575b52-7509-4140-b5e7-295920f73e3d
# ╠═8406faec-66f4-46aa-9582-964ebdbecfba
# ╠═3001da4a-3288-4168-b514-c0b12f552fdd
# ╠═30320c6e-7ada-41b6-a938-f1fd19cc120a
# ╠═27b26afb-a91c-47c4-94ef-1d64761f291d
# ╠═0e1b5c16-f444-41dd-9e44-999969ba647c
# ╠═c42f2b8a-a88d-4e8c-a1c0-1a371ae6a669
# ╠═158f77bf-2eb8-490a-a062-6364a6ec70e9
# ╟─323e10a0-c68f-433a-afc5-0ac2514a90d1
# ╠═52010e3f-2e4f-4778-b53e-4fad04dd75bf
# ╠═36ceae2a-4c04-4f67-9a6d-19a44cb2fa5b
# ╟─2c7f7263-5ba0-4786-8b68-8ca8fb3ffc26
# ╟─b2aac2a9-cf86-44f0-ae97-b2a52327b0f9
# ╠═b127297e-cb0c-46f5-8551-3ba2c4dae4c6
# ╠═c3c4fe2c-6e92-4c98-bce3-e77f394c405d
# ╠═a76dcf16-060e-4ece-9b8d-76c45deb38c4
# ╠═4c3444dd-f2cb-4954-8dad-d2e9466252e8
# ╠═94c43601-cff7-4c7f-a116-8a82aefed5b0
# ╠═569da607-dea8-4994-9ffe-05959071a855
# ╠═0ea4cb86-6be5-49f1-b9b8-56f6a8cc8ed0
# ╟─98be952f-0591-45c6-be83-3c0f6cc5d26b
# ╠═dc218b5e-73f5-4dc8-b523-62bed53a71e7
# ╠═d7cc4b1c-f953-4f64-95bb-09e1d51a38f5
# ╠═ffb15055-c887-47a9-88d7-71a33a3c2dba
# ╠═77ed169b-d426-4f3f-8362-809cb0c8dd22
# ╠═79ddb157-8ee4-4c44-b8bf-b4f36f88acce
# ╠═6c589940-e00a-40c9-b2d9-c10384e4ac62
# ╠═5b5c5f78-75c2-4a85-b918-e7485f492d9e
# ╠═75368c67-8774-4f67-9163-d31475f4e289
# ╠═a14bd64d-4188-4682-8a47-c3999afb0d09
# ╠═06e6d38d-b08e-4583-863a-e6e06726143b
# ╠═41e118e2-2172-4910-ba52-12e199737c6a
