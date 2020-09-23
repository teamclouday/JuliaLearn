# K Nearest Neighbor Module
module KNN
export cosine_sim, dist_sim, majority_vote, predict_naive
export KdTree, create_kdtree, predict_kdtree

include("../tools.jl")
import .JuTools

import Statistics
import Random
import LinearAlgebra

"""
Cosine similarity between 2 vectors\\
`X1` and `X2` should have same shape\\
Returns a float for similarity, in range [0, 1]\\
The nearer to 1, the more similar 2 vectors are
"""
function cosine_sim(X1::Array{T} where T<:Number, X2::Array{T} where T<:Number)::AbstractFloat
    @assert size(X1) == size(X2)
    @assert ndims(X1) == ndims(X2) == 1
    product = LinearAlgebra.dot(X1, X2)
    X1_norm = LinearAlgebra.norm(X1, 2)
    X2_norm = LinearAlgebra.norm(X2, 2)
    return product / (X1_norm * X2_norm)
end

"""
Euclidean distance similarity between 2 vectors\\
`X1` and `X2` should have same shape\\
Returns a float for similarity, in range [0, inf)\\
The nearer to 0, the more similar 2 vectors are
"""
function dist_sim(X1::Array{T} where T<:Number, X2::Array{T} where T<:Number)::AbstractFloat
    @assert size(X1) == size(X2)
    @assert ndims(X1) == ndims(X2) == 1
    return sqrt(sum((X1 .- X2).^2))
end

"""
Majority vote implementation\\
`y` is 1d array\\
Returns the majority number found in `y`
"""
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

"""
Predict function for K Nearest Neighbor (Naive Approach)\\
`X_predict` is the data to predict\\
`X_data` and `Y_data` are the data to learn from\\
Returns an array of predictions
"""
function predict_naive(X_predict::Array{T} where T<:Number, X_data::Array{T} where T<:Number, Y_data::Array{T} where T<:Number; K::Integer=5)::Array
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

"""
The KdTree structure for `predict_kdtree`
"""
mutable struct KdTree
    X_data::Array{T} where T<:Number # 1d vector
    Y_data::Number                   # 1d vector
    child_l::Union{KdTree,Nothing}
    child_r::Union{KdTree,Nothing}
end

"""
Create a `KdTree` given `X_data` and `Y_data`\\
`X_data` has 2 dims, `Y_data` has 1 dims
"""
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

"""
Predict function for K Nearest Neighbor (K-d Tree Approach)\\
`X_predict` is the data to predict\\
`kdtree` is a KdTree created from training data\\
Returns an array of predictions
"""
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
                    kdtree_closest[i] = kdtree
                    kdtree_closest_val[i] = distance
                    break
                end
            end
        else
            curr_max = kdtree_closest_max(kdtree_closest, kdtree_closest_val)
            if distance < curr_max[2]
                kdtree_closest[curr_max[1]] = kdtree
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
            if (X_vec[curr_axis] + kdtree_closest_max(kdtree_closest, kdtree_closest_val)[2] > kdtree.X_data[curr_axis]) && kdtree.child_r !== nothing
                kdtree_recursive_search!(X_vec, kdtree.child_r, depth+1, n_axes, kdtree_closest, kdtree_closest_val)
            end
        else
            if kdtree.child_r !== nothing
                kdtree_recursive_search!(X_vec, kdtree.child_r, depth+1, n_axes, kdtree_closest, kdtree_closest_val)
            end
            if (X_vec[curr_axis] - kdtree_closest_max(kdtree_closest, kdtree_closest_val)[2] < kdtree.X_data[curr_axis]) && kdtree.child_l !== nothing
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

end