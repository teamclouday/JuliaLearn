# K Nearest Neighbor Module
module KNN
export cosine_sim, dist_sim, majority_vote, predict_naive

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

end