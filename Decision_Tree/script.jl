# Decision Tree Module

module Tree
export DecisionTree, create_decision_tree, predict

"""
Data structure for Decision Tree\\
One of `children` and `target` will be `nothing`
"""
mutable struct DecisionTree
    col_id::Integer
    children::Union{AbstractDict{Function,DecisionTree},Nothing}
    target::Union{AbstractFloat,Nothing}
end

"""
C4.5 implementation for Decision Tree\\
`X_data` has 2 dims, `Y_data` has 1 dims\\
Returns a computed `DecisionTree` structure
"""
function create_decision_tree(X_data::Array{T} where T<:Number, Y_data::Array{T} where T<:Number; max_depth::Integer=10)::DecisionTree
    @assert ndims(X_data) == ndims(Y_data) + 1 == 2
    @assert size(X_data)[1] == size(Y_data)[1]
    @assert max_depth >= 1
    
    """
    Output the most frequent value in the vector
    """
    function majority(Y_vec::Array)::Number
        unique_vals = Dict{Number, Integer}()
        for Y_val in Y_vec
            if !haskey(unique_vals, Y_val)
                unique_vals[Y_val] = 1
            else
                unique_vals[Y_val] += 1
            end
        end
        result = sort(collect(unique_vals), by=m->m[2])
        return result[end][1]
    end
    
    """
    Check if vector has same value
    """
    function is_same_class(Y_vec::Array)::Bool
        val = Y_vec[1]
        for m in Y_vec[2:end]
            if m != val
                return false
            end
        end
        return true
    end
    
    """
    Get an array of unique values from a vector
    """
    function get_unique_array(X_vec::Array)::Array
        X_max = maximum(X_vec)
        X_min = minimum(X_vec)
        round_digits = -Integer(trunc(log10(X_max-X_min)-2))
        result = []
        for m in X_vec
            if !(trunc(m, digits=round_digits) in result)
                push!(result, m)
            end
        end
        return result
    end
    
    """
    The entropy function
    """
    function info(Y_data::Array)::AbstractFloat
        portion = Dict{Number,Integer}()
        for m in Y_data
            if !haskey(portion, m)
                portion[m] = 1
            else
                portion[m] += 1
            end
        end
        T = size(Y_data)[1]
        result = 0.0
        for m in keys(portion)
            p = portion[m] / T
            result += -p * log2(p)
        end
        return result
    end

    """
    Detect types for each column of `X_data`\\
    `true` if continuous\\
    `false` if categorical
    """
    function detect_types(X_data::Array)::Array
        result = Array{Bool}(undef, size(X_data)[2])
        for i in 1:size(X_data)[2]
            X_unique = get_unique_array(X_data[:, i])
            if length(X_unique) > 10
                result[i] = true
            else
                result[i] = false
            end
        end
        return result
    end
    
    function create_decision_tree_recursive(X_data::Array, Y_data::Array, X_type::Array, max_depth::Integer)::DecisionTree
        if is_same_class(Y_data)
            return DecisionTree(-1, nothing, Y_data[1])
        end
        if max_depth <= 0
            return DecisionTree(-1, nothing, majority(Y_data))
        end
        max_index = -1
        max_ratio = 0.0
        max_fn = nothing
        for i in 1:size(X_data)[2]
            X_vec_unique = get_unique_array(X_data[:, i])
            if X_type[i]
                # continous data, then choose binary threshold
                fn_max = nothing
                ratio_max = 0
                for j in 1:(length(X_vec_unique)-1)
                    # choose average as threshold
                    threshold = (X_vec_unique[j] + X_vec_unique[j+1]) / 2.0
                    fn = (m) -> m <= threshold
                    # compute information gain
                    Ii = info(Y_data)
                    Y_1 = Y_data[fn.(X_data[:, i])]
                    Y_2 = Y_data[(!fn).(X_data[:, i])]
                    I1 = info(Y_1)
                    I2 = info(Y_2)
                    gain = Ii - (length(Y_1)/length(Y_data)*I1 + length(Y_2)/length(Y_data)*I2)
                    splitInfo = -(length(Y_1)/length(Y_data))*log2(length(Y_1)/length(Y_data)) - (length(Y_2)/length(Y_data))*log2(length(Y_2)/length(Y_data))
                    ratio = gain / splitInfo
                    if fn_max === nothing
                        fn_max = fn
                        ratio_max = ratio
                    else
                        if ratio > ratio_max
                            fn_max = fn
                            ratio_max = ratio
                        end
                    end
                end
                if ratio_max > max_ratio
                    max_index = i
                    max_fn = fn_max
                    max_ratio = ratio_max
                end
            else
                # skip visited categorical feature (homogeneous)
                if length(X_vec_unique) <= 1
                    continue
                end
                # categorical data, then compute by splitting on each category
                Ii = info(Y_data)
                gain = Ii
                splitInfo = 0.0
                for val in X_vec_unique
                    fn = (m) -> m == val
                    Y0 = Y_data[fn.(X_data[:, i])]
                    I0 = info(Y0)
                    gain -= (length(Y0)/length(Y_data))*I0
                    splitInfo -= (length(Y0)/length(Y_data))*log2(length(Y0)/length(Y_data))
                end
                ratio = gain / splitInfo
                if ratio > max_ratio
                    max_index = i
                    max_fn = nothing
                    max_ratio = ratio
                end
            end
        end
        if max_index < 0
            return DecisionTree(-1, nothing, majority(Y_data))
        end
        children = Dict{Function,DecisionTree}()
        if max_fn === nothing
            # categorical
            X_vec_unique = get_unique_array(X_data[:, max_index])
            for val in X_vec_unique
                fn = (m) -> m == val
                identify = fn.(X_data[:, max_index])
                X_part = X_data[identify, :]
                Y_part = Y_data[identify]
                children[fn] = create_decision_tree_recursive(X_part, Y_part, X_type, max_depth-1)
            end
        else
            # continuous
            identify_1 = max_fn.(X_data[:, max_index])
            X_part_1 = X_data[identify_1, :]
            Y_part_1 = Y_data[identify_1]
            children[max_fn] = create_decision_tree_recursive(X_part_1, Y_part_1, X_type, max_depth-1)
            identify_2 = (!max_fn).(X_data[:, max_index])
            X_part_2 = X_data[identify_2, :]
            Y_part_2 = Y_data[identify_2]
            children[(!max_fn)] = create_decision_tree_recursive(X_part_2, Y_part_2, X_type, max_depth-1)
        end
        return DecisionTree(max_index, children, nothing)
    end
    
    return create_decision_tree_recursive(X_data, Y_data, detect_types(X_data), max_depth)
end

"""
Prediction function for Decision Tree\\
`X_data` has ndims of 1 or 2\\
Returns an array of predictions\\
Print log when error occurs (undecided children tree)
"""
function predict(X_data::Array{T} where T<:Number, tree::DecisionTree)::Array
    if ndims(X_data) == 1
        X_data = reshape(X_data, (1, size(X_data)[1]))
    end
    @assert ndims(X_data) == 2
    prediction = []
    for i in 1:size(X_data)[1]
        X_vec = X_data[i, :]
        Y_pred = nothing
        tree_copy = tree
        while true
            if tree_copy.target !== nothing
                Y_pred = tree_copy.target
                break
            end
            @assert tree_copy.children !== nothing
            updated = false
            for key_fn in keys(tree_copy.children)
                if key_fn(X_vec[tree_copy.col_id])
                    tree_copy = tree_copy.children[key_fn]
                    updated = true
                    break
                end
            end
            if !updated
                println("Error occured for $X_vec")
                break
            end
        end
        push!(prediction, Y_pred)
    end
    return prediction
end

end