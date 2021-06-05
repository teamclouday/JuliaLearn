### A Pluto.jl notebook ###
# v0.14.7

using Markdown
using InteractiveUtils

# ╔═╡ d0106330-c5a6-11eb-17b7-77a6ed6e1efe
md"""
# Decision Tree  

[Reference 1](https://en.wikipedia.org/wiki/Decision_tree)\
[Reference 2](https://scikit-learn.org/stable/modules/tree.html)
"""

# ╔═╡ 88897152-4b3d-4a62-bc67-32970342d24f
md"""
### C4.5 Approach

Algorithm described [here](https://en.wikipedia.org/wiki/C4.5_algorithm)\
My implementation is inspired by the description on [this page](https://cis.temple.edu/~giorgio/cis587/readings/id3-c45.html)


Use __Entropy__ to determine purity:

$$\text{Entropy}(S)=\sum^c_{i=1}-p_i\log_2(p_i)$$

For a given segment of data $S$, $c$ is the number of different class levels, and $p_i$ is the proportion of values falling into class level $i$\
Entropy of 0 means that data is homogenous, and 1 indicates the maximum amount of disorder


Then compute __information gain__ for each feature to decide which to split on:

$$\text{InfoGain}(F)=\text{Entropy}(S_1)-\text{Entropy}(S_2)$$

where $F$ is a feature, $S_1$ is the segment before split, and $S_2$ is the partitions after split\
Entropy on $S_2$ is the weighed sum of entropy on all its partitions ($P_i$), $w_i$ is the proportion of samples falled into $P_i$

$$\text{Entropy}(S_2)=\sum^n_{i=1}w_i\text{Entropy}(P_i)$$

The larger the information gain, the better a feature is splitted to be homogenous


An improvement on information gain is __gain ratios__:

$$\text{GainRatio}=\frac{\text{InfoGain}(F)}{\text{SplitInfo}(F)}$$

where $\text{SplitInfo}$ is the entropy due to the split of $F$

$$\text{SplitInfo}=\sum^c_{j=1}-P_j\log_2(P_j)$$

where each $P_j$ is the proportion of values after partitioning $F$, without considering target classes like entropy function


__Pre-pruning__ (early stopping): As the tree will grow large by dividing features and overfit, we can stop the growing tree after it reaches certain amount of decisions


__Post-pruning__: After the tree grows to its maximum size, cut out nodes and branches that have little effect on accuracy
"""

# ╔═╡ 43e44b5a-62d7-4f95-8685-62470ee55453
module tools include("../tools.jl") end

# ╔═╡ 2a6db899-de2d-497f-8290-f805d59deec6
JuTools = tools.JuTools

# ╔═╡ d5361f35-8091-4f26-83bd-40bd3cd033d3
begin
X_data, Y_data = JuTools.data_generate_cluster_2d(pos1=(30.0, 80.0), pos2=(80.0, 30.0),
    radius1=5.0, radius2=10.0, random_scale=8.0, data_size=1000)
size(X_data), size(Y_data)
end

# ╔═╡ d76345d3-759e-4cde-a371-7c59fdee0b3e
mutable struct DecisionTree
    col_id::Integer
    children::Union{AbstractDict{Function,DecisionTree},Nothing}
    target::Union{AbstractFloat,Nothing}
end

# ╔═╡ a5c720d7-aacc-4e91-a045-37aecfcef305
function create_decision_tree(X_data::Array{T} where T<:Number, Y_data::Array{T} where T<:Number; max_depth::Integer=10)::DecisionTree
    @assert ndims(X_data) == ndims(Y_data) + 1 == 2
    @assert size(X_data)[1] == size(Y_data)[1]
    @assert max_depth >= 1
    
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
    
    function is_same_class(Y_vec::Array)::Bool
        val = Y_vec[1]
        for m in Y_vec[2:end]
            if m != val
                return false
            end
        end
        return true
    end
    
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
    
    function create_decision_tree_recursive(X_data::Array, Y_data::Array, X_index_visited::Array, max_depth::Integer)::DecisionTree
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
            if i in X_index_visited
                continue
            end
            X_vec_unique = get_unique_array(X_data[:, i])
            if length(X_vec_unique) > 10
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
        X_index_visited = copy(X_index_visited)
        push!(X_index_visited, max_index)
        children = Dict{Function,DecisionTree}()
        if max_fn === nothing
            # categorical
            X_vec_unique = get_unique_array(X_data[:, max_index])
            for val in X_vec_unique
                fn = (m) -> m == val
                identify = fn.(X_data[:, max_index])
                X_part = X_data[identify, :]
                Y_part = Y_data[identity]
                children[fn] = create_decision_tree_recursive(X_part, Y_part, X_index_visited, max_depth-1)
            end
        else
            # continuous
            identify_1 = max_fn.(X_data[:, max_index])
            X_part_1 = X_data[identify_1, :]
            Y_part_1 = Y_data[identify_1]
            children[max_fn] = create_decision_tree_recursive(X_part_1, Y_part_1, X_index_visited, max_depth-1)
            identify_2 = (!max_fn).(X_data[:, max_index])
            X_part_2 = X_data[identify_2, :]
            Y_part_2 = Y_data[identify_2]
            children[(!max_fn)] = create_decision_tree_recursive(X_part_2, Y_part_2, X_index_visited, max_depth-1)
        end
        return DecisionTree(max_index, children, nothing)
    end
    
    return create_decision_tree_recursive(X_data, Y_data, [], max_depth)
end

# ╔═╡ fe83da85-c0fd-4955-98f5-270a8e6caa77
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

# ╔═╡ 28c70ec7-ebc5-4826-a0d0-b80cc8603dc7
tree = create_decision_tree(X_data, Y_data, max_depth=20)

# ╔═╡ 2301126e-6659-4162-bb5f-4eadc0601f88
JuTools.compute_accuracy(predict(X_data, tree), Y_data)

# ╔═╡ Cell order:
# ╟─d0106330-c5a6-11eb-17b7-77a6ed6e1efe
# ╟─88897152-4b3d-4a62-bc67-32970342d24f
# ╠═43e44b5a-62d7-4f95-8685-62470ee55453
# ╠═2a6db899-de2d-497f-8290-f805d59deec6
# ╠═d5361f35-8091-4f26-83bd-40bd3cd033d3
# ╠═d76345d3-759e-4cde-a371-7c59fdee0b3e
# ╠═a5c720d7-aacc-4e91-a045-37aecfcef305
# ╠═fe83da85-c0fd-4955-98f5-270a8e6caa77
# ╠═28c70ec7-ebc5-4826-a0d0-b80cc8603dc7
# ╠═2301126e-6659-4162-bb5f-4eadc0601f88
