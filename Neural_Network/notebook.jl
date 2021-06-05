### A Pluto.jl notebook ###
# v0.14.7

using Markdown
using InteractiveUtils

# ╔═╡ 5ccca702-c59f-11eb-34b6-011ca82d5945
md"""
# Neural Network
Only support classification tasks
"""

# ╔═╡ 79e17e30-a87c-450c-ac6f-bd7849c5c9c3
md"""
### Linear Layers (The Basics)
"""

# ╔═╡ e1d2cdc9-2628-40e3-9ca8-e63341dc1f2e
module tools include("../tools.jl") end

# ╔═╡ a4ab2ea6-8467-4995-b093-890f288404b2
JuTools = tools.JuTools

# ╔═╡ 14ff27dd-2926-4219-acdf-39fe6f1168e3
import Random

# ╔═╡ b0005ec6-5119-4786-b3d0-042677c61ee7
import Plots

# ╔═╡ bb1b6558-737a-4121-92a7-a78d1f1798a4
begin
X_train, Y_train = JuTools.data_generate_linear_2d(random_scale=10.0)
I = X_train
Y = Y_train
size(I), size(Y)
end

# ╔═╡ f19893ad-dc1d-40ee-8022-ee9535cc5a27
let
# plot the data
plot1 = X_train[Y_train .== 0, :]
plot2 = X_train[Y_train .== 1, :]
Plots.gr()
Plots.scatter(plot1[:, 1], plot1[:, 2], leg=false, c="red", background_color=Plots.RGB(0.2, 0.2, 0.2))
Plots.scatter!(plot2[:, 1], plot2[:, 2], leg=false, c="blue")
Plots.xlims!((0.0, 100.0))
Plots.ylims!((0.0, 100.0))
end

# ╔═╡ b3aebb54-d447-400d-8291-473c5abb606a
begin
W₁ = Random.randn((size(X_train)[2], 10)) ./ 1000 # input layer (weights)
b₁ = zeros((1, 10))                               # input layer (biases)
W₂ = Random.randn((10, 5)) ./ 1000                # hidden layer (weights)
b₂ = zeros((1, 5))                                # hidden layer (biases)
W₃ = Random.randn((5, 1)) ./ 1000                 # output layer (weights)
b₃ = zeros((1, 1))                                # output layer (biases)
size(W₁), size(W₂), size(W₃)
end

# ╔═╡ ea42ea92-5794-462d-ba7e-0adad9848697
md"""
$$\text{MSE}=\frac{1}{2n}\sum^n_{i=1}(Y_i-\hat{Y}_i)^2$$
"""

# ╔═╡ da41e33b-ad12-4c23-b992-68bdbead07e4
# define a loss function (MSE)
function loss(Y::AbstractArray, Ŷ::AbstractArray)
    @assert size(Y) == size(Ŷ)
    N = size(Y)[1]
    L = (Y .- Ŷ) .^ 2
    return sum(L) / (N * 2)
end

# ╔═╡ 0cde5d94-41e6-4133-adb2-597b632ca863
md"""
$$C=\frac{1}{2n}\sum^n_{i=1}(\hat{Y} - Y)^2$$\


$$\hat{Y} = I_3\cdot W_3+b_3$$\


$$I_3 = I_2\cdot W_2+b_2$$\


$$I_2 = I\cdot W_1+b_1$$\

where $C$ is cost, $Y$ is true target, $W_i$ is the weights of layer $i$, $b_i$ is the biases of layer $i$, and $I$ is input data
"""

# ╔═╡ 50cc214c-3399-4cc0-a039-d7354f490bd6
begin
# forward pass
I₂ = I  * W₁ .+ b₁
I₃ = I₂ * W₂ .+ b₂
Ŷ  = I₃ * W₃ .+ b₃
Ŷ  = reshape(Ŷ, :)
loss(Y, Ŷ)
end

# ╔═╡ 0a10a7ff-22c2-4a70-a9ac-8b85026dbeda
md"""
Then we can compute gradient for each variable:\


$$\begin{aligned}
\frac{\partial C}{\partial W_3} &= \frac{\partial C}{\partial \hat{Y}}\cdot\frac{\partial \hat{Y}}{\partial W_3} \\
&= \frac{1}{n}\sum^n_{i=1} (\hat{Y}-Y) \cdot (I_3) \\
\end{aligned}$$


$$\begin{aligned}
\frac{\partial C}{\partial b_3} &= \frac{\partial C}{\partial \hat{Y}}\cdot\frac{\partial \hat{Y}}{\partial b_3} \\
&= \frac{1}{n}\sum^n_{i=1}(\hat{Y}-Y) \cdot (1) \\
\end{aligned}$$


$$\begin{aligned}
\frac{\partial C}{\partial W_2} &= \frac{\partial C}{\partial \hat{Y}}\cdot\frac{\partial \hat{Y}}{\partial I_3}\cdot\frac{\partial I_3}{\partial W_2} \\
&= \frac{1}{n}\sum^n_{i=1}(\hat{Y}-Y) \cdot (W_3) \cdot (I_2) \\
\end{aligned}$$


$$\begin{aligned}
\frac{\partial C}{\partial b_2} &= \frac{\partial C}{\partial \hat{Y}}\cdot\frac{\partial \hat{Y}}{\partial I_3}\cdot\frac{\partial I_3}{\partial b_2} \\
&= \frac{1}{n}\sum^n_{i=1}(\hat{Y}-Y) \cdot (W_3) \cdot (1) \\
\end{aligned}$$


$$\begin{aligned}
\frac{\partial C}{\partial W_1} &= \frac{\partial C}{\partial \hat{Y}}\cdot\frac{\partial \hat{Y}}{\partial I_3}\cdot\frac{\partial I_3}{\partial I_2}\cdot\frac{\partial I_2}{\partial W_1} \\
&= \frac{1}{n}\sum^n_{i=1}(\hat{Y}-Y) \cdot (W_3) \cdot (W_2) \cdot (I) \\
\end{aligned}$$


$$\begin{aligned}
\frac{\partial C}{\partial b_1} &= \frac{\partial C}{\partial \hat{Y}}\cdot\frac{\partial \hat{Y}}{\partial I_3}\cdot\frac{\partial I_3}{\partial I_2}\cdot\frac{\partial I_2}{\partial b_1} \\
&= \frac{1}{n}\sum^n_{i=1}(\hat{Y}-Y) \cdot (W_3) \cdot (W_2) \cdot (1) \\
\end{aligned}$$
"""

# ╔═╡ a0173ce1-8a16-4493-85f6-aaa138ef83c3
# define learning rate r
r = 0.1

# ╔═╡ 083f9702-bbfa-405e-bc18-f01f9c23d195
n = size(I)[1]

# ╔═╡ 4cb51df1-73d8-44b8-beba-9cbae56c3db2
gW₃ = I₃' * ((Ŷ .- Y) ./ n)

# ╔═╡ 91de2648-115e-4128-a98c-c69712ada92b
gb₃ = sum(Ŷ .- Y, dims=1) ./ n

# ╔═╡ e2b9b837-7b27-4543-be24-b1f261f2d824
gW₂ = I₂' * (((Ŷ .- Y) ./ n) * W₃')

# ╔═╡ 8fa62798-fc61-4b11-82f1-f5302bf574fb
gb₂ = (sum(Ŷ .- Y, dims=1) ./ n) * W₃'

# ╔═╡ fb07afc6-abfb-4434-a502-f3b67926ff3b
gW₁ = I' * ((((Ŷ .- Y) ./ n) * W₃') * W₂')

# ╔═╡ 0deb5d6b-991b-40dc-9cef-be3797d41ff5
gb₁ = ((sum(Ŷ .- Y, dims=1) ./ n) * W₃') * W₂'

# ╔═╡ bd0756cb-00ee-48c3-b663-d122e9e03553
begin
# backpropagate
W₁ .= W₁ .- r .* gW₁
b₁ .= b₁ .- r .* gb₁
W₂ .= W₂ .- r .* gW₂
b₂ .= b₂ .- r .* gb₂
W₃ .= W₃ .- r .* gW₃
b₃ .= b₃ .- r .* gb₃
end

# ╔═╡ 2af547e8-960e-4542-9cea-fcb7e97e06c2
let
# forward pass again
I₂ = I  * W₁ .+ b₁
I₃ = I₂ * W₂ .+ b₂
Ŷ  = I₃ * W₃ .+ b₃
Ŷ  = reshape(Ŷ, :)
loss(Y, Ŷ)
end

# ╔═╡ a8adef33-608f-4fdb-b482-0cac3a084220
# now ensemble everything into structs and functions
mutable struct ComputeGraph
    W_array::AbstractArray{AbstractArray} # weights
    b_array::AbstractArray{AbstractArray} # biases
    gW_array::AbstractArray{AbstractArray} # gradients of weights
    gb_array::AbstractArray{AbstractArray} # gradients of biases
end

# ╔═╡ 2b3e8e94-580a-4796-9218-2d16dff110a7
function createGraph(weightShapes::AbstractArray; randomize=false)::ComputeGraph
    # check validity of weight shapes
    for (shape1, shape2) in zip(weightShapes, weightShapes[2:end])
        @assert length(shape1) == length(shape2)
        @assert shape1[2] == shape2[1]
    end
    # initialize data
    W_array = []
    b_array = []
    gW_array = []
    gb_array = []
    for shape in weightShapes
        if randomize
            push!(W_array, Random.randn(shape) .* ((2.0 / sum(shape)) ^ 0.5))
        else
            push!(W_array, zeros(shape))
        end
        push!(b_array, zeros((1, shape[2])))
        push!(gW_array, zeros(shape))
        push!(gb_array, zeros((1, shape[2])))
    end
    # create graph
    graph = ComputeGraph(W_array, b_array, gW_array, gb_array)
    return graph
end

# ╔═╡ 75f3884d-540d-469d-a3b8-15245503224c
# this function compute output at layer defined by index recursively
function computeI(index::Integer, X::AbstractArray, graph::ComputeGraph)
    @assert index >= 1
    @assert index <= length(graph.W_array) + 1
    if index == 1
        return X
    end
    I = computeI(index-1, X, graph) * graph.W_array[index-1] .+ graph.b_array[index-1]
    return I
end

# ╔═╡ 63a5d479-ca9a-4f70-8c1b-5ce0d840fef1
# this function compute forward pass
# and return intermediate outputs at each layer
# this function will create huge arrays if input data is large
# so I will not use it in practice
function forward(X::AbstractArray, graph::ComputeGraph)::AbstractArray
    I_array = []
    for (W,b) in zip(graph.W_array, graph.b_array)
        if length(I_array <= 0)
            push!(I_array, (W * X .+ b))
        else
            push!(I_array, (W * I_array[end] .+ b))
        end
    end
    return I_array
end

# ╔═╡ a4e21ead-df90-4e58-9659-db9074199ff9
# this function compute gradients for each W,b and set values inplace
# it includes forward pass computation as well
function backpropagate!(X::AbstractArray, Y::AbstractArray, graph::ComputeGraph)
    tmp = (reshape(computeI(length(graph.W_array)+1, X, graph), :) .- Y) ./ length(Y)
    for index in length(graph.W_array):-1:1
        graph.gW_array[index] .= (computeI(index, X, graph)' * tmp)
        graph.gb_array[index] .= sum(tmp, dims=1)
        tmp = tmp * graph.W_array[index]'
    end
end

# ╔═╡ 312200d0-87f5-4591-91ca-035d6a1dcf04
# this function updates W,b based on their gradients
function step!(lr::AbstractFloat, graph::ComputeGraph)
    for index in 1:length(graph.W_array)
        graph.W_array[index] .= graph.W_array[index] .- lr .* graph.gW_array[index]
        graph.b_array[index] .= graph.b_array[index] .- lr .* graph.gb_array[index]
    end
end

# ╔═╡ 4cf70783-3486-4ab4-bf78-f862bc996d1e
# this function maps continuous prediction to categorical values (0, 1)
function predict(X::AbstractArray, graph::ComputeGraph)
    out = reshape(computeI(length(graph.W_array)+1, X, graph), :)
    out[out .>= 0.5] .= 1.0
    out[out .< 0.5]  .= 0.0
    return out
end

# ╔═╡ 8bfbf0e3-ffff-4e89-a5bd-aaca59ba7555
function predict_proba(X::AbstractArray, graph::ComputeGraph)
    out = reshape(computeI(length(graph.W_array)+1, X, graph), :)
    return out
end

# ╔═╡ a15c1a7d-15d8-44b9-8cff-b2c65afc9371
function train!(X::AbstractArray, Y::AbstractArray, graph::ComputeGraph, lr::AbstractFloat; max_iter::Integer=20, batch_size=50, verbose=false)
    @assert max_iter >= 0
    @assert 0 <= batch_size <= length(Y)
    for epoch in 1:max_iter
        X, Y = JuTools.shuffle_data(X, Y)
        for i in 1:batch_size:length(Y)
            if i + batch_size - 1 > length(Y)
                backpropagate!(X[i:end, :], Y[i:end, :], graph)
            else
                backpropagate!(X[i:(i+batch_size-1), :], Y[i:(i+batch_size-1), :], graph)
            end
            step!(lr, graph)
        end
        if verbose
            Ŷ = predict_proba(X, graph)
            L = loss(Y, Ŷ)
            println(string("Epoch: ", epoch, "\tLoss: ", L))
        end
    end
end

# ╔═╡ 046c8937-7acf-4e87-a8dd-421dfedc7ea2
begin
shapes = [(size(X_train)[2], 5), (5, 1)]
graph = createGraph(shapes, randomize=true)
train!(X_train, Y_train, graph, 0.0001, max_iter=10000, batch_size=50, verbose=false)
end

# ╔═╡ a5a626c0-ae4b-47fd-820a-aeef016add10
let
# plot decision probs
plot1 = X_train[Y_train .== 0, :]
plot2 = X_train[Y_train .== 1, :]
plot3 = collect(0:100)
plot_contour = hcat(vec(repeat(reshape(plot3, :, 1), 1, 101)'), vec(repeat(plot3, 101, 1)))
plot_contour = reshape(predict_proba(plot_contour, graph), 101, :)
Plots.gr()
Plots.contour(plot3, plot3, plot_contour, leg=true, fill=true, c=Plots.cgrad(:matter, rev=false, categorical=false),
    linewidth=0, background_color=Plots.RGB(0.2, 0.2, 0.2))
Plots.scatter!(plot1[:, 1], plot1[:, 2], leg=false, c="red")
Plots.scatter!(plot2[:, 1], plot2[:, 2], leg=false, c="blue")
Plots.xlims!((0.0, 100.0))
Plots.ylims!((0.0, 100.0))
end

# ╔═╡ eed53a70-abbe-4fbc-b105-da35cc237c23
let
# plot decision boundary
plot1 = X_train[Y_train .== 0, :]
plot2 = X_train[Y_train .== 1, :]
plot3 = collect(0:100)
plot_contour = hcat(vec(repeat(reshape(plot3, :, 1), 1, 101)'), vec(repeat(plot3, 101, 1)))
plot_contour = reshape(predict(plot_contour, graph), 101, :)
Plots.gr()
Plots.contour(plot3, plot3, plot_contour, leg=true, fill=true, c=Plots.cgrad(:matter, rev=false, categorical=false),
    linewidth=0, background_color=Plots.RGB(0.2, 0.2, 0.2))
Plots.scatter!(plot1[:, 1], plot1[:, 2], leg=false, c="red")
Plots.scatter!(plot2[:, 1], plot2[:, 2], leg=false, c="blue")
Plots.xlims!((0.0, 100.0))
Plots.ylims!((0.0, 100.0))
end

# ╔═╡ 35b5af9f-a410-47e5-bf77-09d9ab60b3be
JuTools.compute_accuracy(predict(X_train, graph), Y_train)

# ╔═╡ 49d0407a-224f-4200-83a5-c7564bcf1a8c
md"""
### About weights initialization

In current Neural Network setup, weights should __never__ be initialized to all zeros


For example:\
$$\begin{aligned}
\frac{\partial C}{\partial W_2} &= \frac{\partial C}{\partial \hat{Y}}\cdot\frac{\partial \hat{Y}}{\partial I_3}\cdot\frac{\partial I_3}{\partial W_2} \\
&= \frac{1}{n}\sum^n_{i=1}(\hat{Y}-Y) \cdot (W_3) \cdot (I_2)
\end{aligned}$$\

where gradient of $W_2$ is dependent on $W_3$\


However, since:\
$$\begin{aligned}
\frac{\partial C}{\partial W_3} &= \frac{\partial C}{\partial \hat{Y}}\cdot\frac{\partial \hat{Y}}{\partial W_3} \\
&= \frac{1}{n}\sum^n_{i=1} (\hat{Y}-Y) \cdot (I_3)
\end{aligned}$$\

where gradient of $W_3$ is dependent on input of layer 3, which is computed from $W_2$\


Both $W_2$ and $W_3$ will remain zeros during training\


Therefore, we need to initialize weights randomly\
But there are 2 problems:
* If weights are too large, gradients will go infinity, which is called __exploding gradients__  
* If weights are too small, gradients will be near zero, which is called __vanishing gradients__  

In both cases, the network fails to learn\


Assume we want a weight $w$ of shape $(m,n)$\
To solve this issue, we have 2 methods to initialize a weight:
* He Initialization  
$$w=\text{randn}\cdot\sqrt{\frac{2}{n}}$$  
* Xavier Initialization  
$$w=\text{randn}\cdot\sqrt{\frac{1}{n}}$$  

And I use the following method:

$$w=\text{randn}\cdot\sqrt{\frac{2}{m+n}}$$
"""

# ╔═╡ e3afc7de-f6ea-44e4-9be6-78ee952f62ab
md"""
### About learning rate

A large learning rate makes weights go infinity, while a small learning rate makes learning very slow


A better solution than fixed learning rate is to adjust learning rate based on gradient updates (Adaptive Learning Rate)\
Popular algorithms are:
* RMSProp  
* Adagrad  
* Adam  

Other gradient descent optimization algorithms can be found [here](https://ruder.io/optimizing-gradient-descent/index.html#gradientdescentoptimizationalgorithms)
"""

# ╔═╡ Cell order:
# ╟─5ccca702-c59f-11eb-34b6-011ca82d5945
# ╟─79e17e30-a87c-450c-ac6f-bd7849c5c9c3
# ╠═e1d2cdc9-2628-40e3-9ca8-e63341dc1f2e
# ╠═a4ab2ea6-8467-4995-b093-890f288404b2
# ╠═14ff27dd-2926-4219-acdf-39fe6f1168e3
# ╠═b0005ec6-5119-4786-b3d0-042677c61ee7
# ╠═bb1b6558-737a-4121-92a7-a78d1f1798a4
# ╠═f19893ad-dc1d-40ee-8022-ee9535cc5a27
# ╠═b3aebb54-d447-400d-8291-473c5abb606a
# ╟─ea42ea92-5794-462d-ba7e-0adad9848697
# ╠═da41e33b-ad12-4c23-b992-68bdbead07e4
# ╟─0cde5d94-41e6-4133-adb2-597b632ca863
# ╠═50cc214c-3399-4cc0-a039-d7354f490bd6
# ╟─0a10a7ff-22c2-4a70-a9ac-8b85026dbeda
# ╠═a0173ce1-8a16-4493-85f6-aaa138ef83c3
# ╠═083f9702-bbfa-405e-bc18-f01f9c23d195
# ╠═4cb51df1-73d8-44b8-beba-9cbae56c3db2
# ╠═91de2648-115e-4128-a98c-c69712ada92b
# ╠═e2b9b837-7b27-4543-be24-b1f261f2d824
# ╠═8fa62798-fc61-4b11-82f1-f5302bf574fb
# ╠═fb07afc6-abfb-4434-a502-f3b67926ff3b
# ╠═0deb5d6b-991b-40dc-9cef-be3797d41ff5
# ╠═bd0756cb-00ee-48c3-b663-d122e9e03553
# ╠═2af547e8-960e-4542-9cea-fcb7e97e06c2
# ╠═a8adef33-608f-4fdb-b482-0cac3a084220
# ╠═2b3e8e94-580a-4796-9218-2d16dff110a7
# ╠═75f3884d-540d-469d-a3b8-15245503224c
# ╠═63a5d479-ca9a-4f70-8c1b-5ce0d840fef1
# ╠═a4e21ead-df90-4e58-9659-db9074199ff9
# ╠═312200d0-87f5-4591-91ca-035d6a1dcf04
# ╠═4cf70783-3486-4ab4-bf78-f862bc996d1e
# ╠═8bfbf0e3-ffff-4e89-a5bd-aaca59ba7555
# ╠═a15c1a7d-15d8-44b9-8cff-b2c65afc9371
# ╠═046c8937-7acf-4e87-a8dd-421dfedc7ea2
# ╠═a5a626c0-ae4b-47fd-820a-aeef016add10
# ╠═eed53a70-abbe-4fbc-b105-da35cc237c23
# ╠═35b5af9f-a410-47e5-bf77-09d9ab60b3be
# ╟─49d0407a-224f-4200-83a5-c7564bcf1a8c
# ╟─e3afc7de-f6ea-44e4-9be6-78ee952f62ab
