### A Pluto.jl notebook ###
# v0.14.7

using Markdown
using InteractiveUtils

# ╔═╡ 59c9b880-c5a5-11eb-3fbb-8f6c9389cbbd
md"""
### Test Code for Module KNN
"""

# ╔═╡ 44bde682-a0d6-4e06-8123-ff380bec0266
module script include("./script.jl") end

# ╔═╡ 4ba5bce5-61f5-4e34-9627-2971112ce3fc
KNN = script.KNN

# ╔═╡ 0a12f2ac-459b-4ead-ae71-ded368bcbb5a
import Random

# ╔═╡ 32bb3911-1d61-4ca4-bba6-af98b3d6e545
import Plots

# ╔═╡ 6febcd4a-facd-4301-9183-10853be03633
module tools include("../tools.jl") end

# ╔═╡ 4aa17a82-5872-42d1-9315-620678eb1634
JuTools = tools.JuTools

# ╔═╡ 541ebf6c-0059-4305-ae2b-a16331fa5232
begin
# prepare data, no need to scale
linear_func = m -> 2 * m - 20
X_data, Y_data = JuTools.data_generate_linear_2d(linear_func=linear_func, data_size=500, 
    range_min=0.0, range_max=100.0, random_scale=20.0)
size(X_data), size(Y_data)
end

# ╔═╡ 9b8a6f72-ac55-4567-bc4c-387602df923e
let
# plot the data
plot_X1 = [X_data[i, 1] for i in 1:(size(X_data)[1]) if Y_data[i] == 0]
plot_Y1 = [X_data[i, 2] for i in 1:(size(X_data)[1]) if Y_data[i] == 0]
plot_X2 = [X_data[i, 1] for i in 1:(size(X_data)[1]) if Y_data[i] == 1]
plot_Y2 = [X_data[i, 2] for i in 1:(size(X_data)[1]) if Y_data[i] == 1]
plot_X3 = [i for i in 0:100]
plot_Y3 = [linear_func(m) for m in plot_X3]
Plots.gr()
Plots.scatter(plot_X1, plot_Y1, leg=false, c="red", background_color=Plots.RGB(0.2, 0.2, 0.2))
Plots.scatter!(plot_X2, plot_Y2, leg=false, c="blue")
Plots.plot!(plot_X3, plot_Y3, leg=false, c="orange", linewidth=5)
Plots.xlims!((0.0, 100.0))
Plots.ylims!((0.0, 100.0))
end

# ╔═╡ 62b49cde-f3fc-42b4-bbfb-ce2d01248925
md"""
### Naive Approach
"""

# ╔═╡ 0a4cc82c-4e14-4224-a172-d0fd3643db27
let
plot_X1 = [X_data[i, 1] for i in 1:(size(X_data)[1]) if Y_data[i] == 0]
plot_Y1 = [X_data[i, 2] for i in 1:(size(X_data)[1]) if Y_data[i] == 0]
plot_X2 = [X_data[i, 1] for i in 1:(size(X_data)[1]) if Y_data[i] == 1]
plot_Y2 = [X_data[i, 2] for i in 1:(size(X_data)[1]) if Y_data[i] == 1]
plot_X3 = [i for i in 0:100]
plot_Y3 = [linear_func(m) for m in plot_X3]
# plot contour
X_test_1 = 0:1:100
X_test_2 = 0:1:100
f_test(X1, X2) = begin
   KNN.predict_naive([X1, X2], X_data, Y_data, K=5)[1] 
end
Plots.gr()
Plots.contour(X_test_1, X_test_2, f_test, leg=false, fill=true, c=:matter,
    linewidth=0, background_color=Plots.RGB(0.2, 0.2, 0.2))
Plots.scatter!(plot_X1, plot_Y1, leg=false, c="red")
Plots.scatter!(plot_X2, plot_Y2, leg=false, c="blue")
Plots.plot!(plot_X3, plot_Y3, leg=false, c="orange", linewidth=5)
Plots.xlims!((0.0, 100.0))
Plots.ylims!((0.0, 100.0))
end

# ╔═╡ ecfeb65a-b7bd-4941-8b83-179860c3bf23
let
plot_X1 = [X_data[i, 1] for i in 1:(size(X_data)[1]) if Y_data[i] == 0]
plot_Y1 = [X_data[i, 2] for i in 1:(size(X_data)[1]) if Y_data[i] == 0]
plot_X2 = [X_data[i, 1] for i in 1:(size(X_data)[1]) if Y_data[i] == 1]
plot_Y2 = [X_data[i, 2] for i in 1:(size(X_data)[1]) if Y_data[i] == 1]
plot_X3 = [i for i in 0:100]
plot_Y3 = [linear_func(m) for m in plot_X3]
# test with different Ks
X_test_1 = 0:1:100
X_test_2 = 0:1:100
f_test(X1, X2) = begin
   KNN.predict_naive([X1, X2], X_data, Y_data, K=2)[1] 
end

Plots.gr()
Plots.contour(X_test_1, X_test_2, f_test, leg=false, fill=true, c=:matter,
    linewidth=0, background_color=Plots.RGB(0.2, 0.2, 0.2))
Plots.scatter!(plot_X1, plot_Y1, leg=false, c="red")
Plots.scatter!(plot_X2, plot_Y2, leg=false, c="blue")
Plots.plot!(plot_X3, plot_Y3, leg=false, c="orange", linewidth=5)
Plots.xlims!((0.0, 100.0))
Plots.ylims!((0.0, 100.0))
end

# ╔═╡ cdc0071f-a7c6-4d17-b559-ddbf777b9969
let
plot_X1 = [X_data[i, 1] for i in 1:(size(X_data)[1]) if Y_data[i] == 0]
plot_Y1 = [X_data[i, 2] for i in 1:(size(X_data)[1]) if Y_data[i] == 0]
plot_X2 = [X_data[i, 1] for i in 1:(size(X_data)[1]) if Y_data[i] == 1]
plot_Y2 = [X_data[i, 2] for i in 1:(size(X_data)[1]) if Y_data[i] == 1]
plot_X3 = [i for i in 0:100]
plot_Y3 = [linear_func(m) for m in plot_X3]
X_test_1 = 0:1:100
X_test_2 = 0:1:100
f_test(X1, X2) = begin
   KNN.predict_naive([X1, X2], X_data, Y_data, K=10)[1] 
end

Plots.gr()
Plots.contour(X_test_1, X_test_2, f_test, leg=false, fill=true, c=:matter,
    linewidth=0, background_color=Plots.RGB(0.2, 0.2, 0.2))
Plots.scatter!(plot_X1, plot_Y1, leg=false, c="red")
Plots.scatter!(plot_X2, plot_Y2, leg=false, c="blue")
Plots.plot!(plot_X3, plot_Y3, leg=false, c="orange", linewidth=5)
Plots.xlims!((0.0, 100.0))
Plots.ylims!((0.0, 100.0))
end

# ╔═╡ 86bcd18d-4562-4ba6-bdb8-c45b2a935540
md"""
With larger `K`, the algorithm can find the boundary more easily
"""

# ╔═╡ bcaaf451-1430-43d5-b737-87129dd0579b
md"""
Now compare with euclidean distance metric
"""

# ╔═╡ 22719609-1a57-4fe2-a1d5-0d7494c54aaf
let
plot_X1 = [X_data[i, 1] for i in 1:(size(X_data)[1]) if Y_data[i] == 0]
plot_Y1 = [X_data[i, 2] for i in 1:(size(X_data)[1]) if Y_data[i] == 0]
plot_X2 = [X_data[i, 1] for i in 1:(size(X_data)[1]) if Y_data[i] == 1]
plot_Y2 = [X_data[i, 2] for i in 1:(size(X_data)[1]) if Y_data[i] == 1]
plot_X3 = [i for i in 0:100]
plot_Y3 = [linear_func(m) for m in plot_X3]
X_test_1 = 0:1:100
X_test_2 = 0:1:100
f_test(X1, X2) = begin
   KNN.predict_naive_fun([X1, X2], X_data, Y_data, K=10)[1] 
end

Plots.gr()
Plots.contour(X_test_1, X_test_2, f_test, leg=false, fill=true, c=:matter,
    linewidth=0, background_color=Plots.RGB(0.2, 0.2, 0.2))
Plots.scatter!(plot_X1, plot_Y1, leg=false, c="red")
Plots.scatter!(plot_X2, plot_Y2, leg=false, c="blue")
Plots.plot!(plot_X3, plot_Y3, leg=false, c="orange", linewidth=5)
Plots.xlims!((0.0, 100.0))
Plots.ylims!((0.0, 100.0))
end

# ╔═╡ 8bba7dab-bbf6-439f-a091-b5f4cfe41dbf
let
plot_X1 = [X_data[i, 1] for i in 1:(size(X_data)[1]) if Y_data[i] == 0]
plot_Y1 = [X_data[i, 2] for i in 1:(size(X_data)[1]) if Y_data[i] == 0]
plot_X2 = [X_data[i, 1] for i in 1:(size(X_data)[1]) if Y_data[i] == 1]
plot_Y2 = [X_data[i, 2] for i in 1:(size(X_data)[1]) if Y_data[i] == 1]
plot_X3 = [i for i in 0:100]
plot_Y3 = [linear_func(m) for m in plot_X3]
X_test_1 = 0:1:100
X_test_2 = 0:1:100
f_test(X1, X2) = begin
   KNN.predict_naive_fun([X1, X2], X_data, Y_data, K=2)[1] 
end

Plots.gr()
Plots.contour(X_test_1, X_test_2, f_test, leg=false, fill=true, c=:matter,
    linewidth=0, background_color=Plots.RGB(0.2, 0.2, 0.2))
Plots.scatter!(plot_X1, plot_Y1, leg=false, c="red")
Plots.scatter!(plot_X2, plot_Y2, leg=false, c="blue")
Plots.plot!(plot_X3, plot_Y3, leg=false, c="orange", linewidth=5)
Plots.xlims!((0.0, 100.0))
Plots.ylims!((0.0, 100.0))
end

# ╔═╡ 8f6f1ae4-9b17-4972-9fa0-d9684e79771b
let
plot_X1 = [X_data[i, 1] for i in 1:(size(X_data)[1]) if Y_data[i] == 0]
plot_Y1 = [X_data[i, 2] for i in 1:(size(X_data)[1]) if Y_data[i] == 0]
plot_X2 = [X_data[i, 1] for i in 1:(size(X_data)[1]) if Y_data[i] == 1]
plot_Y2 = [X_data[i, 2] for i in 1:(size(X_data)[1]) if Y_data[i] == 1]
plot_X3 = [i for i in 0:100]
plot_Y3 = [linear_func(m) for m in plot_X3]
X_test_1 = 0:1:100
X_test_2 = 0:1:100
f_test(X1, X2) = begin
   KNN.predict_naive_fun([X1, X2], X_data, Y_data, K=30)[1] 
end

Plots.gr()
Plots.contour(X_test_1, X_test_2, f_test, leg=false, fill=true, c=:matter,
    linewidth=0, background_color=Plots.RGB(0.2, 0.2, 0.2))
Plots.scatter!(plot_X1, plot_Y1, leg=false, c="red")
Plots.scatter!(plot_X2, plot_Y2, leg=false, c="blue")
Plots.plot!(plot_X3, plot_Y3, leg=false, c="orange", linewidth=5)
Plots.xlims!((0.0, 100.0))
Plots.ylims!((0.0, 100.0))
end

# ╔═╡ 64a60947-e048-4d53-8b9c-14dc76e5cd0b
md"""
#### Note

From the comparison, we see the difference between euclidean distance and vector angle in terms of measuring data point similarity


The following `KdTree` and `BallTree` are all measured based on euclidean distance  
So we will see similar boundary shape
"""

# ╔═╡ 5ee1cef1-376c-4401-b197-29e242403bb0
md"""
### K-d Tree Approach
"""

# ╔═╡ 96296a72-8f2c-4ea0-a476-31f929dda576
kdtree = KNN.create_kdtree(X_data, Y_data)

# ╔═╡ 18bbb469-96b2-4f19-8c7a-4b863a73871a
let
plot_X1 = [X_data[i, 1] for i in 1:(size(X_data)[1]) if Y_data[i] == 0]
plot_Y1 = [X_data[i, 2] for i in 1:(size(X_data)[1]) if Y_data[i] == 0]
plot_X2 = [X_data[i, 1] for i in 1:(size(X_data)[1]) if Y_data[i] == 1]
plot_Y2 = [X_data[i, 2] for i in 1:(size(X_data)[1]) if Y_data[i] == 1]
plot_X3 = [i for i in 0:100]
plot_Y3 = [linear_func(m) for m in plot_X3]
X_test_1 = 0:1:100
X_test_2 = 0:1:100
f_test(X1, X2) = begin
   KNN.predict_kdtree([X1, X2], kdtree, K=10)[1] 
end

Plots.gr()
Plots.contour(X_test_1, X_test_2, f_test, leg=false, fill=true, c=:matter,
    linewidth=0, background_color=Plots.RGB(0.2, 0.2, 0.2))
Plots.scatter!(plot_X1, plot_Y1, leg=false, c="red")
Plots.scatter!(plot_X2, plot_Y2, leg=false, c="blue")
Plots.plot!(plot_X3, plot_Y3, leg=false, c="orange", linewidth=5)
Plots.xlims!((0.0, 100.0))
Plots.ylims!((0.0, 100.0))
end

# ╔═╡ 8cce776f-5853-4246-a039-47eedeaea266
let
plot_X1 = [X_data[i, 1] for i in 1:(size(X_data)[1]) if Y_data[i] == 0]
plot_Y1 = [X_data[i, 2] for i in 1:(size(X_data)[1]) if Y_data[i] == 0]
plot_X2 = [X_data[i, 1] for i in 1:(size(X_data)[1]) if Y_data[i] == 1]
plot_Y2 = [X_data[i, 2] for i in 1:(size(X_data)[1]) if Y_data[i] == 1]
plot_X3 = [i for i in 0:100]
plot_Y3 = [linear_func(m) for m in plot_X3]
X_test_1 = 0:1:100
X_test_2 = 0:1:100
f_test(X1, X2) = begin
   KNN.predict_kdtree([X1, X2], kdtree, K=2)[1] 
end

Plots.gr()
Plots.contour(X_test_1, X_test_2, f_test, leg=false, fill=true, c=:matter,
    linewidth=0, background_color=Plots.RGB(0.2, 0.2, 0.2))
Plots.scatter!(plot_X1, plot_Y1, leg=false, c="red")
Plots.scatter!(plot_X2, plot_Y2, leg=false, c="blue")
Plots.plot!(plot_X3, plot_Y3, leg=false, c="orange", linewidth=5)
Plots.xlims!((0.0, 100.0))
Plots.ylims!((0.0, 100.0))
end

# ╔═╡ d5a9ebda-e772-4888-bc8f-3faff0413aad
let
plot_X1 = [X_data[i, 1] for i in 1:(size(X_data)[1]) if Y_data[i] == 0]
plot_Y1 = [X_data[i, 2] for i in 1:(size(X_data)[1]) if Y_data[i] == 0]
plot_X2 = [X_data[i, 1] for i in 1:(size(X_data)[1]) if Y_data[i] == 1]
plot_Y2 = [X_data[i, 2] for i in 1:(size(X_data)[1]) if Y_data[i] == 1]
plot_X3 = [i for i in 0:100]
plot_Y3 = [linear_func(m) for m in plot_X3]
X_test_1 = 0:1:100
X_test_2 = 0:1:100
f_test(X1, X2) = begin
   KNN.predict_kdtree([X1, X2], kdtree, K=30)[1] 
end

Plots.gr()
Plots.contour(X_test_1, X_test_2, f_test, leg=false, fill=true, c=:matter,
    linewidth=0, background_color=Plots.RGB(0.2, 0.2, 0.2))
Plots.scatter!(plot_X1, plot_Y1, leg=false, c="red")
Plots.scatter!(plot_X2, plot_Y2, leg=false, c="blue")
Plots.plot!(plot_X3, plot_Y3, leg=false, c="orange", linewidth=5)
Plots.xlims!((0.0, 100.0))
Plots.ylims!((0.0, 100.0))
end

# ╔═╡ d311739d-2455-448f-b343-06ecdf9d725b
md"""
With larger `K`, the boundary is more linear
"""

# ╔═╡ ee593cd1-21bc-4b33-bf87-fd5254676e9d
md"""
### Ball Tree Approach
"""

# ╔═╡ dc3f28e4-7acc-45bf-b720-fe33d8d6bb3f
balltree = KNN.create_balltree(X_data, Y_data)

# ╔═╡ 6642011a-15ee-4b5b-afb6-fd7ac990b503
let
plot_X1 = [X_data[i, 1] for i in 1:(size(X_data)[1]) if Y_data[i] == 0]
plot_Y1 = [X_data[i, 2] for i in 1:(size(X_data)[1]) if Y_data[i] == 0]
plot_X2 = [X_data[i, 1] for i in 1:(size(X_data)[1]) if Y_data[i] == 1]
plot_Y2 = [X_data[i, 2] for i in 1:(size(X_data)[1]) if Y_data[i] == 1]
plot_X3 = [i for i in 0:100]
plot_Y3 = [linear_func(m) for m in plot_X3]
X_test_1 = 0:1:100
X_test_2 = 0:1:100
f_test(X1, X2) = begin
   KNN.predict_balltree([X1, X2], balltree, K=10)[1] 
end

Plots.gr()
Plots.contour(X_test_1, X_test_2, f_test, leg=false, fill=true, c=:matter,
    linewidth=0, background_color=Plots.RGB(0.2, 0.2, 0.2))
Plots.scatter!(plot_X1, plot_Y1, leg=false, c="red")
Plots.scatter!(plot_X2, plot_Y2, leg=false, c="blue")
Plots.plot!(plot_X3, plot_Y3, leg=false, c="orange", linewidth=5)
Plots.xlims!((0.0, 100.0))
Plots.ylims!((0.0, 100.0))
end

# ╔═╡ 9e702dff-6ab7-4e9b-8dcb-de2afd85ef80
let
plot_X1 = [X_data[i, 1] for i in 1:(size(X_data)[1]) if Y_data[i] == 0]
plot_Y1 = [X_data[i, 2] for i in 1:(size(X_data)[1]) if Y_data[i] == 0]
plot_X2 = [X_data[i, 1] for i in 1:(size(X_data)[1]) if Y_data[i] == 1]
plot_Y2 = [X_data[i, 2] for i in 1:(size(X_data)[1]) if Y_data[i] == 1]
plot_X3 = [i for i in 0:100]
plot_Y3 = [linear_func(m) for m in plot_X3]
X_test_1 = 0:1:100
X_test_2 = 0:1:100
f_test(X1, X2) = begin
   KNN.predict_balltree([X1, X2], balltree, K=2)[1] 
end

Plots.gr()
Plots.contour(X_test_1, X_test_2, f_test, leg=false, fill=true, c=:matter,
    linewidth=0, background_color=Plots.RGB(0.2, 0.2, 0.2))
Plots.scatter!(plot_X1, plot_Y1, leg=false, c="red")
Plots.scatter!(plot_X2, plot_Y2, leg=false, c="blue")
Plots.plot!(plot_X3, plot_Y3, leg=false, c="orange", linewidth=5)
Plots.xlims!((0.0, 100.0))
Plots.ylims!((0.0, 100.0))
end

# ╔═╡ 1720df3b-7290-4b74-b4b7-6eb7b68e551d
let
plot_X1 = [X_data[i, 1] for i in 1:(size(X_data)[1]) if Y_data[i] == 0]
plot_Y1 = [X_data[i, 2] for i in 1:(size(X_data)[1]) if Y_data[i] == 0]
plot_X2 = [X_data[i, 1] for i in 1:(size(X_data)[1]) if Y_data[i] == 1]
plot_Y2 = [X_data[i, 2] for i in 1:(size(X_data)[1]) if Y_data[i] == 1]
plot_X3 = [i for i in 0:100]
plot_Y3 = [linear_func(m) for m in plot_X3]
X_test_1 = 0:1:100
X_test_2 = 0:1:100
f_test(X1, X2) = begin
   KNN.predict_balltree([X1, X2], balltree, K=30)[1] 
end

Plots.gr()
Plots.contour(X_test_1, X_test_2, f_test, leg=false, fill=true, c=:matter,
    linewidth=0, background_color=Plots.RGB(0.2, 0.2, 0.2))
Plots.scatter!(plot_X1, plot_Y1, leg=false, c="red")
Plots.scatter!(plot_X2, plot_Y2, leg=false, c="blue")
Plots.plot!(plot_X3, plot_Y3, leg=false, c="orange", linewidth=5)
Plots.xlims!((0.0, 100.0))
Plots.ylims!((0.0, 100.0))
end

# ╔═╡ 655441e7-ff51-4f7c-839d-fa60339edadf
# compare time
@time KNN.predict_naive(X_data, X_data, Y_data, K=10)

# ╔═╡ 2af2c142-8cd3-49f2-a646-450cb49bf080
@time KNN.predict_kdtree(X_data, kdtree, K=10)

# ╔═╡ 8d4d270f-b569-42bf-ad4b-215975a98bc6
@time KNN.predict_balltree(X_data, balltree, K=10)

# ╔═╡ ff118ee9-12cc-40d7-9219-c654256abdfe
md"""
We see that K-d Tree approach is faster than naive approach  
Ball Tree is similar to K-d Tree, but should be faster on larger dataset
"""

# ╔═╡ Cell order:
# ╟─59c9b880-c5a5-11eb-3fbb-8f6c9389cbbd
# ╠═44bde682-a0d6-4e06-8123-ff380bec0266
# ╠═4ba5bce5-61f5-4e34-9627-2971112ce3fc
# ╠═0a12f2ac-459b-4ead-ae71-ded368bcbb5a
# ╠═32bb3911-1d61-4ca4-bba6-af98b3d6e545
# ╠═6febcd4a-facd-4301-9183-10853be03633
# ╠═4aa17a82-5872-42d1-9315-620678eb1634
# ╠═541ebf6c-0059-4305-ae2b-a16331fa5232
# ╠═9b8a6f72-ac55-4567-bc4c-387602df923e
# ╟─62b49cde-f3fc-42b4-bbfb-ce2d01248925
# ╠═0a4cc82c-4e14-4224-a172-d0fd3643db27
# ╠═ecfeb65a-b7bd-4941-8b83-179860c3bf23
# ╠═cdc0071f-a7c6-4d17-b559-ddbf777b9969
# ╟─86bcd18d-4562-4ba6-bdb8-c45b2a935540
# ╟─bcaaf451-1430-43d5-b737-87129dd0579b
# ╠═22719609-1a57-4fe2-a1d5-0d7494c54aaf
# ╠═8bba7dab-bbf6-439f-a091-b5f4cfe41dbf
# ╠═8f6f1ae4-9b17-4972-9fa0-d9684e79771b
# ╟─64a60947-e048-4d53-8b9c-14dc76e5cd0b
# ╟─5ee1cef1-376c-4401-b197-29e242403bb0
# ╠═96296a72-8f2c-4ea0-a476-31f929dda576
# ╠═18bbb469-96b2-4f19-8c7a-4b863a73871a
# ╠═8cce776f-5853-4246-a039-47eedeaea266
# ╠═d5a9ebda-e772-4888-bc8f-3faff0413aad
# ╟─d311739d-2455-448f-b343-06ecdf9d725b
# ╟─ee593cd1-21bc-4b33-bf87-fd5254676e9d
# ╠═dc3f28e4-7acc-45bf-b720-fe33d8d6bb3f
# ╠═6642011a-15ee-4b5b-afb6-fd7ac990b503
# ╠═9e702dff-6ab7-4e9b-8dcb-de2afd85ef80
# ╠═1720df3b-7290-4b74-b4b7-6eb7b68e551d
# ╠═655441e7-ff51-4f7c-839d-fa60339edadf
# ╠═2af2c142-8cd3-49f2-a646-450cb49bf080
# ╠═8d4d270f-b569-42bf-ad4b-215975a98bc6
# ╟─ff118ee9-12cc-40d7-9219-c654256abdfe
