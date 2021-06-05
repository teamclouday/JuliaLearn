### A Pluto.jl notebook ###
# v0.14.7

using Markdown
using InteractiveUtils

# ╔═╡ 3ef58460-c5a7-11eb-3bc6-f73e35e42d97
md"""
### Test Code for Module Tree
"""

# ╔═╡ 2e2c77ad-49e3-4e21-a119-1bee9118d848
module script include("./script.jl") end

# ╔═╡ 6e2eec65-d0e6-4e27-a2f4-11b4b820a805
Tree = script.Tree

# ╔═╡ 8279d02f-0684-4a9b-b1ab-705e74d7cd6e
module tools include("../tools.jl") end

# ╔═╡ b5ef1e5b-8078-490c-8a2e-79087653918f
JuTools = tools.JuTools

# ╔═╡ 4ed8ef3a-c7ee-48ce-baa8-84a4ecff939d
import Plots

# ╔═╡ 9db5eb6f-5822-4af5-95d8-997a9cdf3084
import Random

# ╔═╡ 360ed8fc-f1a3-4970-a3e8-a00e78734fc7
begin
X_data1, Y_data1 = JuTools.data_generate_cluster_2d(pos1=(20.0, 20.0), pos2=(50.0, 50.0),
    radius1=5.0, radius2=5.0, random_scale=8.0, data_size=1000)
size(X_data1), size(Y_data1)
end

# ╔═╡ 0eba36a8-316a-47a7-9d6c-f25eefa80192
begin
max_depth = 3
tree1 = Tree.create_decision_tree(X_data1, Y_data1, max_depth=max_depth)
end

# ╔═╡ 9284844b-a29c-49c2-bc78-2aa66bfa6460
let
plot_X1 = [X_data1[i, 1] for i in 1:(size(X_data1)[1]) if Y_data1[i] == 0.0]
plot_Y1 = [X_data1[i, 2] for i in 1:(size(X_data1)[1]) if Y_data1[i] == 0.0]
plot_X2 = [X_data1[i, 1] for i in 1:(size(X_data1)[1]) if Y_data1[i] == 1.0]
plot_Y2 = [X_data1[i, 2] for i in 1:(size(X_data1)[1]) if Y_data1[i] == 1.0]
X_test_1 = 0:70
X_test_2 = 0:70
X_test_contour = hcat(vec(repeat(reshape(X_test_1, :, 1), 1, 71)'), vec(repeat(X_test_2, 71, 1)))
X_test_contour = reshape(Tree.predict(X_test_contour, tree1), 71, :)
Plots.gr()
Plots.contour(X_test_1, X_test_2, X_test_contour, fill=true, c=Plots.cgrad(:matter, rev=false, categorical=false),
    linewidth=0, background_color=Plots.RGB(0.2, 0.2, 0.2))
Plots.scatter!(plot_X1, plot_Y1, leg=false, c="red")
Plots.scatter!(plot_X2, plot_Y2, leg=false, c="blue")
Plots.xlims!((0.0, 70.0))
Plots.ylims!((0.0, 70.0))
Plots.title!("Decision Tree C4.5, max_depth=$max_depth")
end

# ╔═╡ 52e82e96-9a5e-4294-b516-e4bcce028ef2
JuTools.compute_accuracy(Tree.predict(X_data1, tree1), Y_data1)

# ╔═╡ ad8c0261-e9fd-414d-8ed5-357b61162216
tree2 = Tree.create_decision_tree(X_data1, Y_data1, max_depth=10)

# ╔═╡ cbbfed54-c98d-4f4b-a347-4e437625e90b
let
max_depth = 10
plot_X1 = [X_data1[i, 1] for i in 1:(size(X_data1)[1]) if Y_data1[i] == 0.0]
plot_Y1 = [X_data1[i, 2] for i in 1:(size(X_data1)[1]) if Y_data1[i] == 0.0]
plot_X2 = [X_data1[i, 1] for i in 1:(size(X_data1)[1]) if Y_data1[i] == 1.0]
plot_Y2 = [X_data1[i, 2] for i in 1:(size(X_data1)[1]) if Y_data1[i] == 1.0]
X_test_1 = 0:70
X_test_2 = 0:70
X_test_contour = hcat(vec(repeat(reshape(X_test_1, :, 1), 1, 71)'), vec(repeat(X_test_2, 71, 1)))
X_test_contour = reshape(Tree.predict(X_test_contour, tree2), 71, :)
Plots.gr()
Plots.contour(X_test_1, X_test_2, X_test_contour, fill=true, c=Plots.cgrad(:matter, rev=false, categorical=false),
    linewidth=0, background_color=Plots.RGB(0.2, 0.2, 0.2))
Plots.scatter!(plot_X1, plot_Y1, leg=false, c="red")
Plots.scatter!(plot_X2, plot_Y2, leg=false, c="blue")
Plots.xlims!((0.0, 70.0))
Plots.ylims!((0.0, 70.0))
Plots.title!("Decision Tree C4.5, max_depth=$max_depth")
end

# ╔═╡ 7027f706-328e-4181-8d66-6267e55f8982
JuTools.compute_accuracy(Tree.predict(X_data1, tree2), Y_data1)

# ╔═╡ Cell order:
# ╟─3ef58460-c5a7-11eb-3bc6-f73e35e42d97
# ╠═2e2c77ad-49e3-4e21-a119-1bee9118d848
# ╠═6e2eec65-d0e6-4e27-a2f4-11b4b820a805
# ╠═8279d02f-0684-4a9b-b1ab-705e74d7cd6e
# ╠═b5ef1e5b-8078-490c-8a2e-79087653918f
# ╠═4ed8ef3a-c7ee-48ce-baa8-84a4ecff939d
# ╠═9db5eb6f-5822-4af5-95d8-997a9cdf3084
# ╠═360ed8fc-f1a3-4970-a3e8-a00e78734fc7
# ╠═0eba36a8-316a-47a7-9d6c-f25eefa80192
# ╠═9284844b-a29c-49c2-bc78-2aa66bfa6460
# ╠═52e82e96-9a5e-4294-b516-e4bcce028ef2
# ╠═ad8c0261-e9fd-414d-8ed5-357b61162216
# ╠═cbbfed54-c98d-4f4b-a347-4e437625e90b
# ╠═7027f706-328e-4181-8d66-6267e55f8982
