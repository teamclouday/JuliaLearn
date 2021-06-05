### A Pluto.jl notebook ###
# v0.14.7

using Markdown
using InteractiveUtils

# ╔═╡ 4c508010-c5a2-11eb-1fc1-19d8e9de32c4
md"""
### Test Code for Module NB
"""

# ╔═╡ 942227ca-947b-497e-a20b-c7a251e72f74
module script include("./script.jl") end

# ╔═╡ da372bf3-dd21-40fe-84af-eee8722a401a
NB = script.NB

# ╔═╡ bf0086c8-58b1-442b-ac75-f61a30349c63
module tools include("../tools.jl") end

# ╔═╡ 1d91c553-ba63-4c45-97c8-c14effbe5712
JuTools = tools.JuTools

# ╔═╡ 13e3bf10-217d-40d4-a23d-1f4f6b450b1e
import Plots

# ╔═╡ 22fe721d-390b-42a0-9abe-44ab67afdd38
import Random

# ╔═╡ d90f92cf-1ab3-4a7f-a1ba-c32a2ca76a04
begin
X_data1, Y_data1 = JuTools.data_generate_cluster_2d(pos1=(20.0, 20.0), pos2=(50.0, 50.0),
    radius1=5.0, radius2=5.0, random_scale=8.0, data_size=1000)
size(X_data1), size(Y_data1)
end

# ╔═╡ 23f676f9-430e-4264-b059-2494c00bb3ba
let
trained = NB.train(X_data1, Y_data1)
plot_X1 = [X_data1[i, 1] for i in 1:(size(X_data1)[1]) if Y_data1[i] == 0.0]
plot_Y1 = [X_data1[i, 2] for i in 1:(size(X_data1)[1]) if Y_data1[i] == 0.0]
plot_X2 = [X_data1[i, 1] for i in 1:(size(X_data1)[1]) if Y_data1[i] == 1.0]
plot_Y2 = [X_data1[i, 2] for i in 1:(size(X_data1)[1]) if Y_data1[i] == 1.0]
X_test_1 = 0:70
X_test_2 = 0:70
X_test_contour = hcat(vec(repeat(reshape(X_test_1, :, 1), 1, 71)'), vec(repeat(X_test_2, 71, 1)))
X_test_contour = reshape(NB.predict(X_test_contour, trained), 71, :)
Plots.gr()
Plots.contour(X_test_1, X_test_2, X_test_contour, fill=true, c=Plots.cgrad(:matter, rev=false, categorical=false),
    linewidth=0, background_color=Plots.RGB(0.2, 0.2, 0.2))
Plots.scatter!(plot_X1, plot_Y1, leg=false, c="red")
Plots.scatter!(plot_X2, plot_Y2, leg=false, c="blue")
Plots.xlims!((0.0, 70.0))
Plots.ylims!((0.0, 70.0))
Plots.title!("Naive Bayes (Gaussian)")
end

# ╔═╡ Cell order:
# ╟─4c508010-c5a2-11eb-1fc1-19d8e9de32c4
# ╠═942227ca-947b-497e-a20b-c7a251e72f74
# ╠═da372bf3-dd21-40fe-84af-eee8722a401a
# ╠═bf0086c8-58b1-442b-ac75-f61a30349c63
# ╠═1d91c553-ba63-4c45-97c8-c14effbe5712
# ╠═13e3bf10-217d-40d4-a23d-1f4f6b450b1e
# ╠═22fe721d-390b-42a0-9abe-44ab67afdd38
# ╠═d90f92cf-1ab3-4a7f-a1ba-c32a2ca76a04
# ╠═23f676f9-430e-4264-b059-2494c00bb3ba
