### A Pluto.jl notebook ###
# v0.14.7

using Markdown
using InteractiveUtils

# ╔═╡ b9dae3f2-5db7-4465-9c1e-f95df29fd48c
function ingredients(path::String)
	# this is from the Julia source code (evalfile in base/loading.jl)
	# but with the modification that it returns the module instead of the last object
	name = Symbol(basename(path))
	m = Module(name)
	Core.eval(m,
        Expr(:toplevel,
             :(eval(x) = $(Expr(:core, :eval))($name, x)),
             :(include(x) = $(Expr(:top, :include))($name, x)),
             :(include(mapexpr::Function, x) = $(Expr(:top, :include))(mapexpr, $name, x)),
             :(include($path))))
	m
end

# ╔═╡ db715d40-c599-11eb-0b28-f5c7b076c6ab
SVM = ingredients("./script.jl").SVM

# ╔═╡ 27defb3d-f5fb-4334-a441-50f7a347819d
import Plots

# ╔═╡ fc31cf17-5e20-4130-b02c-9c2bbf9d3b72
import Random

# ╔═╡ cc3033c8-4c74-4d28-ad2c-344246bbc827
JuTools = ingredients("../tools.jl").JuTools

# ╔═╡ 3c0bdafc-b98d-47c8-8b9f-69a7b94c5d85
begin
	X_data1, Y_data1 = JuTools.data_generate_cluster_2d(pos1=(20.0, 20.0), pos2=(50.0, 50.0), radius1=5.0, radius2=5.0, random_scale=8.0, data_size=1000)
	Y_data1 .= Y_data1 .* 2.0 .- 1.0;
end

# ╔═╡ b4f64a60-b8a7-4487-8897-6bd4efa37c76
size(X_data1)

# ╔═╡ 71ed19eb-602f-4b21-a874-3c0cd6ef19d7
size(Y_data1)

# ╔═╡ de69ceeb-c9fb-4038-bf3d-2c08430e973c
begin
	plot_X1 = [X_data1[i, 1] for i in 1:(size(X_data1)[1]) if Y_data1[i] == -1.0]
	plot_Y1 = [X_data1[i, 2] for i in 1:(size(X_data1)[1]) if Y_data1[i] == -1.0]
	plot_X2 = [X_data1[i, 1] for i in 1:(size(X_data1)[1]) if Y_data1[i] == 1.0]
	plot_Y2 = [X_data1[i, 2] for i in 1:(size(X_data1)[1]) if Y_data1[i] == 1.0]
end

# ╔═╡ e18d49b7-3ef2-4d5e-ae80-83af5836277f
let
	Plots.gr()
	p = Plots.scatter(plot_X1, plot_Y1, leg=false, c="red", background_color=Plots.RGB(0.2, 0.2, 0.2))
	Plots.scatter!(p, plot_X2, plot_Y2, leg=false, c="blue")
	Plots.xlims!(p, (0.0, 70.0))
	Plots.ylims!(p, (0.0, 70.0))
	p
end

# ╔═╡ 43aff583-73af-482a-82be-f3823fd40799
md"""
### Linear SVM
"""

# ╔═╡ 6e7a9a46-e0da-442d-8f45-6b5089f6d953
let
	weights = SVM.train_linear(X_data1, Y_data1, 10.0, learning_rate=0.05, max_iter=10, random_weights=false)
	JuTools.compute_accuracy(SVM.predict(X_data1, weights), Y_data1)
end

# ╔═╡ 394fb389-532f-41a7-a387-22d86e1c8e3e
let
	weights = SVM.train_linear(X_data1, Y_data1, 10.0, learning_rate=0.05, max_iter=100, random_weights=false)
	JuTools.compute_accuracy(SVM.predict(X_data1, weights), Y_data1)
end

# ╔═╡ 7ab6dbb1-194b-4816-8ad6-52abe6a7a80f
let
	weights = SVM.train_linear(X_data1, Y_data1, 10.0, learning_rate=0.05, max_iter=1000, random_weights=false)
	JuTools.compute_accuracy(SVM.predict(X_data1, weights), Y_data1)
end

# ╔═╡ 3bdb1ccd-18fe-4c91-ab1d-2e149de06097
let
	weights = SVM.train_linear(X_data1, Y_data1, 10.0, learning_rate=0.05, max_iter=1000, random_weights=false)
	X_test_1 = 0:0.1:70
	X_test_2 = 0:0.1:70
	f_test1(X1, X2) = begin
   		SVM.predict([X1, X2], weights)[1] 
	end
	Plots.gr()
	Plots.contour(X_test_1, X_test_2, f_test1, leg=false, fill=true, c=Plots.cgrad(:matter, rev=true, categorical=false), linewidth=0, background_color=Plots.RGB(0.2, 0.2, 0.2))
	Plots.scatter!(plot_X1, plot_Y1, leg=false, c="red")
	Plots.scatter!(plot_X2, plot_Y2, leg=false, c="blue")
	Plots.xlims!((0.0, 70.0))
	Plots.ylims!((0.0, 70.0))
end

# ╔═╡ fcc3d16f-1545-40ca-b87d-df959fb25ed4
let
	weights = SVM.train_linear(X_data1, Y_data1, 10.0, learning_rate=0.05, max_iter=1000, random_weights=false)
	X_test_1 = 0:0.1:70
	X_test_2 = 0:0.1:70
	f_test2(X1, X2) = begin
   		abs(SVM.predict_proba([X1, X2], weights)[1])
	end
	Plots.gr()
	Plots.contour(X_test_1, X_test_2, f_test2, leg=false, fill=true, c=Plots.cgrad(:matter, rev=false, categorical=false), linewidth=0, background_color=Plots.RGB(0.2, 0.2, 0.2))
	Plots.scatter!(plot_X1, plot_Y1, leg=false, c="red")
	Plots.scatter!(plot_X2, plot_Y2, leg=false, c="blue")
	plot_X3 = collect(0:70)
	plot_Y3 = -(weights.b .+ (weights.w[1] .* plot_X3)) ./ weights.w[2]
	Plots.plot!(plot_X3, plot_Y3, c="purple", linewidth=5, alpha=0.5)
	Plots.xlims!((0.0, 70.0))
	Plots.ylims!((0.0, 70.0))
end

# ╔═╡ 869aacb1-74bd-4b8c-99f9-416732c4e783
begin
	# try with different data
	X_data2, Y_data2 = JuTools.data_generate_cluster_2d(pos1=(30.0, 80.0), pos2=(80.0, 30.0),
	    radius1=5.0, radius2=10.0, random_scale=8.0, data_size=1000)
	Y_data2 .= Y_data2 .* 2.0 .- 1.0
end

# ╔═╡ 76d280c5-3838-409c-b016-b4311ba6f071
let
	weights = SVM.train_linear(X_data2, Y_data2, 10.0, learning_rate=0.05, max_iter=1000, random_weights=false)
	JuTools.compute_accuracy(SVM.predict(X_data2, weights), Y_data2)
end

# ╔═╡ 9c94a40f-2dd0-4222-b7fd-4cb83fda0080
let
	weights = SVM.train_linear(X_data2, Y_data2, 10.0, learning_rate=0.05, max_iter=1000, random_weights=false)
	X_test_1 = 0:120
	X_test_2 = 0:120
	f_test1(X1, X2) = begin
   		SVM.predict([X1, X2], weights)[1] 
	end
	plot_X1 = [X_data2[i, 1] for i in 1:(size(X_data2)[1]) if Y_data2[i] == -1.0]
	plot_Y1 = [X_data2[i, 2] for i in 1:(size(X_data2)[1]) if Y_data2[i] == -1.0]
	plot_X2 = [X_data2[i, 1] for i in 1:(size(X_data2)[1]) if Y_data2[i] == 1.0]
	plot_Y2 = [X_data2[i, 2] for i in 1:(size(X_data2)[1]) if Y_data2[i] == 1.0]
	Plots.gr()
	Plots.contour(X_test_1, X_test_2, f_test1, leg=false, fill=true, c=Plots.cgrad(:matter, rev=true, categorical=false), linewidth=0, background_color=Plots.RGB(0.2, 0.2, 0.2))
	Plots.scatter!(plot_X1, plot_Y1, leg=false, c="red")
	Plots.scatter!(plot_X2, plot_Y2, leg=false, c="blue")
	Plots.xlims!((0.0, 120.0))
	Plots.ylims!((0.0, 120.0))
end

# ╔═╡ e1788d22-2d0d-4bfd-962e-500d5ba67eab
let
	weights = SVM.train_linear(X_data2, Y_data2, 10.0, learning_rate=0.05, max_iter=1000, random_weights=false)
	X_test_1 = 0:120
	X_test_2 = 0:120
	f_test2(X1, X2) = begin
   		abs(SVM.predict_proba([X1, X2], weights)[1])
	end
	plot_X1 = [X_data2[i, 1] for i in 1:(size(X_data2)[1]) if Y_data2[i] == -1.0]
	plot_Y1 = [X_data2[i, 2] for i in 1:(size(X_data2)[1]) if Y_data2[i] == -1.0]
	plot_X2 = [X_data2[i, 1] for i in 1:(size(X_data2)[1]) if Y_data2[i] == 1.0]
	plot_Y2 = [X_data2[i, 2] for i in 1:(size(X_data2)[1]) if Y_data2[i] == 1.0]
	Plots.gr()
	Plots.contour(X_test_1, X_test_2, f_test2, leg=false, fill=true, c=Plots.cgrad(:matter, rev=false, categorical=false), linewidth=0, background_color=Plots.RGB(0.2, 0.2, 0.2))
	Plots.scatter!(plot_X1, plot_Y1, leg=false, c="red")
	Plots.scatter!(plot_X2, plot_Y2, leg=false, c="blue")
	plot_X3 = collect(0:120)
	plot_Y3 = -(weights.b .+ (weights.w[1] .* plot_X3)) ./ weights.w[2]
	Plots.plot!(plot_X3, plot_Y3, c="purple", linewidth=5, alpha=0.5)
	Plots.xlims!((0.0, 120.0))
	Plots.ylims!((0.0, 120.0))
end

# ╔═╡ 5e6de600-114c-495f-80a6-494565971fe2
begin
	weights_linear = SVM.train(X_data2, Y_data2, 1.0, kernel="linear", gamma="scale", coef=0.0, degree=2.0, verbose=false)
	string("Linear Kernel Accuracy: ", JuTools.compute_accuracy(SVM.predict(X_data2, X_data2, Y_data2, weights_linear), Y_data2))
end

# ╔═╡ b53fea06-c0bf-4cab-99fc-394a2f6ca0bd
begin
	weights_rbf = SVM.train(X_data2, Y_data2, 1.0, kernel="rbf", gamma="scale", coef=0.0, degree=2.0, verbose=false)
	string("RBF Kernel Accuracy: ", JuTools.compute_accuracy(SVM.predict(X_data2, X_data2, Y_data2, weights_rbf), Y_data2))
end

# ╔═╡ 82f44cd6-c1a4-4ad6-9d6c-484e4909aedc
begin
	weights_poly = SVM.train(X_data2, Y_data2, 1.0, kernel="polynomial", gamma="scale", coef=0.0, degree=2.0, verbose=false)
	string("Polynomial Kernel (degree=2) Accuracy: ", JuTools.compute_accuracy(SVM.predict(X_data2, X_data2, Y_data2, weights_poly), Y_data2))
end

# ╔═╡ af2ec3ff-3468-4954-b5b7-b35ee45f7833
begin
	weights_poly_3 = SVM.train(X_data2, Y_data2, 1.0, kernel="polynomial", gamma="scale", coef=0.0, degree=3.0, verbose=false)
	string("Polynomial Kernel (degree=3) Accuracy: ", JuTools.compute_accuracy(SVM.predict(X_data2, X_data2, Y_data2, weights_poly_3), Y_data2))
end

# ╔═╡ d0bd3eb0-b6ea-47a9-959e-e385e033cd49
begin
	weights_sigmoid = SVM.train(X_data2, Y_data2, 10.0, kernel="sigmoid", gamma="scale", coef=0.0, degree=2.0, verbose=false)
	string("Sigmoid Kernel Accuracy: ", JuTools.compute_accuracy(SVM.predict(X_data2, X_data2, Y_data2, weights_sigmoid), Y_data2))
end

# ╔═╡ 1390261d-f5e1-4b55-9865-78450a944885
let
	# plot the data
	X_test_1 = 0:120
	X_test_2 = 0:120
	plot_X1 = [X_data2[i, 1] for i in 1:(size(X_data2)[1]) if Y_data2[i] == -1.0]
	plot_Y1 = [X_data2[i, 2] for i in 1:(size(X_data2)[1]) if Y_data2[i] == -1.0]
	plot_X2 = [X_data2[i, 1] for i in 1:(size(X_data2)[1]) if Y_data2[i] == 1.0]
	plot_Y2 = [X_data2[i, 2] for i in 1:(size(X_data2)[1]) if Y_data2[i] == 1.0]

	X_test_linear = hcat(vec(repeat(reshape(X_test_1, :, 1), 1, 121)'), vec(repeat(X_test_2, 121, 1)))
	X_test_linear = abs.(reshape(SVM.predict_proba(X_test_linear, X_data2, Y_data2, weights_linear), 121, :))

	Plots.gr()
	plot_linear = Plots.contour(X_test_1, X_test_2, X_test_linear, fill=true, c=Plots.cgrad(:matter, rev=false, categorical=false), linewidth=0, background_color=Plots.RGB(0.2, 0.2, 0.2))
	Plots.scatter!(plot_X1, plot_Y1, leg=false, c="red")
	Plots.scatter!(plot_X2, plot_Y2, leg=false, c="blue")
	Plots.xlims!((0.0, 120.0))
	Plots.ylims!((0.0, 120.0))
	Plots.title!("SVM (Linear Kernel)")
end

# ╔═╡ dfde4be0-4d84-42b7-96eb-9cfa75976947
let
	# plot the data
	X_test_1 = 0:120
	X_test_2 = 0:120
	plot_X1 = [X_data2[i, 1] for i in 1:(size(X_data2)[1]) if Y_data2[i] == -1.0]
	plot_Y1 = [X_data2[i, 2] for i in 1:(size(X_data2)[1]) if Y_data2[i] == -1.0]
	plot_X2 = [X_data2[i, 1] for i in 1:(size(X_data2)[1]) if Y_data2[i] == 1.0]
	plot_Y2 = [X_data2[i, 2] for i in 1:(size(X_data2)[1]) if Y_data2[i] == 1.0]

	X_test_rbf = hcat(vec(repeat(reshape(X_test_1, :, 1), 1, 121)'), vec(repeat(X_test_2, 121, 1)))
X_test_rbf = abs.(reshape(SVM.predict_proba(X_test_rbf, X_data2, Y_data2, weights_rbf), 121, :))

	Plots.gr()
	plot_linear = Plots.contour(X_test_1, X_test_2, X_test_rbf, fill=true, c=Plots.cgrad(:matter, rev=false, categorical=false), linewidth=0, background_color=Plots.RGB(0.2, 0.2, 0.2))
	Plots.scatter!(plot_X1, plot_Y1, leg=false, c="red")
	Plots.scatter!(plot_X2, plot_Y2, leg=false, c="blue")
	Plots.xlims!((0.0, 120.0))
	Plots.ylims!((0.0, 120.0))
	Plots.title!("SVM (RBF Kernel)")
end

# ╔═╡ d2a3a80f-5096-440c-9cdd-bd6aa80d81f5
let
	# plot the data
	X_test_1 = 0:120
	X_test_2 = 0:120
	plot_X1 = [X_data2[i, 1] for i in 1:(size(X_data2)[1]) if Y_data2[i] == -1.0]
	plot_Y1 = [X_data2[i, 2] for i in 1:(size(X_data2)[1]) if Y_data2[i] == -1.0]
	plot_X2 = [X_data2[i, 1] for i in 1:(size(X_data2)[1]) if Y_data2[i] == 1.0]
	plot_Y2 = [X_data2[i, 2] for i in 1:(size(X_data2)[1]) if Y_data2[i] == 1.0]

	X_test_sigmoid = hcat(vec(repeat(reshape(X_test_1, :, 1), 1, 121)'), vec(repeat(X_test_2, 121, 1)))
X_test_sigmoid = abs.(reshape(SVM.predict_proba(X_test_sigmoid, X_data2, Y_data2, weights_sigmoid), 121, :))

	Plots.gr()
	plot_linear = Plots.contour(X_test_1, X_test_2, X_test_sigmoid, fill=true, c=Plots.cgrad(:matter, rev=false, categorical=false), linewidth=0, background_color=Plots.RGB(0.2, 0.2, 0.2))
	Plots.scatter!(plot_X1, plot_Y1, leg=false, c="red")
	Plots.scatter!(plot_X2, plot_Y2, leg=false, c="blue")
	Plots.xlims!((0.0, 120.0))
	Plots.ylims!((0.0, 120.0))
	Plots.title!("SVM (Sigmoid Kernel)")
end

# ╔═╡ f45c2f60-45e9-48f4-b81d-ea7111111f5c
let
	# plot the data
	X_test_1 = 0:120
	X_test_2 = 0:120
	plot_X1 = [X_data2[i, 1] for i in 1:(size(X_data2)[1]) if Y_data2[i] == -1.0]
	plot_Y1 = [X_data2[i, 2] for i in 1:(size(X_data2)[1]) if Y_data2[i] == -1.0]
	plot_X2 = [X_data2[i, 1] for i in 1:(size(X_data2)[1]) if Y_data2[i] == 1.0]
	plot_Y2 = [X_data2[i, 2] for i in 1:(size(X_data2)[1]) if Y_data2[i] == 1.0]

	X_test_poly = hcat(vec(repeat(reshape(X_test_1, :, 1), 1, 121)'), vec(repeat(X_test_2, 121, 1)))
X_test_poly = abs.(reshape(SVM.predict_proba(X_test_poly, X_data2, Y_data2, weights_poly), 121, :))
	
	Plots.gr()
	plot_linear = Plots.contour(X_test_1, X_test_2, X_test_poly, fill=true, c=Plots.cgrad(:matter, rev=false, categorical=false), linewidth=0, background_color=Plots.RGB(0.2, 0.2, 0.2))
	Plots.scatter!(plot_X1, plot_Y1, leg=false, c="red")
	Plots.scatter!(plot_X2, plot_Y2, leg=false, c="blue")
	Plots.xlims!((0.0, 120.0))
	Plots.ylims!((0.0, 120.0))
	Plots.title!("SVM (Polynomial Kernel, degree=2)")
end

# ╔═╡ ad5bb232-ed6b-416c-8472-a15a25c65b5b
let
	# plot the data
	X_test_1 = 0:120
	X_test_2 = 0:120
	plot_X1 = [X_data2[i, 1] for i in 1:(size(X_data2)[1]) if Y_data2[i] == -1.0]
	plot_Y1 = [X_data2[i, 2] for i in 1:(size(X_data2)[1]) if Y_data2[i] == -1.0]
	plot_X2 = [X_data2[i, 1] for i in 1:(size(X_data2)[1]) if Y_data2[i] == 1.0]
	plot_Y2 = [X_data2[i, 2] for i in 1:(size(X_data2)[1]) if Y_data2[i] == 1.0]

	X_test_poly = hcat(vec(repeat(reshape(X_test_1, :, 1), 1, 121)'), vec(repeat(X_test_2, 121, 1)))
X_test_poly = abs.(reshape(SVM.predict_proba(X_test_poly, X_data2, Y_data2, weights_poly_3), 121, :))
	
	Plots.gr()
	plot_linear = Plots.contour(X_test_1, X_test_2, X_test_poly, fill=true, c=Plots.cgrad(:matter, rev=false, categorical=false), linewidth=0, background_color=Plots.RGB(0.2, 0.2, 0.2))
	Plots.scatter!(plot_X1, plot_Y1, leg=false, c="red")
	Plots.scatter!(plot_X2, plot_Y2, leg=false, c="blue")
	Plots.xlims!((0.0, 120.0))
	Plots.ylims!((0.0, 120.0))
	Plots.title!("SVM (Polynomial Kernel, degree=3)")
end

# ╔═╡ Cell order:
# ╟─b9dae3f2-5db7-4465-9c1e-f95df29fd48c
# ╠═db715d40-c599-11eb-0b28-f5c7b076c6ab
# ╠═27defb3d-f5fb-4334-a441-50f7a347819d
# ╠═fc31cf17-5e20-4130-b02c-9c2bbf9d3b72
# ╠═cc3033c8-4c74-4d28-ad2c-344246bbc827
# ╠═3c0bdafc-b98d-47c8-8b9f-69a7b94c5d85
# ╠═b4f64a60-b8a7-4487-8897-6bd4efa37c76
# ╠═71ed19eb-602f-4b21-a874-3c0cd6ef19d7
# ╠═de69ceeb-c9fb-4038-bf3d-2c08430e973c
# ╠═e18d49b7-3ef2-4d5e-ae80-83af5836277f
# ╟─43aff583-73af-482a-82be-f3823fd40799
# ╠═6e7a9a46-e0da-442d-8f45-6b5089f6d953
# ╠═394fb389-532f-41a7-a387-22d86e1c8e3e
# ╠═7ab6dbb1-194b-4816-8ad6-52abe6a7a80f
# ╠═3bdb1ccd-18fe-4c91-ab1d-2e149de06097
# ╠═fcc3d16f-1545-40ca-b87d-df959fb25ed4
# ╠═869aacb1-74bd-4b8c-99f9-416732c4e783
# ╠═76d280c5-3838-409c-b016-b4311ba6f071
# ╠═9c94a40f-2dd0-4222-b7fd-4cb83fda0080
# ╠═e1788d22-2d0d-4bfd-962e-500d5ba67eab
# ╠═5e6de600-114c-495f-80a6-494565971fe2
# ╠═b53fea06-c0bf-4cab-99fc-394a2f6ca0bd
# ╠═82f44cd6-c1a4-4ad6-9d6c-484e4909aedc
# ╠═af2ec3ff-3468-4954-b5b7-b35ee45f7833
# ╠═d0bd3eb0-b6ea-47a9-959e-e385e033cd49
# ╠═1390261d-f5e1-4b55-9865-78450a944885
# ╠═dfde4be0-4d84-42b7-96eb-9cfa75976947
# ╠═d2a3a80f-5096-440c-9cdd-bd6aa80d81f5
# ╠═f45c2f60-45e9-48f4-b81d-ea7111111f5c
# ╠═ad5bb232-ed6b-416c-8472-a15a25c65b5b
