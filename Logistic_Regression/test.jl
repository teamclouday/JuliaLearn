### A Pluto.jl notebook ###
# v0.14.7

using Markdown
using InteractiveUtils

# ╔═╡ a11a5250-c5a3-11eb-2cb2-65a0eb1d220d
md"""
### Test Code for Module LogReg
"""

# ╔═╡ cff452a8-1123-4228-bdba-a6613e02c5bd
module script include("script.jl") end

# ╔═╡ a54d8027-2db7-4b0e-8794-f614e6b5ad50
LogReg = script.LogReg

# ╔═╡ 65be2db3-70fb-4cc3-ae2d-d53d9c94740f
import Plots

# ╔═╡ bfefa9b3-ecfb-496c-9176-9882672541d8
import Random

# ╔═╡ 394a6ebc-7544-499b-ad97-7b4452b1555e
begin
# generate linearly separated random data
X_data = Random.rand(0.0:0.5:100.0, (800, 2)) # first column X-axis, second column Y-axis
Y_data = Array{Int64}(undef, size(X_data)[1])
linear_func(x) = 2 * x - 3
for i = 1:size(X_data)[1]
    if X_data[i, 2] > linear_func(X_data[i, 1]) + randn()*20
        Y_data[i] = 0
    else
        Y_data[i] = 1
    end
end
size(X_data), size(Y_data)
end

# ╔═╡ 389887b7-2f1e-4dda-90c0-40f42f1eb5e7
# scale data
X_data_scaled = LogReg.scale(X_data);

# ╔═╡ 649f29fe-9f96-4be3-bff7-dfbbf0c1700f
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

# ╔═╡ c7df3c2c-54d8-4177-ba02-4d0098087cf6
let
# run logistic regression
weights = LogReg.train(X_data_scaled, Y_data, learning_rate=0.1, max_iter=10, early_stop=false)
predictions = LogReg.predict(X_data_scaled, weights)
string("Accuracy: ", LogReg.accuracy(predictions, Y_data))
end

# ╔═╡ c4cdf49e-3929-4393-a9fc-5db66cd2a812
let
# increase iteractions
weights = LogReg.train(X_data_scaled, Y_data, learning_rate=0.1, max_iter=100, early_stop=false)
predictions = LogReg.predict(X_data_scaled, weights)
string("Accuracy: ", LogReg.accuracy(predictions, Y_data))
end

# ╔═╡ 5e698890-39fb-47dc-a975-53d6c8e716a6
let
# increase iteractions
weights = LogReg.train(X_data_scaled, Y_data, learning_rate=0.1, max_iter=1000, early_stop=false)
predictions = LogReg.predict(X_data_scaled, weights)
string("Accuracy: ", LogReg.accuracy(predictions, Y_data))
end

# ╔═╡ 6bb2b4a7-2f56-4d42-a5ac-29333a69abba
let
# increase iteractions
weights = LogReg.train(X_data_scaled, Y_data, learning_rate=0.1, max_iter=10000, early_stop=false)
predictions = LogReg.predict(X_data_scaled, weights)
string("Accuracy: ", LogReg.accuracy(predictions, Y_data))
end

# ╔═╡ 50f6b449-6e24-48c1-8d33-dc13c1fb5b9c
let
# plot comparison graph
plot_X1 = [X_data[i, 1] for i in 1:(size(X_data)[1]) if Y_data[i] == 0]
plot_Y1 = [X_data[i, 2] for i in 1:(size(X_data)[1]) if Y_data[i] == 0]
plot_X2 = [X_data[i, 1] for i in 1:(size(X_data)[1]) if Y_data[i] == 1]
plot_Y2 = [X_data[i, 2] for i in 1:(size(X_data)[1]) if Y_data[i] == 1]
plot_X3 = [i for i in 0:100]
plot_Y3 = [linear_func(m) for m in plot_X3]
Plots.gr()
Plots.scatter(plot_X1, plot_Y1, c="red", background_color=Plots.RGB(0.2, 0.2, 0.2), label="")
Plots.scatter!(plot_X2, plot_Y2, c="blue", label="")
Plots.plot!(plot_X3, plot_Y3, c="orange", linewidth=5, label="Actual")
accuracy = Array{Float64}(undef, 4)
for (i, iternum) in enumerate([10, 100, 1000, 10000])
    weights = LogReg.train(X_data_scaled, Y_data, learning_rate=0.1, max_iter=iternum)
    accuracy[i] = LogReg.accuracy(LogReg.predict(X_data_scaled, weights), Y_data)
    @assert size(weights)[1] == 3
    plot_Y3_ = -(weights[3] .+ (weights[1] .* plot_X3)) ./ weights[2]
    Plots.plot!(plot_X3, plot_Y3_, palette=:lightrainbow, linewidth=3, label="iter=$iternum", alpha=0.5)
end
Plots.xlims!((0.0, 100.0))
Plots.ylims!((0.0, 100.0))
end

# ╔═╡ Cell order:
# ╟─a11a5250-c5a3-11eb-2cb2-65a0eb1d220d
# ╠═cff452a8-1123-4228-bdba-a6613e02c5bd
# ╠═a54d8027-2db7-4b0e-8794-f614e6b5ad50
# ╠═65be2db3-70fb-4cc3-ae2d-d53d9c94740f
# ╠═bfefa9b3-ecfb-496c-9176-9882672541d8
# ╠═394a6ebc-7544-499b-ad97-7b4452b1555e
# ╠═389887b7-2f1e-4dda-90c0-40f42f1eb5e7
# ╠═649f29fe-9f96-4be3-bff7-dfbbf0c1700f
# ╠═c7df3c2c-54d8-4177-ba02-4d0098087cf6
# ╠═c4cdf49e-3929-4393-a9fc-5db66cd2a812
# ╠═5e698890-39fb-47dc-a975-53d6c8e716a6
# ╠═6bb2b4a7-2f56-4d42-a5ac-29333a69abba
# ╠═50f6b449-6e24-48c1-8d33-dc13c1fb5b9c
