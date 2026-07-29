include(joinpath(@__DIR__, "..", "src", "NeuralCellularAutomata.jl"))
using .NeuralCellularAutomata
using DataDrivenDiffEq
using DataDrivenSparse       
using Symbolics 
using CUDA, FileIO, ImageIO
using Dates

# 1. get model function
image_set = ARGS[1]
path = ARGS[2]
model, hp = load_model(path)
kernel = NeuralCellularAutomata.build_perception_kernel(hp.channel_n; filters = hp.filters)
data_dir = "sindy_$(image_set)"
run_id = run_timestamp()
output_dir = joinpath(data_dir, run_id)
mkpath(output_dir)
# this is step function, step func has 3 * 16 states x_1 thru x_48
#x_new = step(model, x, kernel; fire_rate = 1.0) # each id, sx, sy are c-dimensional, the number of total states n is 3 * 16 = 48

#initialize
i, j = 21, 21 # use one cell
W, H = 40, 40
x0 = NeuralCellularAutomata.make_seed(W, H, hp.channel_n; visible_channels = hp.visible_channels)  # initial conditions
M = 380 #some number of steps idk
f = length(hp.filters)
# rollout x for m steps

function rollout(model, kernel, x0, M, i, j, C, f)
    x = reshape(x0, size(x0)..., 1)
    X = zeros(Float32, f * C, M)
    for m in 1:M
        x = NeuralCellularAutomata.step(model, x, kernel; fire_rate = 1f0)
        y = NeuralCellularAutomata.perceive(x, kernel, C)
        X[:, m] = Array(y[i, j, :, 1])
        if !all(isfinite, X[:, m])
            error("diverged at step $m")
        end
    end
    return X
end

X = rollout(model, kernel, x0, M, i, j, hp.channel_n, f)

t = collect(1:M)

# 2. datadrivenproblem, get derivatives
dd_prob = DataDrivenProblem(X, t, CleanData())

# 3. build library

# define symbolic variables u1 to u48
@variables u[1:(f * hp.channel_n)]

# poly candidate functions up to degree 2

basis_functions = polynomial_basis(u, 1)
basis = Basis(basis_functions, u)

# 4. sindy part

# Use Sequential Thresholded Least Squares (STLSQ) with a sparsity cutoff of 0.1
optimizer = STLSQ(0.1)
res = solve(dd_prob, basis, optimizer)


# 5. print readable model




open(joinpath(output_dir, "read.txt"), "w") do io
    println(io, "Discovered Model:")
    println(io, result(res))

    println(io, "\nEstimated Parameter Values:")
    println(io, get_parameter_map(res))
end