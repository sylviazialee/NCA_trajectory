
include(joinpath(@__DIR__, "sindy_lib.jl"))
include(joinpath(@__DIR__, "sindy_plots.jl"))
include(joinpath(@__DIR__, "sindy_nca.jl"))
include(joinpath(@__DIR__, "..", "src", "NeuralCellularAutomata.jl"))
using .NeuralCellularAutomata
using DataDrivenDiffEq
using DataDrivenSparse       
using Symbolics 
using CUDA, FileIO, ImageIO
using Dates
using Printf
using FFMPEG_jll

# 1. get model function
image_set = ARGS[1]
path = ARGS[2]
model, hp = load_model(path)
kernel = NeuralCellularAutomata.build_perception_kernel(hp.channel_n; filters = hp.filters)
data_dir = "sindy_$(image_set)"
run_id = run_timestamp()
output_dir = joinpath(data_dir, run_id)
mkpath(output_dir)

frames = [load_target_rgba("$(image_set)/snapshot_00.png"), 
          load_target_rgba("$(image_set)/snapshot_01.png"),
          load_target_rgba("$(image_set)/snapshot_02.png"),
          load_target_rgba("$(image_set)/snapshot_03.png"),
          load_target_rgba("$(image_set)/snapshot_04.png"), 
          load_target_rgba("$(image_set)/snapshot_05.png"),
          load_target_rgba("$(image_set)/snapshot_06.png"),
          load_target_rgba("$(image_set)/snapshot_07.png"),
          load_target_rgba("$(image_set)/snapshot_08.png"),
          load_target_rgba("$(image_set)/snapshot_09.png"), 
          load_target_rgba("$(image_set)/snapshot_10.png"),
          load_target_rgba("$(image_set)/snapshot_11.png"),
          load_target_rgba("$(image_set)/snapshot_12.png"),
          load_target_rgba("$(image_set)/snapshot_13.png"), 
          load_target_rgba("$(image_set)/snapshot_14.png"),
          load_target_rgba("$(image_set)/snapshot_15.png"),
          load_target_rgba("$(image_set)/snapshot_16.png"),
          load_target_rgba("$(image_set)/snapshot_17.png"), 
          load_target_rgba("$(image_set)/snapshot_18.png"),
          load_target_rgba("$(image_set)/snapshot_19.png")]

# this is step function, step func has 3 * 16 states x_1 thru x_48
#x_new = step(model, x, kernel; fire_rate = 1.0) # each id, sx, sy are c-dimensional, the number of total states n is 3 * 16 = 48
obs_steps = 20 .* (0:(length(frames)-1))
#initialize
i, j = 21, 21 # use one cell
W, H = 40, 40
x0 = NeuralCellularAutomata.make_seed(W, H, hp.channel_n; visible_channels = hp.visible_channels)  # initial conditions
M = 380 #some number of steps idk
f = length(hp.filters)
# rollout x for m steps

# Mirror the NCA's own structure: dx = MLP(perceive(x)), x_new = x + dx.
# So we regress the STATE INCREMENT (C-dim) on the PERCEPTION FEATURES (f*C-dim).
# Feats -> regressors, DX -> targets.
function rollout(model, kernel, x0, M, i, j, C, f)
    x = reshape(x0, size(x0)..., 1)
    Feats = zeros(Float64, f * C, M)
    DX    = zeros(Float64, C, M)
    for m in 1:M
        y = NeuralCellularAutomata.perceive(x, kernel, C)
        Feats[:, m] = Array(y[i, j, :, 1])
        x_next = NeuralCellularAutomata.step(model, x, kernel; fire_rate = 1f0)
        DX[:, m] = Array(x_next[i, j, :, 1]) .- Array(x[i, j, :, 1])
        if !all(isfinite, Feats[:, m]) || !all(isfinite, DX[:, m])
            error("diverged at step $m")
        end
        x = x_next
    end
    return Feats, DX
end

Feats, DX = rollout(model, kernel, x0, M, i, j, hp.channel_n, f)

t = collect(1:M)

# 2. datadrivenproblem: direct input->output map (features -> state increment)
dd_prob = DirectDataDrivenProblem(Feats, DX)

# 3. build library

# define symbolic variables u1 to u48
@variables u[1:(f * hp.channel_n)]

# poly candidate functions up to degree 2

basis_functions = polynomial_basis(u, 2)
basis = Basis(basis_functions, u)

#build exps, in the same way?
n_x, degree = f * hp.channel_n, 2
tuples = [t for t in Iterators.product([0:degree for _ in 1:n_x]...) if sum(t) <= degree]
# tuples[1] is the constant; library_row prepends it, so drop it
exps = [vcat((fill(i, t[i]) for i in 1:n_x)...) for t in tuples[2:end]]


# 4. sindy part

# Use Sequential Thresholded Least Squares (STLSQ) with a sparsity cutoff of 0.1
optimizer = STLSQ(0.067)
res = solve(dd_prob, basis, optimizer)


# 5. print readable model

result_basis = get_basis(res)

open(joinpath(output_dir, "read.txt"), "w") do io
    println(io, "Discovered Model:")
    println(io, result_basis)

    println(io, "\nEstimated Parameter Values:")
    println(io, get_parameter_map(result_basis))
end

# get_coefficients returns (n_targets x n_terms); step_sparse indexes Ξ[term, channel]
sparse_func = permutedims(Matrix{Float64}(DataDrivenSparse.get_coefficients(first(get_results(res)))))
@info "Ξ size (terms x channels) = $(size(sparse_func))  nonzeros = $(count(!iszero, sparse_func))"

# library_row is provided by sindy_lib.jl

"""
Roll the discovered sparse-polynomial map forward from `seed` and encode the
visible channels to an mp4. Mirrors `save_rollout_video` but substitutes
`step_sparse` for the neural `step`.
"""

W, H = size(frames[1], 1), size(frames[1], 2)
true_seed = CUDA.zeros(Float32, W, H, hp.channel_n)
true_seed[:, :, 1:hp.visible_channels] .= frames[1][:, :, 1:hp.visible_channels]

save_sparse_rollout_video(
    true_seed, kernel, hp.channel_n, sparse_func, exps;
    n_steps = M,
    frame_dir = joinpath(output_dir, "sparse_rollout_frames"),
    out_path = joinpath(output_dir, "sparse_rollout.mp4"),
    vc = hp.visible_channels,
)
