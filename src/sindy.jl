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
    X = zeros(Float64, f * C, M)
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
dd_prob = DiscreteDataDrivenProblem(X, t)

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

sparse_func = get_coefficients(first(get_results(res)))

function library_row(feats::AbstractVector{<:Real}, exps::Vector{Vector{Int}})
    P = 1 + length(exps)
    θ = ones(Float64, P)
    for (k, exp) in enumerate(exps)
        v = 1.0
        @inbounds for i in exp
            v *= Float64(feats[i])
        end
        θ[k + 1] = v
    end
    return θ
end

function step_sparse(
    x::CuArray{Float32,4},
    kernel::CuArray{Float32,4},
    channel_n::Int,
    Ξ::Matrix{Float64},
    exps::Vector{Vector{Int}},
)
    y = NeuralCellularAutomata.perceive(x, kernel, channel_n)
    y_cpu = Array(y)
    W, H, n_feat, B = size(y_cpu)
    C = size(Ξ, 2)
    P = size(Ξ, 1)
    dx = zeros(Float32, W, H, C, B)
    @inbounds for b in 1:B, j in 1:H, i in 1:W
        feats = view(y_cpu, i, j, :, b)
        θ = library_row(feats, exps)
        @assert length(θ) == P
        for c in 1:C
            s = 0.0
            for p in 1:P
                s += θ[p] * Ξ[p, c]
            end
            dx[i, j, c, b] = Float32(s)
        end
    end
    return x .+ CuArray(dx)
end

step_sparse(x, kernel, channel_n, sparse_func, exps)
function eval_sparse_rollout!(
    out_dir::String,
    run_dir::String,
    model::CAModel,
    kernel::CuArray{Float32,4},
    channel_n::Int,
    vc::Int,
    Ξ::Matrix{Float64},
    exps::Vector{Vector{Int}},
    labels::Vector{String},
    frames::Vector{<:CuArray{Float32,3}},
    frame_source::String,
    hp;
    max_degree::Int,
    threshold::Float64,
)
    n_frames = length(frames)
    n_frames >= 2 || error("need at least two target frames")
    obs_guess = resolve_obs_steps(hp, n_frames)
    obs_steps_all = obs_guess[1:n_frames]

    n_train = Int(floor(TRAIN_RATIO * n_frames))
    train_idx = 1:n_train
    test_idx = (n_train + 1):n_frames
    train_obs = obs_steps_all[train_idx]
    test_obs = obs_steps_all[test_idx]
    train_frames = frames[train_idx]
    test_frames = frames[test_idx]

    W, H = size(frames[1], 1), size(frames[1], 2)
    x0_single = CUDA.zeros(Float32, W, H, channel_n)
    x0_single[:, :, 1:vc] .= frames[1][:, :, 1:vc]
    x0 = repeat(reshape(x0_single, W, H, channel_n, 1), 1, 1, 1, BATCH_SIZE)

    K_full = maximum(obs_steps_all)
    @info "SPARSE ROLLOUT EVAL @ fire_rate=$(EVAL_FIRE_RATE)"
    @info "  frames=$frame_source  n_frames=$n_frames  obs=$(obs_steps_all)"
    @info "  IC = first frame visible channels + zero hidden"
    @info "  train/test = $n_train/$(n_frames - n_train)  K_full=$K_full"

    obs_positions = Dict(t => i for (i, t) in enumerate(obs_steps_all))
    captured = Vector{typeof(x0)}(undef, n_frames)
    blew_up = false
    let state = copy(x0)
        haskey(obs_positions, 0) && (captured[obs_positions[0]] = copy(state))
        for t in 1:K_full
            state = step_sparse(state, kernel, channel_n, Ξ, exps)
            if !all(isfinite, Array(state))
                @warn "sparse state non-finite at CA step $t"
                blew_up = true
                # keep going with whatever we have so metrics still write
                for tt in t:K_full
                    haskey(obs_positions, tt) || continue
                    idx = obs_positions[tt]
                    captured[idx] = copy(state)
                end
                break
            end
            haskey(obs_positions, t) && (captured[obs_positions[t]] = copy(state))
        end
        for i in 1:n_frames
            if !isassigned(captured, i)
                captured[i] = copy(state)
            end
        end
    end

    ckpt_mse = Float32[loss_fn(captured[i], frames[i]; vc = vc) for i in 1:n_frames]
    ckpt_re = Float32[sqrt(ckpt_mse[i] / norm_one(frames[i]; vc = vc)) for i in 1:n_frames]

    # keep rolling past the last frame vs that last target (persistence check)
    final_frame = frames[end]
    final_loss = ckpt_mse[end]
    after_loss_log = Float32[]
    let state = copy(captured[end])
        for _ in 1:K_full
            state = step_sparse(state, kernel, channel_n, Ξ, exps)
            push!(after_loss_log, loss_fn(state, final_frame; vc = vc))
            !all(isfinite, Array(state)) && break
        end
    end
    after_loss = isempty(after_loss_log) ? NaN32 : mean(after_loss_log)
    re_final = ckpt_re[end]
    re_after = sqrt(after_loss / norm_one(final_frame; vc = vc))
    re_train = sqrt(mean(ckpt_mse[train_idx]) / norm_vec(train_frames; vc = vc))
    re_test = sqrt(mean(ckpt_mse[test_idx]) / norm_vec(test_frames; vc = vc))
    re_full = sqrt(mean(ckpt_mse) / norm_vec(frames; vc = vc))

    tag = "1.0"
    @info "SPARSE RE TRAIN=$re_train | TEST=$re_test | FULL=$re_full | FINAL=$re_final | AFTER=$re_after"

    # same fields as the NCA read_fr1.0.txt so they're easy to diff
    read_path = joinpath(out_dir, "read_fr$(tag).txt")
    open(read_path, "w") do io
        println(io, Dates.now())
        println(io, "SPARSE POLY ROLLOUT (not neural NCA)")
        println(io, "ckpt / fit from = $(joinpath(run_dir, "checkpoints", "traj_final.jld2"))")
        println(io, "FRAME_SOURCE = $frame_source")
        println(io, "INITIAL CONDITION = first frame visible channels; hidden channels zero")
        println(io, "TRAIN/TEST SPLIT = $(n_train)/$(n_frames) train, $(length(test_idx)) test")
        println(io, "TRAIN OBS = $(train_obs)")
        println(io, "TEST OBS = $(test_obs)")
        println(io, "DELTA_T / OBS = $(obs_steps_all)")
        println(io, "CHANNEL_N = $(channel_n)  VC = $(vc)")
        println(io, "MAX_DEGREE = $(max_degree)  THRESHOLD = $(threshold)")
        println(io, "LIBRARY TERMS = $(labels)")
        println(io, "EVAL FIRE RATE = $(EVAL_FIRE_RATE)  (deterministic; sparse has no stochastic mask)")
        println(io, "BLEW_UP = $(blew_up)")
        println(io, "FINAL LOSS (at last frame) = $final_loss")
        println(io, "AFTER LOSS (persistence past last frame vs last tgt) = $after_loss")
        println(io, "CKPT MSE (full rollout) = $(ckpt_mse)")
        println(io, "CKPT RE  (full rollout) = $(ckpt_re)")
        println(io, "RE TRAIN = $re_train")
        println(io, "RE TEST = $re_test")
        println(io, "RE FULL = $re_full")
        println(io, "RE FINAL = $re_final")
        println(io, "RE AFTER = $re_after")
    end

    # also drop a copy next to the NCA read file
    cp(read_path, joinpath(run_dir, "read_sparse_fr$(tag).txt"); force = true)

    snap_path = joinpath(out_dir, "full_traj_train_test_sparse_fr$(tag).png")
    save_sparse_trajectory_snapshot(captured, frames, re_test, snap_path; vc = vc)

    split_t = 0.5f0 * (train_obs[end] + test_obs[1])
    re_y = Float64.(ckpt_re)
    p_re = plot(
        obs_steps_all, re_y;
        title = "Sparse poly RE vs CA time (fr=$(tag))",
        xlabel = "CA step",
        ylabel = "relative error",
        lw = 2,
        color = :gray,
        label = "RE(t)",
        size = (900, 560),
        legend = :topleft,
    )
    scatter!(p_re, train_obs, re_y[train_idx]; color = :steelblue, markersize = 6, label = "train")
    scatter!(p_re, test_obs, re_y[test_idx]; color = :crimson, markersize = 6, label = "test")
    vline!(p_re, [split_t]; color = :black, ls = :dash, lw = 2, label = "train | test")
    re_plot_path = joinpath(out_dir, "relative_error_vs_time_sparse_fr$(tag).png")
    savefig(p_re, re_plot_path)

    re_y_log = max.(re_y, eps(Float64))
    p_re_log = plot(
        obs_steps_all, re_y_log;
        title = "Sparse poly RE vs CA time (log y, fr=$(tag))",
        xlabel = "CA step",
        ylabel = "relative error",
        yscale = :log10,
        lw = 2,
        color = :gray,
        label = "RE(t)",
        size = (900, 560),
        legend = :topleft,
    )
    scatter!(p_re_log, train_obs, re_y_log[train_idx]; color = :steelblue, markersize = 6, label = "train")
    scatter!(p_re_log, test_obs, re_y_log[test_idx]; color = :crimson, markersize = 6, label = "test")
    vline!(p_re_log, [split_t]; color = :black, ls = :dash, lw = 2, label = "train | test")
    re_log_path = joinpath(out_dir, "relative_error_vs_time_logy_sparse_fr$(tag).png")
    savefig(p_re_log, re_log_path)

    video_steps = K_full
    video_path = joinpath(out_dir, "teacher_forced_sparse_vs_nca_fr$(tag).mp4")
    frame_dir = joinpath(out_dir, "teacher_forced_sparse_vs_nca_frames")
    save_teacher_forced_sparse_vs_nca_video(
        model, x0[:, :, :, 1], kernel, channel_n, Ξ, exps;
        n_steps = video_steps,
        frame_dir = frame_dir,
        out_path = video_path,
        vc = vc,
    )

    return (;
        read_path, snap_path, re_plot_path, re_log_path, video_path,
        re_train, re_test, re_full, re_final, re_after, blew_up,
    )
end