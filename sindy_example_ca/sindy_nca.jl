# NCA-specific helpers for sparse symbolic regression.
#
# Expects sindy_lib.jl already included (library_row, etc.).
# Expects NeuralCellularAutomata loaded, plus these consts from the app:
#   EVAL_FIRE_RATE, ACTIVE_SAMPLE_FRACTION, ACTIVE_POOL_FRACTION,
#   TRAIN_RATIO, BATCH_SIZE, VIDEO_FPS, VIDEO_SCALE, RUN_DIR
#

# names like id_c0, sobel_c1, … — same order as build_perception_kernel
"""Feature names for each (filter, channel) slot in the perception vector."""
function perception_feature_names(filters::Vector{Symbol}, channel_n::Int)
    names = String[]
    for c in 0:(channel_n - 1)
        for f in filters
            push!(names, "$(f)_c$(c)")
        end
    end
    return names
end

"""Reaction polys + linear Laplacians only (no id×Lap products).

Represents dx_c = f_c(u) + sum_j D_cj Δu_j. Excludes Laplacian products,
which can destabilize free rollouts off the sampled NCA trajectory.
"""
function reaction_diffusion_exponents(
    filters::Vector{Symbol}, channel_n::Int, max_degree::Int,
)
    id_offset = findfirst(==(:id), filters)
    lap_offset = findfirst(==(:laplacian), filters)
    id_offset === nothing && error("reaction-diffusion library requires an :id filter")
    lap_offset === nothing && error("reaction-diffusion library requires a :laplacian filter")

    n_filters = length(filters)
    id_features = [(c - 1) * n_filters + id_offset for c in 1:channel_n]
    lap_features = [(c - 1) * n_filters + lap_offset for c in 1:channel_n]
    exps = Vector{Vector{Int}}()

    function rec_ids!(cur::Vector{Int}, start::Int, deg_left::Int)
        if deg_left == 0
            push!(exps, copy(cur))
            return
        end
        for k in start:length(id_features)
            push!(cur, id_features[k])
            rec_ids!(cur, k, deg_left - 1)
            pop!(cur)
        end
    end
    for degree in 1:max_degree
        rec_ids!(Int[], 1, degree)
    end
    append!(exps, ([i] for i in lap_features))
    return exps
end

"""Reaction polys + linear / state-dependent diffusion (from SINDY_new).

Covers
    dx_c = f_c(u) + sum_j (D_cj + sum_i A_cij u_i) Δu_j
i.e. id monomials, plain Laplacians, and id*Laplacian — no Laplacian powers.
"""
function state_dependent_diffusion_exponents(
    filters::Vector{Symbol}, channel_n::Int, max_degree::Int,
)
    exps = reaction_diffusion_exponents(filters, channel_n, max_degree)
    id_offset = findfirst(==(:id), filters)
    lap_offset = findfirst(==(:laplacian), filters)
    n_filters = length(filters)
    id_features = [(c - 1) * n_filters + id_offset for c in 1:channel_n]
    lap_features = [(c - 1) * n_filters + lap_offset for c in 1:channel_n]
    for id_feature in id_features, lap_feature in lap_features
        push!(exps, [id_feature, lap_feature])
    end
    return exps
end

# neural update only (Dense1 → ReLU → Dense2), no mask
function dx_from_perception(model::CAModel, y::CuArray{Float32,4})
    h = NNlib.conv(y, model.dense1_w) .+ reshape(model.dense1_b, 1, 1, :, 1)
    h = relu.(h)
    return NNlib.conv(h, model.dense2_w) .+ reshape(model.dense2_b, 1, 1, :, 1)
end

"""Grab (perception, dx) samples while rolling the NCA at fire_rate=1.

At each step we store y and the neural dx, then do x ← x+dx. Half the cells
are random; half come from big-|dx| spots so fronts aren't swamped by quiet
background. Returns Y (N×n_feat) and dX (N×C) on the CPU.
"""
function sample_nca_trajectory_perception_and_dx(
    model::CAModel,
    kernel::CuArray{Float32,4},
    seed::CuArray{Float32,3};
    n_samples::Int,
    n_steps::Int,
    rng::AbstractRNG = Random.default_rng(),
)
    C = model.channel_n
    n_feat = size(kernel, 4)  # filters × channels
    size(seed, 3) == C || error("seed has $(size(seed,3)) channels; model has $C")
    W, H = size(seed, 1), size(seed, 2)
    cells_per_step = W * H
    n_times = n_steps + 1

    # spread samples across time (including t=0)
    counts = fill(div(n_samples, n_times), n_times)
    remainder = rem(n_samples, n_times)
    if remainder > 0
        extra_idx = unique(round.(Int, range(1, n_times; length = remainder)))
        # range() can collide; pad until we have enough distinct slots
        k = 1
        while length(extra_idx) < remainder
            k ∉ extra_idx && push!(extra_idx, k)
            k = mod1(k + 1, n_times)
        end
        counts[extra_idx[1:remainder]] .+= 1
    end

    Y = Matrix{Float64}(undef, n_samples, n_feat)
    dX = Matrix{Float64}(undef, n_samples, C)
    row = 0
    x = reshape(copy(seed), W, H, C, 1)

    for t in 0:n_steps
        y = NeuralCellularAutomata.perceive(x, kernel, C)
        dx = dx_from_perception(model, y)

        n_take = counts[t + 1]
        if n_take > 0
            y_cpu = Array(y)
            dx_cpu = Array(dx)
            n_active = round(Int, ACTIVE_SAMPLE_FRACTION * n_take)
            n_uniform = n_take - n_active

            uniform_idx = rand(rng, 1:cells_per_step, n_uniform)

            # bias toward cells that are actually moving
            scores = vec(sum(abs2, dx_cpu; dims = 3))
            pool_size = clamp(
                round(Int, ACTIVE_POOL_FRACTION * cells_per_step),
                max(1, n_active),
                cells_per_step,
            )
            active_pool = partialsortperm(scores, 1:pool_size; rev = true)
            active_idx = n_active > 0 ?
                rand(rng, active_pool, n_active) :
                Int[]
            cell_idx = shuffle!(rng, vcat(uniform_idx, active_idx))
            for q in cell_idx
                i = mod1(q, W)
                j = cld(q, W)
                row += 1
                Y[row, :] .= @view y_cpu[i, j, :, 1]
                dX[row, :] .= @view dx_cpu[i, j, :, 1]
            end
        end

        if t < n_steps
            x = x .+ dx  # same as NCA at fr=1
            all(isfinite, Array(x)) ||
                error("original NCA fire_rate=1 trajectory became non-finite at step $(t+1)")
        end
    end
    row == n_samples || error("internal sampling error: collected $row / $n_samples")
    return Y, dX
end

# --- sparse CA + rollout vs u_traj ---

"""One step of the poly CA: perceive → library → Ξ, then add to x."""
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

function resolve_traj_path(source::AbstractString)
    p = abspath(source)
    isfile(p) && return p
    isdir(p) || error("u_traj source not found: $source")
    for name in ("u_traj.jld2", "file_u.jld2", "u.jld2")
        candidate = joinpath(p, name)
        isfile(candidate) && return candidate
    end
    jlds = sort(filter(f -> endswith(lowercase(f), ".jld2"), readdir(p)))
    isempty(jlds) && error("No .jld2 file in u_traj directory: $p")
    return joinpath(p, first(jlds))
end

"""Load u_traj into GPU frames (W,H,1) each."""
function load_u_traj_frames(source::AbstractString)
    path = resolve_traj_path(source)
    raw = JLD2.load(path)
    u = if haskey(raw, "u_traj")
        raw["u_traj"]
    elseif haskey(raw, "data")
        raw["data"]
    else
        # last resort: first 3-D array in the file
        found = nothing
        for value in values(raw)
            if value isa AbstractArray && ndims(value) == 3
                found = value
                break
            end
        end
        found === nothing && error("No 3-D trajectory in $path (tried u_traj/data)")
        found
    end
    ndims(u) == 3 || error("Expected u_traj shaped (W,H,T); got $(size(u))")
    W, H, T = size(u)
    T >= 2 || error("u_traj needs at least two frames; got T=$T")
    frames = [
        CuArray(Float32.(reshape(u[:, :, t], W, H, 1)))
        for t in 1:T
    ]
    clim = max(Float32(maximum(abs, u)), eps(Float32))
    NeuralCellularAutomata.set_visible_clim!(clim)
    @info "Loaded u_traj=$path size=$(size(u)); raw IC=u_traj[:,:,1]; clim=±$clim"
    return frames, path
end

"""Load snapshot_00.png … from REPO/IMAGE_SET (same as SINDY_new)."""
function load_image_set_frames(image_set::String, n_frames::Int; repo::AbstractString = REPO)
    frames = Vector{CuArray{Float32,3}}(undef, n_frames)
    for i in 0:(n_frames - 1)
        path = joinpath(repo, image_set, "snapshot_$(lpad(i, 2, '0')).png")
        isfile(path) || error("missing training frame: $path")
        frames[i + 1] = load_target_rgba(path)
    end
    return frames
end

function count_image_set_frames(image_set::String; repo::AbstractString = REPO, max_check::Int = 99)
    snap_dir = joinpath(repo, image_set)
    return count(i -> isfile(joinpath(snap_dir, "snapshot_$(lpad(i, 2, '0')).png")), 0:max_check)
end

function resolve_obs_steps(hp, n_frames::Int)
    if hasproperty(hp, :obs_steps) && hp.obs_steps isa AbstractVector
        steps = Vector{Int}(hp.obs_steps)
        length(steps) == n_frames || @warn("hp.obs_steps length $(length(steps)) ≠ n_frames=$n_frames; truncating/padding to frames")
        if length(steps) >= n_frames
            return steps[1:n_frames]
        end
    end
    # fall back to parsing Delta_T out of the run folder name
    Δ = 20
    m = match(r"Delta_T_(\d+)", RUN_DIR)
    m !== nothing && (Δ = parse(Int, m.captures[1]))
    return collect(Δ .* (0:(n_frames - 1)))
end

"""Side-by-side video: SINDy one-step vs NCA one-step from the same teacher state.

Both sides start from the NCA state each frame, so SINDy mistakes don't pile up
in the movie — you just see the local update quality.
"""
function save_teacher_forced_sparse_vs_nca_video(
    model::CAModel,
    seed::CuArray{Float32,3},
    kernel::CuArray{Float32,4},
    channel_n::Int,
    Ξ::Matrix{Float64},
    exps::Vector{Vector{Int}};
    n_steps::Int,
    frame_dir::String,
    out_path::String,
    fps::Int = VIDEO_FPS,
    scale::Int = VIDEO_SCALE,
    vc::Int,
)
    rm(frame_dir; force = true, recursive = true)
    mkpath(frame_dir)
    mkpath(dirname(out_path))

    teacher = reshape(copy(seed), size(seed)..., 1)
    progress = Progress(n_steps + 1; desc = "Writing teacher-forced SINDy|NCA frames: ")

    for i in 0:n_steps
        if i == 0
            sparse_next = copy(teacher)
            nca_next = copy(teacher)
        else
            # same starting state on both sides
            sparse_next = step_sparse(teacher, kernel, channel_n, Ξ, exps)
            nca_next = NeuralCellularAutomata.step(
                model, teacher, kernel; fire_rate = EVAL_FIRE_RATE,
            )
        end

        sparse_arr = Array(sparse_next)[:, :, :, 1]
        nca_arr = Array(nca_next)[:, :, :, 1]
        panel_clim = if vc == 1
            max(
                NeuralCellularAutomata._resolve_diverging_clim(@view(sparse_arr[:, :, 1]), nothing),
                NeuralCellularAutomata._resolve_diverging_clim(@view(nca_arr[:, :, 1]), nothing),
            )
        else
            nothing
        end
        canvas = NeuralCellularAutomata.side_by_side_rgb(
            NeuralCellularAutomata.to_rgb(sparse_arr; vc, clim = panel_clim),
            NeuralCellularAutomata.to_rgb(nca_arr; vc, clim = panel_clim),
        )
        panel_clim !== nothing &&
            (canvas = NeuralCellularAutomata.append_diverging_colorbar(canvas, panel_clim))
        frame_path = joinpath(frame_dir, @sprintf("frame_%05d.ppm", i))
        NeuralCellularAutomata.write_rgb_ppm(frame_path, canvas; scale)
        next!(progress; showvalues = [(:frame, i), (:total_frames, n_steps + 1)])

        if !all(isfinite, sparse_arr)
            @warn "teacher-forced SINDy one-step result became non-finite at step $i"
            break
        end
        if i > 0
            teacher = nca_next
        end
    end

    ffmpeg = FFMPEG_jll.ffmpeg()
    run(`$ffmpeg -y -framerate $fps -i $(joinpath(frame_dir, "frame_%05d.ppm")) -pix_fmt yuv420p $out_path`)
    @info "Saved teacher-forced SINDy(left) | NCA(right) one-step video: $out_path"
    return out_path
end

function save_sparse_trajectory_snapshot(
    captured::Vector,
    targets::Vector{<:AbstractArray{Float32,3}},
    re_test::Real,
    out_path::String;
    vc::Int,
    max_panels::Int = 24,
)
    n_all = length(targets)
    # don't blow up the strip if there are tons of checkpoints
    panel_idx = if n_all <= max_panels
        collect(1:n_all)
    else
        unique(round.(Int, range(1, n_all; length = max_panels)))
    end

    panels = map(panel_idx) do i
        pred_arr = Array(captured[i])[:, :, :, 1]
        tgt_arr = Array(targets[i])
        NeuralCellularAutomata.side_by_side_rgb(
            NeuralCellularAutomata.to_rgb(pred_arr; vc),
            NeuralCellularAutomata.to_rgb(tgt_arr; vc),
        )
    end

    gap = 6
    W, H, _ = size(panels[1])
    n_p = length(panels)
    canvas = ones(Float32, n_p * W + (n_p - 1) * gap, H, 3)
    for (i, p) in enumerate(panels)
        x0c = (i - 1) * (W + gap) + 1
        canvas[x0c:x0c+W-1, :, :] .= p
    end

    panel_px = clamp(8192 ÷ max(n_p, 1), 120, 420)
    title = @sprintf("sparse poly | RE_test %.5f | showing %d/%d ckpts", re_test, n_p, n_all)
    NeuralCellularAutomata.save_rgb_png(
        out_path, canvas;
        title,
        fig_size = (n_p * panel_px, panel_px),
    )
    @info "Saved sparse trajectory snapshot: $out_path"
    return out_path
end

"""Free-run the poly model from the same IC, score against target frames, dump plots/video."""
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
