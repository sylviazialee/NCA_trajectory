function to_rgb(x::AbstractArray{Float32}; vc::Int = 3)
    if vc == 1
        # Monochrome: channel 1 is the gray output; no alpha compositing.
        if ndims(x) == 4
            g = clamp.(x[:, :, 1:1, :], 0f0, 1f0)
            return repeat(g, 1, 1, 3, 1)
        elseif ndims(x) == 3
            g = clamp.(x[:, :, 1:1], 0f0, 1f0)
            return repeat(g, 1, 1, 3)
        else
            error("Expected 3D or 4D array")
        end
    elseif vc == 2
        # Premultiplied gray (ch1) + alpha (ch2); composite over white like RGBA.
        if ndims(x) == 4
            a = clamp.(x[:, :, 2:2, :], 0f0, 1f0)
            pg = x[:, :, 1:1, :]
            rgb = repeat(pg, 1, 1, 3, size(x, 4))
            return 1f0 .- a .+ rgb
        elseif ndims(x) == 3
            a = clamp.(x[:, :, 2:2], 0f0, 1f0)
            pg = x[:, :, 1:1]
            rgb = repeat(pg, 1, 1, 3)
            return 1f0 .- a .+ rgb
        else
            error("Expected 3D or 4D array")
        end
    elseif vc==3
        #RGB only NCA, no alpha!
        if ndims(x) == 4# for batch sizes
            g = clamp.(x[:, :, 1:3, :], 0f0, 1f0)
            return g
        elseif ndims(x) == 3
            g = clamp.(x[:, :, 1:3], 0f0, 1f0)
            return g
        else
            error("WRONG!")
        end
    else
        # RGBA: composite premultiplied RGB (ch1–3) over white using channel 4 (alpha).
        C = size(x, ndims(x) == 4 ? 3 : 3)
        life_ch = min(4, C)
        if ndims(x) == 4
            a   = clamp.(x[:, :, life_ch:life_ch, :], 0f0, 1f0)
            rgb = x[:, :, 1:3, :]
        elseif ndims(x) == 3
            a   = clamp.(x[:, :, life_ch:life_ch], 0f0, 1f0)
            rgb = x[:, :, 1:3]
        else
            error("Expected 3D or 4D array")
        end
        return 1f0 .- a .+ rgb
    end
end

function save_rgb_png(path::String, rgb::AbstractArray{<:Real,3};
                      title::String = "",
                      subtitle::String = "",
                      fig_size::Tuple{Int,Int} = (420, 420))
    mkpath(dirname(path))
    rgb_f32 = Float32.(rgb)
    if any(!isfinite, rgb_f32)
        # A diverged model can produce NaN/Inf cell states; GR's color conversion
        # crashes on those (InexactError converting NaN to UInt64), so sanitize
        # before plotting rather than losing the whole training run.
        @warn "save_rgb_png: non-finite values in image data at $path; replacing with 0"
        rgb_f32 = map(v -> isfinite(v) ? v : 0f0, rgb_f32)
    end
    img = permutedims(clamp.(rgb_f32, 0f0, 1f0), (2, 1, 3))
    rgb_mat = Matrix{RGB{Float64}}(undef, size(img, 1), size(img, 2))
    @inbounds for y in axes(img, 1), x in axes(img, 2)
        rgb_mat[y, x] = RGB(Float64(img[y, x, 1]), Float64(img[y, x, 2]), Float64(img[y, x, 3]))
    end

    if isempty(subtitle)
        p = plot(
            rgb_mat;
            title,
            showaxis = false,
            grid = false,
            ticks = false,
            aspect_ratio = 1,
            size = fig_size,
        )
    else
        p = plot(
            rgb_mat;
            title,
            subtitle,
            showaxis = false,
            grid = false,
            ticks = false,
            aspect_ratio = 1,
            size = fig_size,
        )
    end
    savefig(p, path)
    return path
end

function write_rgb_ppm(path::String, rgb::AbstractArray{<:Real,3}; scale::Int = 8)
    @assert size(rgb, 3) == 3 "Expected an RGB image with shape (W, H, 3)"
    mkpath(dirname(path))

    W, H, _ = size(rgb)
    rgb_clamped = clamp.(Float32.(rgb), 0f0, 1f0)

    open(path, "w") do io
        write(io, "P6\n$(W * scale) $(H * scale)\n255\n")
        row = Vector{UInt8}(undef, W * scale * 3)
        for y in 1:H
            for _ in 1:scale
                k = 1
                for x in 1:W
                    r = UInt8(round(Int, 255 * rgb_clamped[x, y, 1]))
                    g = UInt8(round(Int, 255 * rgb_clamped[x, y, 2]))
                    b = UInt8(round(Int, 255 * rgb_clamped[x, y, 3]))
                    for _ in 1:scale
                        row[k] = r
                        row[k+1] = g
                        row[k+2] = b
                        k += 3
                    end
                end
                write(io, row)
            end
        end
    end

    return path
end

function to_hidden(x::AbstractArray{Float32,3}, channel_num::Int)
    g = clamp.(x[:, :, channel_num:channel_num], 0f0, 1f0)
    return repeat(g, 1, 1, 3)
end

function write_hidden_ppm(path::String, hidden::AbstractArray{<:Real,3}; scale::Int = 8)
    return write_rgb_ppm(path, hidden; scale)
end

function side_by_side_rgb(left::AbstractArray{<:Real,3}, right::AbstractArray{<:Real,3}; gap::Int = 4)
    @assert size(left) == size(right) "Images must have matching dimensions"
    W, H, _ = size(left)
    img = ones(Float32, 2W + gap, H, 3)
    img[1:W, :, :] .= Float32.(left)
    img[W+gap+1:end, :, :] .= Float32.(right)
    return img
end

function save_loss_plots(loss_log::AbstractVector{<:Real}, output_dir::String; run_id::String = run_timestamp())
    isempty(loss_log) && return nothing
    mkpath(output_dir)

    iterations = 1:length(loss_log)
    ys = Float64.(loss_log)
    loss_path = joinpath(output_dir, "loss_vs_iterations.png")
    loss_log_path = joinpath(output_dir, "loss_vs_iterations_logy.png")

    p_lin = plot(
        iterations,
        ys;
        title = "NCA training loss",
        xlabel = "iteration",
        ylabel = "loss",
        legend = false,
        lw = 2,
        color = :steelblue,
        size = (900, 560),
    )
    savefig(p_lin, loss_path)

    p_log = plot(
        iterations,
        max.(ys, eps(Float64));
        title = "NCA training loss (log y)",
        xlabel = "iteration",
        ylabel = "loss",
        yscale = :log10,
        legend = false,
        lw = 2,
        color = :darkgreen,
        size = (900, 560),
    )
    savefig(p_log, loss_log_path)

    @info "Saved loss plots: $loss_path, $loss_log_path"
    return loss_path, loss_log_path
end

# this one is NOT working unfortunately
function plot_multi(
    l1::AbstractVector{<:Real}, 
    l2::AbstractVector{<:Real},
    l3::AbstractVector{<:Real}, 
    l4::AbstractVector{<:Real}, 
    l5::AbstractVector{<:Real}, 
    output_dir::String;
    run_id::String = run_timestamp()
    )

    mkpath(output_dir)

    iterations = 1:length(l1)
    y1 = Float64.(l1)
    y2 = Float64.(l2)
    y3 = Float64.(l3)
    y4 = Float64.(l4)
    y5 = Float64.(l5)
    loss_path = joinpath(output_dir, "loss_vs_iterations.png")
    loss_log_path = joinpath(output_dir, "loss_vs_iterations_logy.png")

    p_log = plot(
        iterations,
        max.(y1, eps(Float64));
        title = "NCA training loss (log y)",
        xlabel = "iteration",
        ylabel = "loss",
        yscale = :log10,
        legend = false,
        lw = 2,
        color = :red,
        size = (900, 560),
        label = "fr02"
    )
    plot!(p_log, iterations, max.(y2, eps(Float64)); color = :orange, label = "fr04")
    plot!(p_log, iterations, max.(y3, eps(Float64)); color = :green, label = "fr06")
    plot!(p_log, iterations, max.(y4, eps(Float64)); color = :blue, label = "fr08")
    plot!(p_log, iterations, max.(y5, eps(Float64)); color = :purple, label = "fr10")
    savefig(p_log, loss_log_path)

    @info "Saved loss plots: $loss_path, $loss_log_path"
    return loss_path, loss_log_path
end

function rollout_state(
    model::CAModel,
    seed::CuArray{Float32,3},
    kernel::CuArray{Float32,4},
    n_steps::Int;
    fire_rate::Float32 = 1f0,
)
    x = reshape(seed, size(seed)..., 1)
    for _ in 1:n_steps
        x = step(model, x, kernel; fire_rate)
    end
    return x
end

function save_training_snapshot(
    model::CAModel,
    seed::CuArray{Float32,3},
    kernel::CuArray{Float32,4},
    target::CuArray{Float32,3},
    step_i::Int,
    loss_val::Real,
    snapshot_dir::String;
    run_id::String = run_timestamp(),
    n_steps::Int,
    fire_rate::Float32 = 1f0,
    scale::Int = 8,
    vc::Int = 3,
)
    x = rollout_state(model, seed, kernel, n_steps; fire_rate)
    snapshot_loss = Float32(loss_fn(x, target; vc))
    pred_rgb = to_rgb(Array(x)[:, :, :, 1]; vc)
    target_rgb = to_rgb(Array(target); vc)
    snapshot = side_by_side_rgb(pred_rgb, target_rgb)
    path = joinpath(snapshot_dir, @sprintf("train_step_%05d.png", step_i))
    save_rgb_png(
        path,
        snapshot;
        title = @sprintf("step %05d", step_i),
        subtitle = @sprintf("train %.5f | snapshot %.5f", loss_val, snapshot_loss),
    )
    @info "Saved training snapshot: $path  train_loss = $(round(loss_val; sigdigits=4))  snapshot_loss = $(round(snapshot_loss; sigdigits=4))"
    return path
end

"""Roll `x0` out to step `K = maximum(obs_steps)`, capturing the state at each obs step
(mirrors the two-substep update used in `train_trajectory!`, but with a plain `Vector`
since this isn't run under Zygote)."""
function rollout_trajectory_states(
    model::CAModel,
    x0::CuArray{Float32,4},
    kernel::CuArray{Float32,4},
    obs_steps::Vector{Int};
    fire_rate::Float32,
)
    K = maximum(obs_steps)
    obs_positions = Dict(t => i for (i, t) in enumerate(obs_steps))
    captured = Vector{CuArray{Float32,4}}(undef, length(obs_steps))

    x = x0
    haskey(obs_positions, 0) && (captured[obs_positions[0]] = x)
    for t in 1:K
        x = step(model, x, kernel; fire_rate = 1f0)
        x = step(model, x, kernel; fire_rate)
        haskey(obs_positions, t) && (captured[obs_positions[t]] = x)
    end
    return captured
end

function save_trajectory_snapshot(
    model::CAModel,
    x0::CuArray{Float32,4},
    kernel::CuArray{Float32,4},
    targets::Vector{<:CuArray{Float32,3}},
    obs_steps::Vector{Int},
    step_i::Int,
    loss_val::Real,
    snapshot_dir::String;
    fire_rate::Float32,
    vc::Int = 3,
)
    captured = rollout_trajectory_states(model, x0, kernel, obs_steps; fire_rate)

    panels = [
        side_by_side_rgb(
            to_rgb(Array(captured[i])[:, :, :, 1]; vc),
            to_rgb(Array(targets[i]); vc),
        )
        for i in eachindex(obs_steps)
    ]

    gap = 6
    W, H, _ = size(panels[1])
    n = length(panels)
    canvas = ones(Float32, n * W + (n - 1) * gap, H, 3)
    for (i, p) in enumerate(panels)
        x0c = (i - 1) * (W + gap) + 1
        canvas[x0c:x0c+W-1, :, :] .= p
    end

    path = joinpath(snapshot_dir, @sprintf("traj_step_%05d.png", step_i))
    save_rgb_png(
        path, canvas;
        title = @sprintf("step %05d | loss %.5f", step_i, loss_val),
        fig_size = (n * 420, 420),
    )
    @info "Saved trajectory snapshot: $path  loss = $(round(loss_val; sigdigits=4))"
    return path
end

function save_rollout_video(
    model::CAModel,
    seed::CuArray{Float32,3},
    kernel::CuArray{Float32,4};
    n_steps::Int,
    frame_dir::String,
    out_path::String,
    fps::Int = 30,
    fire_rate::Float32 = 1f0,
    scale::Int = 8,
    vc::Int = model.visible_channels,
)
    rm(frame_dir; force=true, recursive=true)
    mkpath(frame_dir)

    x = reshape(seed, size(seed)..., 1)
    progress = Progress(n_steps + 1; desc = "Writing final rollout frames: ")

    for i in 0:n_steps
        rgb = to_rgb(Array(x)[:, :, :, 1]; vc)
        frame_path = joinpath(frame_dir, @sprintf("frame_%05d.ppm", i))
        write_rgb_ppm(frame_path, rgb; scale)
        next!(progress; showvalues = [(:frame, i), (:total_frames, n_steps + 1)])

        if i < n_steps
            x = step(model, x, kernel; fire_rate)
        end
    end

    ffmpeg = FFMPEG_jll.ffmpeg()
    run(`$ffmpeg -y -framerate $fps -i $(joinpath(frame_dir, "frame_%05d.ppm")) -pix_fmt yuv420p $out_path`)
    @info "Saved final rollout video: $out_path"
    return out_path
end

function save_multi_seed_videos(
    model::CAModel,
    W::Int,
    H::Int,
    channel_n::Int,
    kernel::CuArray{Float32,4};
    seed_offsets::Vector{Tuple{Int,Int}},
    n_steps::Int,
    output_dir::String,
    run_id::String,
    fps::Int = 30,
    fire_rate::Float32 = 1f0,
    scale::Int = 8,
)
    cx0, cy0 = W ÷ 2 + 1, H ÷ 2 + 1
    paths = String[]

    for (i, (dx, dy)) in enumerate(seed_offsets)
        cx = clamp(cx0 + dx, 1, W)
        cy = clamp(cy0 + dy, 1, H)
        seed = make_seed(W, H, channel_n; cx, cy, visible_channels = model.visible_channels)
        label = @sprintf("seed_%02d_x%+d_y%+d", i, dx, dy)
        path = save_rollout_video(
            model,
            seed,
            kernel;
            n_steps,
            frame_dir = joinpath(output_dir, "$(label)_frames"),
            out_path = joinpath(output_dir, "$(label)_rollout_2x.mp4"),
            fps,
            fire_rate,
            scale,
        )
        push!(paths, path)
    end

    return paths
end

function run_ca(
    model::CAModel,
    seed::CuArray{Float32,3},
    kernel::CuArray{Float32,4},
    n_steps::Int;
    fire_rate::Float32 = model.fire_rate,
)
    x = reshape(seed, size(seed)..., 1)
    for _ in 1:n_steps
        x = step(model, x, kernel; fire_rate)
    end
    return Array(x)
end

function save_hidden_video(
    model::CAModel,
    seed::CuArray{Float32,3},
    kernel::CuArray{Float32,4};
    n_steps::Int,
    frame_dir::String,
    out_path::String,
    fps::Int = 30,
    fire_rate::Float32 = 1f0,
    scale::Int = 8,
    vc::Int = model.visible_channels,
    pad_choice::String = "zer",
)
    rm(frame_dir; force=true, recursive=true)
    mkpath(frame_dir)

    x = reshape(seed, size(seed)..., 1)
    hidden_channels = (vc+1):size(x, 3)
    progress = Progress(n_steps + 1; desc = "Writing final hidden channel display: ")

    for i in 0:n_steps
        for channel_num in hidden_channels
            hidden = to_hidden(Array(x)[:, :, :, 1], channel_num)
            frame_path = joinpath(frame_dir, @sprintf("hidden_channel_%d_%05d.ppm", channel_num, i))
            write_hidden_ppm(frame_path, hidden; scale)
        end
        next!(progress; showvalues = [(:frame, i), (:total_frames, n_steps + 1)])

        if i < n_steps
            x = step(model, x, kernel; fire_rate, pad_choice)
        end
    end

    ffmpeg = FFMPEG_jll.ffmpeg()
    base, ext = splitext(out_path)
    for channel_num in hidden_channels
        channel_out_path = "$(base)_ch$(channel_num)$(ext)"
        run(`$ffmpeg -y -framerate $fps -i $(joinpath(frame_dir, "hidden_channel_$(channel_num)_%05d.ppm")) -pix_fmt yuv420p $channel_out_path`)
    end
    @info "Saved final hidden channel video(s) for channels $(hidden_channels) with base path $out_path"
    return out_path
end