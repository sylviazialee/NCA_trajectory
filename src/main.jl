function default_hparams()
    HParams(
        channel_n = 16,
        hidden_n = 128,
        target_size = 40,
        target_padding = 16,
        batch_size = 8,
        pool_size = 1024,
        train_steps = 8000,
        fire_rate = 0.5f0,
        damage_n = 3,
        use_pattern_pool = true,
    )
end

# target_path: .jld2 with `target`, or .png (see load_target_rgba)
function main(
    target_path::String;
    hp::HParams = default_hparams(),
    run_id::String = run_timestamp(),
    data_dir::String = "data",
    snapshot_every::Int = 50,
    snapshot_rollout_steps::Int = hp.max_steps,
    video_rollout_steps::Int = 2 * hp.max_steps,
    video_fps::Int = 30,
    image_scale::Int = 8,
    video_fire_rate::Float32 = 1f0,
    extra_seed_video_offsets::Vector{Tuple{Int,Int}} = Tuple{Int,Int}[],
)
    report_backend!()
    isfile(target_path) || error("bad target_path: $target_path")

    run_dir = joinpath(data_dir, run_id)
    output_dir = joinpath(run_dir, "outputs")
    checkpoint_dir = joinpath(run_dir, "checkpoints")
    mkpath(run_dir)

    model, loss_log = train!(
        hp,
        target_path;
        checkpoint_dir,
        output_dir,
        run_id,
        snapshot_every,
        snapshot_rollout_steps,
        video_rollout_steps,
        video_fps,
        image_scale,
        video_fire_rate,
        extra_seed_video_offsets,
    )
    @info "done, final loss $(loss_log[end])"
    model, loss_log
end

# x0/targets: build these yourself (see example_scripts/nca_traj_trial.jl) --
# unlike main(), there's no single target_path to load them from.
function main_trajectory(
    x0::CuArray{Float32,4},
    targets::Vector{<:CuArray{Float32,3}};
    hp::TrajectoryHParams = TrajectoryHParams(),
    weights::Vector{Float32} = ones(Float32, length(targets)),
    run_id::String = run_timestamp(),
    data_dir::String = "data",
    snapshot_every::Int = 100,
    video_rollout_steps::Int = 2 * maximum(hp.obs_steps),
    video_fps::Int = 30,
    video_fire_rate::Float32 = 1f0,
    image_scale::Int = 8,
)
    report_backend!()

    run_dir = joinpath(data_dir, run_id)
    output_dir = joinpath(run_dir, "outputs")
    checkpoint_dir = joinpath(run_dir, "checkpoints")
    mkpath(run_dir)

    model, loss_log = train_trajectory!(
        hp, x0, targets, weights;
        checkpoint_dir,
        output_dir,
        run_id,
        snapshot_every,
        video_rollout_steps,
        video_fps,
        video_fire_rate,
        image_scale,
    )
    
    #rerun model for length K
    K = maximum(hp.obs_steps)
    kernel = build_perception_kernel(hp.channel_n; filters = hp.filters)
    last_pred = x0 # building final frame from seed frame
    last_target = targets[end] # final target image of comparison
    check_num = length(targets)
    W, H, _, N = size(x0)
    vc = hp.visible_channels

    # generate final image
    for i in 1:K
        last_pred = step(model, last_pred, kernel; fire_rate = hp.fire_rate)
    end

    # loss taken at final checkpoint only
    loss_fin = loss_fn(last_pred, last_target; vc = vc)

    # set initial of rollout of Kth through 2Kth steps to last_pred
    after_pred = last_pred
    #captured = Vector{typeof(x0)}(undef, check_num)
    after_loss_log = Float32[]

    check_index = floor(Int, K/check_num)
    step_counter = 1
    for t in 1:K
        after_pred = step(model, after_pred, kernel; fire_rate = hp.fire_rate)
        push!(after_loss_log, loss_fn(after_pred, targets[end]; vc = vc))
    end

    loss_after = mean(after_loss_log)
    #loss_after = loss_alg2(copy(captured), targets) this was SO wrong this should be a vector of target[end] only
    @info "final loss: $loss_fin"
    @info "after loss: $loss_after"
    @info "after loss log:"
    @info after_loss_log
    save_loss_plots(after_loss_log, joinpath(output_dir, "after_loss"); run_id = run_id)

    @info "trajectory run finished, final TRAJECTORY loss $(loss_log[end])"
    #println("LOSS LOG AT FIRE RATE $(video_fire_rate)")
    println(loss_log)
    #return model, loss_log, loss_fin, loss_after

    loss_traj = loss_log[end]

    re_loss_after = loss_after/norm_one(last_target; vc = vc)
    re_loss_fin = loss_fin/norm_one(last_target; vc = vc)
    re_loss_traj = loss_traj/norm_vec(targets; vc = vc)

    loss_info = [loss_log, loss_fin, loss_after]
    re_loss_info = [re_loss_traj, re_loss_fin, re_loss_after]
    return model, loss_info, re_loss_info
end

function main_subseq(
    x0::CuArray{Float32,4},
    targets::Vector{<:CuArray{Float32,3}};
    hp::TrajectoryHParams = TrajectoryHParams(),
    weights::Vector{Float32} = ones(Float32, length(targets)),
    past_model::CAModel = CAModel(hp),
    run_id::String = run_timestamp(),
    data_dir::String = "data",
    snapshot_every::Int = 100,
    video_rollout_steps::Int = 2 * maximum(hp.obs_steps),
    video_fps::Int = 30,
    video_fire_rate::Float32 = 1f0,
    image_scale::Int = 8,
)
    report_backend!()

    run_dir = joinpath(data_dir, run_id)
    output_dir = joinpath(run_dir, "outputs")
    checkpoint_dir = joinpath(run_dir, "checkpoints")
    mkpath(run_dir)

    model, loss_log = train_subseq!(
        hp, x0, targets, weights;
        past_model,
        checkpoint_dir,
        output_dir,
        run_id,
        snapshot_every,
        video_rollout_steps,
        video_fps,
        video_fire_rate,
        image_scale,
    )
    
    #rerun model for length K
    K = maximum(hp.obs_steps)
    kernel = build_perception_kernel(hp.channel_n; filters = hp.filters)
    last_pred = x0 # building final frame from seed frame
    last_target = targets[end] # final target image of comparison
    check_num = length(targets)
    W, H, _, N = size(x0)
    vc = hp.visible_channels

    # generate final image
    for i in 1:K
        last_pred = step(model, last_pred, kernel; fire_rate = hp.fire_rate)
    end

    # loss taken at final checkpoint only
    loss_fin = loss_fn(last_pred, last_target; vc = vc)

    # set initial of rollout of Kth through 2Kth steps to last_pred
    after_pred = last_pred
    #captured = Vector{typeof(x0)}(undef, check_num)
    after_loss_log = Float32[]

    check_index = floor(Int, K/check_num)
    step_counter = 1
    for t in 1:K
        after_pred = step(model, after_pred, kernel; fire_rate = hp.fire_rate)
        push!(after_loss_log, loss_fn(after_pred, targets[end]; vc = vc))
    end

    loss_after = mean(after_loss_log)
    loss_traj = loss_log[end]

    @show re_loss_after = loss_after/norm_one(last_target; vc = vc)
    @show re_loss_fin = loss_fin/norm_one(last_target; vc = vc)
    @show re_loss_traj = loss_traj/norm_vec(targets; vc = vc)
    
    @info "final loss: $loss_fin"
    @info "after loss: $loss_after"
    @info "after loss log:"
    @info after_loss_log
    save_loss_plots(after_loss_log, joinpath(output_dir, "after_loss"); run_id = run_id)

    @info "trajectory run finished, final TRAJECTORY loss $(loss_traj)"
    #println("LOSS LOG AT FIRE RATE $(video_fire_rate)")
    println(loss_log)
    loss_info = [loss_log, loss_fin, loss_after]
    re_loss_info = [re_loss_traj, re_loss_fin, re_loss_after]
    return model, loss_info, re_loss_info
end