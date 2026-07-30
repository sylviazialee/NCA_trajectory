include(joinpath(@__DIR__, "..", "src", "NeuralCellularAutomata.jl"))
using .NeuralCellularAutomata
include(joinpath(@__DIR__, "target_prep.jl"))
using CUDA, FileIO, ImageIO
using Dates

train_set = ARGS[1]
#image_set = ARGS[2]
num = parse(Int64, ARGS[2])
fr = 0.5
run_id = run_timestamp()
data_dir = "8020_$(train_set)_subs$(num)"
train_steps = 3000
delta = 20


step_080 = 16


filters = [:id, :sobel_x, :sobel_y]
# 1. load your RGBA frames at each observation time as Float32 (W,H,4)
#    (your own loading logic here)
frames = [load_target_rgba("$(train_set)/snapshot_00.png"), 
          load_target_rgba("$(train_set)/snapshot_01.png"),
          load_target_rgba("$(train_set)/snapshot_02.png"),
          load_target_rgba("$(train_set)/snapshot_03.png"),
          load_target_rgba("$(train_set)/snapshot_04.png"), 
          load_target_rgba("$(train_set)/snapshot_05.png"),
          load_target_rgba("$(train_set)/snapshot_06.png"),
          load_target_rgba("$(train_set)/snapshot_07.png"),
          load_target_rgba("$(train_set)/snapshot_08.png"),
          load_target_rgba("$(train_set)/snapshot_09.png"), 
          load_target_rgba("$(train_set)/snapshot_10.png"),
          load_target_rgba("$(train_set)/snapshot_11.png"),
          load_target_rgba("$(train_set)/snapshot_12.png"),
          load_target_rgba("$(train_set)/snapshot_13.png"), 
          load_target_rgba("$(train_set)/snapshot_14.png"),
          load_target_rgba("$(train_set)/snapshot_15.png"),
          load_target_rgba("$(train_set)/snapshot_16.png"),
          load_target_rgba("$(train_set)/snapshot_17.png"), 
          load_target_rgba("$(train_set)/snapshot_18.png"),
          load_target_rgba("$(train_set)/snapshot_19.png")]   # Vector of CuArray{Float32,3}

# 2. map physical times -> CA step indices (your choice of time scale)
obs_steps = delta .* (0:step_080-1)  # e.g. t=0 → step 0, t=1.0 → step 25, t=2.4 → step 60

# define subseq indices from obs_steps
if num == 1
    subseqs = [(1:16)]
elseif num == 2
    subseqs = [(1:8), (1:16)]
elseif num == 3
    subseqs = [(1:3), (1:10), (1:16)]
else
    subseqs = [(1:16)]
end
numseq = length(subseqs)

path = joinpath(data_dir, run_id, "subseq$(numseq)", "checkpoints","traj_final.jld2")
# extract subseq of frames and obs_steps
subframes = [frames[subseqs[i]] for i in 1:numseq]
substeps = [obs_steps[subseqs[i]] for i in 1:numseq]
models = Vector{CAModel}(undef, numseq)
hps = Vector{TrajectoryHParams}(undef, numseq)
loss_info = Vector(undef, numseq)
re_loss_info = Vector(undef, numseq)

# initialize hp for the full trajectory (all obs_steps)

# 3. build x0: RGBA from frame[1], hidden channels = 0


# run step one first 
hp = TrajectoryHParams(
    channel_n  = 16,
    obs_steps  = substeps[1],
    batch_size = 4,
    train_steps = train_steps,
    fire_rate = fr,
    visible_channels = 3,
    filters = filters,
)
hps[1] = hp
current_hp = hp
current_frames = subframes[1]

# make seed BASED OFF OF CURRENT SUBFRAMES, W and H do not need to be re-defined
W, H = size(current_frames[1], 1), size(current_frames[1], 2)
x0_single = CUDA.zeros(Float32, W, H, current_hp.channel_n)
x0_single[:, :, 1:current_hp.visible_channels] .= current_frames[1]            # seed RGBA from data
x0 = repeat(reshape(x0_single, W, H, current_hp.channel_n, 1), 1, 1, 1, current_hp.batch_size)

completed = falses(numseq)

try
    models[1], loss_info[1], re_loss_info[1] = main_subseq(x0, subframes[1]; hp = hps[1], run_id = "$(run_id)/subseq1", data_dir = data_dir)
    completed[1] = true
    @show re_loss_info[1]
catch e
    @warn "subsequence 1 failed; no subsequence results will be available" exception = (e, catch_backtrace())
end

if completed[1]
    for i in 2:numseq
        # pre update
        current_frames = subframes[i] # update frames
        # current hp parameters
        current_hp = TrajectoryHParams(
            channel_n  = 16,
            obs_steps  = substeps[i],
            batch_size = 4,
            train_steps = train_steps,
            fire_rate = fr,
            visible_channels = 3,
            filters = filters,
        )
        hps[i] = current_hp # update hps vector of hp

        # make seed BASED OFF OF CURRENT SUBFRAMES
        x0_single = CUDA.zeros(Float32, W, H, current_hp.channel_n)
        x0_single[:, :, 1:current_hp.visible_channels] .= current_frames[1]            # seed RGBA from data
        x0 = repeat(reshape(x0_single, W, H, current_hp.channel_n, 1), 1, 1, 1, current_hp.batch_size)

        # update models, loss info
        try
            models[i], loss_info[i], re_loss_info[i] = main_subseq(x0, subframes[i]; past_model = models[i-1], hp = hps[i], run_id = "$(run_id)/subseq$(i)", data_dir = data_dir)
            completed[i] = true
            @show re_loss_info[i]
        catch e
            @warn "subsequence $i failed; stopping remaining subsequences" exception = (e, catch_backtrace())
            break
        end
    end
end


output_dir = joinpath(data_dir, run_id)
mkpath(output_dir)

open(joinpath(output_dir, "read.txt"), "w") do io
    println(io, Dates.now())
    println(io, "TRAIN: " * train_set)
    println(io, "TRAINING FIRE RATE = $(hp.fire_rate)")
    for i in 1:numseq
        if !completed[i]
            println(io, "SUBSEQ $i: FAILED / NOT COMPLETED")
            continue
        end
        sub_loss_log, sub_final_loss, sub_after_loss = loss_info[i]
        re_sub_traj, re_sub_fin, re_sub_after = sqrt.(re_loss_info[i])
        println(io, "SUBSEQ $i TRAJ LOSS = $(sub_loss_log[end])")
        println(io, "SUBSEQ $i FINAL LOSS = $sub_final_loss")
        println(io, "SUBSEQ $i AFTER LOSS = $sub_after_loss")
        println(io, "SUBSEQ $i RE TRAJ LOSS = $re_sub_traj")
        println(io, "SUBSEQ $i RE FINAL LOSS = $re_sub_fin")
        println(io, "SUBSEQ $i RE AFTER LOSS = $re_sub_after")
    end
end

# 1. load test frames
#=
all_test_frames = [load_target_rgba("$(image_set)/snapshot_00.png"), 
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
          load_target_rgba("$(image_set)/snapshot_19.png")]   # Vector of CuArray{Float32,3}

=#
# 1. rebuild the perception kernel from hp (not saved in the checkpoint)
#spread_path = "/home/szl129/nca/firerate_spread/20260717_120244/checkpoints/traj_final.jld2"
#flag_path = "/home/szl129/nca/frenchflag/20260716_030958/checkpoints/traj_final.jld2"

#test_frames = all_test_frames[step_080:length(all_test_frames)]
test_frames = frames[step_080:length(all_test_frames)]
#path = spread_path

# The training try/catch above lets a failed subsequence through, but then the
# checkpoint `path` points at was never written -- bail out with a clear reason
# instead of a confusing "no such file" from load_model.
if !completed[numseq]
    @warn "training did not complete through subseq$(numseq); skipping test phase" completed
    exit(0)
end

model, hp = load_model(path)
kernel = NeuralCellularAutomata.build_perception_kernel(hp.channel_n; filters = hp.filters)

# 0002. map physical times -> CA step indices (your choice of time scale)
# test_frames[1] is the seed (t=0), so steps are RELATIVE to it: [0, 20, 40, 60]
new_obs_steps = delta .* (0:length(test_frames)-1)

# 2. build x0 from your test image
frame_0 = test_frames[1]   # CuArray{Float32,3}, (W,H,4)
W, H = size(frame_0, 1), size(frame_0, 2)
x0 = CUDA.zeros(Float32, W, H, hp.channel_n, 1)
x0[:, :, 1:hp.visible_channels, :] .= reshape(frame_0[:, :, 1:hp.visible_channels], W, H, hp.visible_channels, 1)

# 3. run it
obs_positions = Dict(t => i for (i, t) in enumerate(new_obs_steps))  # step -> slot in captured
n_obs = length(new_obs_steps)
n_steps = 2 * maximum(new_obs_steps)
captured = Vector{typeof(x0)}(undef, n_obs) #keep track of every checkpoint

x = x0

if haskey(obs_positions, 0)
    captured[obs_positions[0]] = x
end

for t in 1:n_steps
    global x = NeuralCellularAutomata.step(model, x, kernel; fire_rate = 1.0)

    if haskey(obs_positions, t)
        captured[obs_positions[t]] = x
    end
end

loss = loss_alg2(copy(captured), test_frames; vc = hp.visible_channels)

# 4. view the result
run_id = run_timestamp()
output_dir = joinpath("test_output", run_id)
mkpath(output_dir)

rgb = NeuralCellularAutomata.to_rgb(Array(x)[:, :, :, 1]; vc = hp.visible_channels)
NeuralCellularAutomata.save_rgb_png(
    joinpath(output_dir, "final_state_step$(n_steps).png"), rgb;
    title = "$(train_set) -> $(image_set) | step $(n_steps)",
)

open(joinpath(output_dir, "read.txt"), "w") do io
    #CHANGE THIS PER TRIAL
    println(io, "TRAIN: " * train_set * " | TESTED: " * image_set)
    println(io, path)
    println(io, "Trajectory loss of test frames: $loss")
end

save_rollout_video(
    model,
    x0[:, :, :, 1],
    kernel;
    n_steps = n_steps,
    frame_dir = joinpath(output_dir, "$(image_set)_test_frames_rgb"),
    out_path = joinpath(output_dir, "$(train_set)_to_$(image_set)_rgb.mp4"),
    fps = 30,
    fire_rate = 1.0f0,
    scale = 8,
)

save_hidden_video(
    model,
    x0[:, :, :, 1],
    kernel;
    n_steps = n_steps,
    frame_dir = joinpath(output_dir, "$(image_set)_test_frames_hidden"),
    out_path = joinpath(output_dir, "$(train_set)_to_$(image_set)_hidden.mp4"),
    fps = 30,
    fire_rate = 1.0f0,
    scale = 8,
    vc = hp.visible_channels,
    pad_choice = "zer",
)
@info("Hidden vids acquired")