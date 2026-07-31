include(joinpath(@__DIR__, "..", "src", "NeuralCellularAutomata.jl"))
using .NeuralCellularAutomata
include(joinpath(@__DIR__, "..", "example_scripts", "target_prep.jl"))
using CUDA, FileIO, ImageIO
using Dates

# 1. load test frames
train_set = ARGS[1]
image_set = ARGS[2]
path = ARGS[3]
#=test_frames = [load_target_rgba("$(image_set)/snapshot_00.png"), 
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

test_frames =[load_target_rgba("$(image_set)/snapshot_00.png"),
            load_target_rgba("$(image_set)/snapshot_00.png"),
            load_target_rgba("$(image_set)/snapshot_00.png"),
            load_target_rgba("$(image_set)/snapshot_00.png"),
            load_target_rgba("$(image_set)/snapshot_00.png"),
            load_target_rgba("$(image_set)/snapshot_00.png"),
            load_target_rgba("$(image_set)/snapshot_00.png"),
            load_target_rgba("$(image_set)/snapshot_00.png"),
            load_target_rgba("$(image_set)/snapshot_00.png"),
            load_target_rgba("$(image_set)/snapshot_00.png"),
            load_target_rgba("$(image_set)/snapshot_00.png"),
            load_target_rgba("$(image_set)/snapshot_00.png"),
            load_target_rgba("$(image_set)/snapshot_00.png"),
            load_target_rgba("$(image_set)/snapshot_00.png"),
            load_target_rgba("$(image_set)/snapshot_00.png"),
            load_target_rgba("$(image_set)/snapshot_00.png"),
            load_target_rgba("$(image_set)/snapshot_00.png"),
            load_target_rgba("$(image_set)/snapshot_00.png"),
            load_target_rgba("$(image_set)/snapshot_00.png"),
            load_target_rgba("$(image_set)/snapshot_19.png")]

# 1. rebuild the perception kernel from hp (not saved in the checkpoint)
#spread_path = "/home/szl129/nca/firerate_spread/20260717_120244/checkpoints/traj_final.jld2"
#flag_path = "/home/szl129/nca/frenchflag/20260716_030958/checkpoints/traj_final.jld2"


#path = spread_path
model, hp = load_model(path)
kernel = NeuralCellularAutomata.build_perception_kernel(hp.channel_n; filters = hp.filters)

# 0002. map physical times -> CA step indices (your choice of time scale)
obs_steps = 20 .* (0:length(test_frames)-1)  # e.g. t=0 → step 0, t=1.0 → step 25, t=2.4 → step 60

# 2. build x0 from your test image
frame_0 = test_frames[1]   # CuArray{Float32,3}, (W,H,4)
W, H = size(frame_0, 1), size(frame_0, 2)
x0 = CUDA.zeros(Float32, W, H, hp.channel_n, 1)
x0[:, :, 1:hp.visible_channels, :] .= reshape(frame_0[:, :, 1:hp.visible_channels], W, H, hp.visible_channels, 1)

# 3. run it
obs_positions = Dict(t => i for (i, t) in enumerate(obs_steps))  # step -> slot in captured
n_obs = length(obs_steps)
n_steps = maximum(hp.obs_steps)  
captured = Vector{typeof(x0)}(undef, n_obs) #keep track of every checkpoint

x = x0

if haskey(obs_positions, 0)
    captured[obs_positions[0]] = x
end

for t in 1:n_steps
    global x = NeuralCellularAutomata.step(model, x, kernel; fire_rate = hp.fire_rate)

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
#savefig(rgb, output_dir)

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
    frame_dir = joinpath(output_dir, "$(image_set)_test_frames"),
    out_path = joinpath(output_dir, "$(train_set)_to_$(image_set).mp4"),
    fps = 30,
    fire_rate = 1.0f0,
    scale = 8,
)

save_hidden_video(
    model,
    x0[:, :, :, 1],
    kernel;
    n_steps = n_steps,
    frame_dir = joinpath(output_dir, "$(image_set)_test_frames"),
    out_path = joinpath(output_dir, "$(train_set)_to_$(image_set).mp4"),
    fps = 30,
    fire_rate = 1.0f0,
    scale = 8,
    vc = hp.visible_channels,
    pad_choice = "zer",
)
@info("Hidden vids acquired")