include(joinpath(@__DIR__, "..", "src", "NeuralCellularAutomata.jl"))
using .NeuralCellularAutomata
include(joinpath(@__DIR__, "target_prep.jl"))
using CUDA, FileIO, ImageIO
using Dates

image_set = ARGS[1]
fr = parse(Float32, ARGS[2])
fr = 0.5
channel_num = 2
run_id = run_timestamp()
data_dir = "$(image_set)_ch_$(channel_num)"
delta = 50


@info("STARTING: $(image_set) with $(channel_num) channels")

# 1. load your RGBA frames at each observation time as Float32 (W,H,4)
#    (your own loading logic here)
frames = [load_target_gray("$(image_set)/vis_snapshot_$(lpad(i, 2, '0')).png") for i in 0:39]   # Vector of CuArray{Float32,3}

# 2. map physical times -> CA step indices (your choice of time scale)
obs_steps = delta .* (0:(length(frames)-1))  # e.g. t=0 → step 0, t=1.0 → step 25, t=2.4 → step 60


hp = TrajectoryHParams(
    channel_n  = channel_num,
    obs_steps  = obs_steps,
    batch_size = 4,
    train_steps = 3000,
    fire_rate = fr,
    visible_channels = 1,
)

# 3. build x0: RGBA from frame[1], hidden channels = 0
W, H = size(frames[1], 1), size(frames[1], 2)
x0_single = CUDA.zeros(Float32, W, H, hp.channel_n)
x0_single[:, :, 1:1] .= frames[1]  # seed intensity channel from data
x0 = repeat(reshape(x0_single, W, H, hp.channel_n, 1), 1, 1, 1, hp.batch_size)

# 4. targets = frames at obs_steps (step 0 is your initial, still supervise it)
targets = frames   # Vector of CuArray{Float32,3}

model, loss_info, re_loss_info = main_trajectory(x0, targets; hp=hp, run_id = run_id, data_dir = data_dir)

loss_log, final_loss, after_loss = loss_info
re_traj, re_fin, re_after = sqrt.(re_loss_info)

output_dir = joinpath(data_dir, run_id)
mkpath(output_dir)
println(run_id) # show in slurm what id is ig
open(joinpath(output_dir, "read.txt"), "w") do io
    #CHANGE THIS PER TRIAL
    train_pic = image_set
    println(io, Dates.now())
    println(io, "TRAIN NO ALIVE: " * train_pic)
    println(io, "TRAINING FIRE RATE = $(hp.fire_rate)")
    println(io, "TRAJ LOSS = $(loss_log[end])")
    println(io, "FINAL LOSS = $final_loss")
    println(io, "AFTER LOSS = $after_loss")
    println(io, "RE TRAJ LOSS = $re_traj")
    println(io, "RE FINAL LOSS = $re_fin")
    println(io, "RE AFTER LOSS = $re_after")
end