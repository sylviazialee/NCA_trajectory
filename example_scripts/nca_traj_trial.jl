include(joinpath(@__DIR__, "..", "src", "NeuralCellularAutomata.jl"))
using .NeuralCellularAutomata
include(joinpath(@__DIR__, "target_prep.jl"))
using CUDA, FileIO, ImageIO
using Dates

image_set = ARGS[1]
fr = 0.5
channel_num = 16
combo_string = ARGS[2]
run_id = run_timestamp()
data_dir = "$(image_set)_combo_$(combo_string)"


@info("STARTING: $(image_set) with $(channel_num) channels")

# 1. load your RGBA frames at each observation time as Float32 (W,H,4)
#    (your own loading logic here)
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
          load_target_rgba("$(image_set)/snapshot_19.png")]   # Vector of CuArray{Float32,3}

# 2. map physical times -> CA step indices (your choice of time scale)
obs_steps = 20 .* (0:19)  # e.g. t=0 → step 0, t=1.0 → step 25, t=2.4 → step 60

filter_index = [:id, :sobel_x, :sobel_y, :laplacian5, :laplacian9]

@assert length(filter_index)===length(combo_string)

subset_filter = Symbol[]
for i in 1:length(combo_string)
    if combo_string[i] == '1'
        push!(subset_filter, filter_index[i])
    else
        combo_string[i] != '0' && error("bad combo string")
    end
end

isempty(subset_filter) && error("empty filter set")

hp = TrajectoryHParams(
    channel_n  = channel_num,
    obs_steps  = obs_steps,
    batch_size = 4,
    train_steps = 3000,
    fire_rate = fr,
    visible_channels = 3,
    filters = subset_filter,
)

# 3. build x0: RGBA from frame[1], hidden channels = 0
W, H = size(frames[1], 1), size(frames[1], 2)
x0_single = CUDA.zeros(Float32, W, H, hp.channel_n)
x0_single[:, :, 1:3] .= frames[1]            # seed RGBA from data
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
    println(io, "COMBO STRING (id, sx, sy, lap) = $(combo_string)")
    println(io, "TRAJ LOSS = $(loss_log[end])")
    println(io, "FINAL LOSS = $final_loss")
    println(io, "AFTER LOSS = $after_loss")
    println(io, "RE TRAJ LOSS = $re_traj")
    println(io, "RE FINAL LOSS = $re_fin")
    println(io, "RE AFTER LOSS = $re_after")
end