include(joinpath(@__DIR__, "..", "src", "NeuralCellularAutomata.jl"))
using .NeuralCellularAutomata
include(joinpath(@__DIR__, "target_prep.jl"))
using CUDA, FileIO, ImageIO
using Dates

image_set = ARGS[1]
num = parse(Int64, ARGS[2])
fr = 1.0
run_id = run_timestamp()
data_dir = "subseq_$(image_set)"
train_steps = 3000
delta = parse(Int64, ARGS[3])
delta = 1


# 1. load your RGBA frames at each observation time as Float32 (W,H,4)
#    (your own loading logic here)
frame_paths = ["$(image_set)/vis_snapshot_$(lpad(i, 2, '0')).jld2" for i in 0:9]
println("first frame path: ", frame_paths[1])
@show isfile(frame_paths[1])
frames = [load_jld(p) for p in frame_paths]
frames = [clamp.(f, 0f0, 1f0) for f in frames]  # Vector of CuArray{Float32,3}
@show length(frames)
@show size(frames[1])
# 2. map physical times -> CA step indices (your choice of time scale)
obs_steps = delta .* (0:length(frames))  # e.g. t=0 → step 0, t=1.0 → step 25, t=2.4 → step 60

# define subseq indices from obs_steps
if num == 3
    subseqs = [(1:3), (1:7), (1:10)]
elseif num == 5
    subseqs = [(1:3), (1:4), (1:6), (1:8), (1:10)]
elseif num == 7
    subseqs = [(1:3), (1:4), (1:6), (1:7), (1:8), (1:9), (1:10)]
else 
    subseqs = [(1:10)]
end
numseq = length(subseqs)
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
    visible_channels = 1
)
hps[1] = hp
current_hp = hp
current_frames = subframes[1]

# make seed BASED OFF OF CURRENT SUBFRAMES, W and H do not need to be re-defined
W, H = size(current_frames[1], 1), size(current_frames[1], 2)
x0_single = CUDA.zeros(Float32, W, H, current_hp.channel_n)
x0_single[:, :, 1:current_hp.visible_channels] .= current_frames[1]            # seed RGBA from data
x0 = repeat(reshape(x0_single, W, H, current_hp.channel_n, 1), 1, 1, 1, current_hp.batch_size)

models[1], loss_info[1], re_loss_info[1] = main_subseq(x0, subframes[1]; hp = hps[1], run_id = run_id, data_dir = data_dir)

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
        visible_channels = 1
    )
    hps[i] = current_hp # update hps vector of hp
    
    # make seed BASED OFF OF CURRENT SUBFRAMES
    x0_single = CUDA.zeros(Float32, W, H, current_hp.channel_n)
    x0_single[:, :, 1:current_hp.visible_channels] .= current_frames[1]            # seed RGBA from data
    x0 = repeat(reshape(x0_single, W, H, current_hp.channel_n, 1), 1, 1, 1, current_hp.batch_size)
    
    # update models, loss info
    models[i], loss_info[i], re_loss_info[i] = main_subseq(x0, subframes[i]; past_model = models[i-1], hp = hps[i], run_id = run_id, data_dir = data_dir)
end


output_dir = joinpath(data_dir, run_id)
mkpath(output_dir)

open(joinpath(output_dir, "read.txt"), "w") do io
    println(io, Dates.now())
    println(io, "TRAIN: " * image_set)
    println(io, "TRAINING FIRE RATE = $(hp.fire_rate)")
    for i in 1:numseq
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