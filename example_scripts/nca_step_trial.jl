include(joinpath(@__DIR__, "..", "src", "NeuralCellularAutomata.jl"))
using .NeuralCellularAutomata
include(joinpath(@__DIR__, "target_prep.jl"))
using CUDA, FileIO, ImageIO

# 1. load your RGBA frames at each observation time as Float32 (W,H,4)
#    (your own loading logic here)
frames = [load_target_rgba("frenchflag/01.png"), 
          load_target_rgba("frenchflag/02.png"),
          load_target_rgba("frenchflag/03.png"),
          load_target_rgba("frenchflag/04.png"),
          load_target_rgba("frenchflag/05.png"), 
          load_target_rgba("frenchflag/06.png"),
          load_target_rgba("frenchflag/07.png"),
          load_target_rgba("frenchflag/08.png"),
          load_target_rgba("frenchflag/09.png"), 
          load_target_rgba("frenchflag/10.png"),
          load_target_rgba("frenchflag/11.png"),
          load_target_rgba("frenchflag/12.png"),
          load_target_rgba("frenchflag/13.png"), 
          load_target_rgba("frenchflag/14.png"),
          load_target_rgba("frenchflag/15.png"),
          load_target_rgba("frenchflag/16.png"),
          load_target_rgba("frenchflag/17.png"), 
          load_target_rgba("frenchflag/18.png"),
          load_target_rgba("frenchflag/19.png"),
          load_target_rgba("frenchflag/20.png")]   # Vector of CuArray{Float32,3}

# 2. map physical times -> CA step indices (your choice of time scale)
obs_steps = 20 .* (0:19)  # e.g. t=0 → step 0, t=1.0 → step 25, t=2.4 → step 60

hp = TrajectoryHParams(
    channel_n  = 16,
    obs_steps  = obs_steps,
    batch_size = 4,
    train_steps = 3000,
    fire_rate  = 1.0f0,
)

# 3. build x0: RGBA from frame[1], hidden channels = 0
W, H = size(frames[1], 1), size(frames[1], 2)

# this is for first step, but you have to do this for everything for step trial
x0_single = CUDA.zeros(Float32, W, H, hp.channel_n)
x0_single[:, :, 1:4] .= frames[1]            # seed RGBA from data
x0 = repeat(reshape(x0_single, W, H, hp.channel_n, 1), 1, 1, 1, hp.batch_size)

# 4. targets = frames at obs_steps (step 0 is your initial, still supervise it)
targets = frames   # Vector of CuArray{Float32,3}

model, loss_log = main_step(x0, targets; hp, video_fire_rate = 0.5f0)


