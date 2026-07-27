module NeuralCellularAutomata

using Flux
using Flux: Adam, AdamW, ClipNorm, OptimiserChain
using CUDA
using cuDNN
using NNlib
using Zygote
using Random
using Statistics
using LinearAlgebra
using JLD2
using ProgressMeter
using Printf
using Dates
using FFMPEG_jll
using FileIO: load
using ImageIO
using Colors: RGB, alpha, blue, green, red
using ColorSchemes

ENV["GKSwstype"] = get(ENV, "GKSwstype", "100")
using Plots
gr()

include("config.jl")
include("model.jl")
include("pool.jl")
include("targets.jl")
include("loss.jl")
include("damage.jl")
include("visualization.jl")
include("checkpointing.jl")
include("training.jl")
include("main.jl")

export HParams, CAModel, train!, main, run_ca, load_model, save_model
export image_to_nca_target, image_to_nca_target_mono2, load_target_rgba, run_timestamp
export image_to_gray_target, load_target_gray
export TrajectoryHParams, train_trajectory!, train_subseq!, trajectory_loss, loss_alg2, loss_fn, main_trajectory, main_subseq
export plot_multi, save_loss_plots, save_rollout_video, save_hidden_video
export norm_one, norm_vec
export load_jld
end
