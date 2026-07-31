# See readme.txt for usage and outputs.

include(joinpath(@__DIR__, "sindy_lib.jl"))
include(joinpath(@__DIR__, "sindy_plots.jl"))
include(joinpath(@__DIR__, "..", "src", "NeuralCellularAutomata.jl"))
using .NeuralCellularAutomata
using CUDA
using NNlib
using LinearAlgebra
using Statistics: mean, std
using Random
using Printf: @sprintf, @printf
using Plots
using Dates
using ProgressMeter
using FFMPEG_jll
using JLD2

#------------------------------------------------
# Config
#------------------------------------------------

# CLI can override max_degree / n_samples / threshold / fit_steps below.
const EVAL_FIRE_RATE = 1.0f0  # no stochastic mask — every cell updates
const REPO = normpath(joinpath(@__DIR__, ".."))

# point RUN_DIR at your trained run
const RUN_DIR = joinpath(REPO, "spread_ch_3_Delta_T_20", "20260727_234829")

# training frames live under REPO/IMAGE_SET/snapshot_XX.png
const IMAGE_SET = "spread"

# optional u_traj fallback; leave unused when IMAGE_SET is set
const TRAJ_SOURCE = joinpath(REPO, "fhn_jld2")
const USE_IMAGE_SET = true  # false → load TRAJ_SOURCE instead
const TRAIN_RATIO = 0.8
const BATCH_SIZE = 1  # fr=1 is deterministic, so batch>1 doesn't change anything
const VIDEO_SCALE = 8
const VIDEO_FPS = 30

# STLS + how we pick cells along the traj
const RIDGE_LAMBDA = 1e-4
const ACTIVE_SAMPLE_FRACTION = 0.5  # other half is uniform over the grid
const ACTIVE_POOL_FRACTION = 0.2    # "active" = top 20% |dx|

# library: reaction-diffusion (id polys + Lap + id*Lap) like SINDY_new
const USE_RD_LIBRARY = true

#------------------------------------------------
# CLI / paths
#------------------------------------------------

function resolve_ckpt(arg::AbstractString)
    p = abspath(arg)
    if isfile(p) && endswith(lowercase(p), ".jld2")
        return p
    end
    isdir(p) || error("path not found: $arg")
    for name in ("traj_final.jld2",)
        c = joinpath(p, "checkpoints", name)
        isfile(c) && return abspath(c)
    end
    c = joinpath(p, "traj_final.jld2")
    isfile(c) && return abspath(c)
    error("No traj_final.jld2 under $p (expected checkpoints/traj_final.jld2)")
end

ckpt_path  = resolve_ckpt(RUN_DIR)
max_degree = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 3
n_samples  = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 20_000
threshold  = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 1e-2
fit_steps_arg = if length(ARGS) >= 5
    # old CLI had x_scale/grid here — ignore those, we fit on the traj now
    @warn "Ignoring obsolete x_scale=$(ARGS[4]) and grid=$(ARGS[5]); fitting on the NCA trajectory"
    0
elseif length(ARGS) >= 4
    value = parse(Float64, ARGS[4])
    isinteger(value) || error("fit_steps must be an integer (got $(ARGS[4]))")
    Int(value)
else
    0
end

(1 <= max_degree <= 4) || error("max_degree must be in 1..4 (got $max_degree)")
n_samples >= 100 || error("n_samples too small")
fit_steps_arg >= 0 || error("fit_steps must be >= 0")

# NCA helpers (sampling, sparse step, rollout eval) — see sindy_nca.jl
include(joinpath(@__DIR__, "sindy_nca.jl"))

#------------------------------------------------
# Load model + frames
#------------------------------------------------

NeuralCellularAutomata.report_backend!()
model, hp = load_model(ckpt_path)
filters = Vector{Symbol}(hp.filters)
C = Int(model.channel_n)
kernel = NeuralCellularAutomata.build_perception_kernel(C; filters = filters)
feat_names = perception_feature_names(filters, C)
n_feat = length(feat_names)

# outputs live under the run folder (sibling of checkpoints/)
run_dir = let
    ckpt_parent = dirname(ckpt_path)
    lowercase(basename(ckpt_parent)) == "checkpoints" ? dirname(ckpt_parent) : ckpt_parent
end
stamp = Dates.format(Dates.now(), dateformat"yyyymmdd_HHMMSS")
out_dir = joinpath(run_dir, "outputs", "sparse_symbolic_$(stamp)")
mkpath(out_dir)

if USE_IMAGE_SET
    snap_dir = joinpath(REPO, IMAGE_SET)
    n_avail = count_image_set_frames(IMAGE_SET)
    n_avail >= 2 || error("need ≥2 frames in $snap_dir")
    target_frames = load_image_set_frames(IMAGE_SET, n_avail)
    frame_source = snap_dir
    ic_label = joinpath(snap_dir, "snapshot_00.png")
else
    target_frames, traj_path = load_u_traj_frames(TRAJ_SOURCE)
    frame_source = traj_path
    ic_label = "raw $traj_path u_traj[:,:,1] + zero hidden"
end

@info "SPARSE SYMBOLIC REGRESSION (SINDy / STLS via sindy_lib.jl)"
@info "ckpt        = $ckpt_path"
@info "out_dir     = $out_dir"
@info "filters     = $filters"
@info "channel_n   = $C  (visible=$(model.visible_channels))"
@info "n_feat      = $n_feat  → $feat_names"
@info "fire_rate   = $EVAL_FIRE_RATE  (deterministic dx; train fr was $(model.fire_rate))"
@info "max_degree  = $max_degree | n_samples=$n_samples | threshold=$threshold"
@info "ridge       = $RIDGE_LAMBDA | active sample fraction=$ACTIVE_SAMPLE_FRACTION (top pool=$ACTIVE_POOL_FRACTION)"
@info "frames      = $frame_source"

#------------------------------------------------
# Sample (perception, dx) on the NCA traj
#------------------------------------------------
Random.seed!(0)
n_fit_frames = length(target_frames)
fit_obs_steps = resolve_obs_steps(hp, n_fit_frames)
fit_steps = fit_steps_arg == 0 ? maximum(fit_obs_steps) : fit_steps_arg
fit_vc = Int(model.visible_channels)
fit_seed = CUDA.zeros(Float32, size(target_frames[1], 1), size(target_frames[1], 2), C)
fit_seed[:, :, 1:fit_vc] .= target_frames[1][:, :, 1:fit_vc]

@info "sample      = original NCA fire_rate=1 trajectory, t=0:$fit_steps"
@info "sample IC   = $ic_label; $n_samples cells balanced across time"
Y, dX = sample_nca_trajectory_perception_and_dx(
    model, kernel, fit_seed;
    n_samples, n_steps = fit_steps,
)

@info "feature stats: mean=$(round.(mean(Y; dims=1); digits=4))  std=$(round.(std(Y; dims=1); digits=4))"
@info "dx stats:      mean=$(round.(mean(dX; dims=1); digits=4))  std=$(round.(std(dX; dims=1); digits=4))"

#------------------------------------------------
# Fit sparse polys
#------------------------------------------------

library_exps = if USE_RD_LIBRARY
    state_dependent_diffusion_exponents(filters, C, max_degree)
else
    nothing
end
library_type = if USE_RD_LIBRARY
    "identity reaction monomials degree ≤$max_degree + linear Laplacians + id*Laplacian"
else
    "all NCA perception-feature monomials degree ≤$max_degree"
end

fit = fit_sindy(
    Y, dX, feat_names;
    max_degree = max_degree,
    exps = library_exps,
    threshold = Float64(threshold),
    ridge = RIDGE_LAMBDA,
)
Ξ, Ξ_normalized = fit.Ξ, fit.Ξ_normalized
labels, exps, metrics = fit.labels, fit.exps, fit.metrics
column_rms = fit.column_rms
P = length(labels)
@info "library size P = $P  ($library_type)"

println()
println("="^72)
println("Sparse polynomial approx of dx = f(perception) @ fire_rate=$(EVAL_FIRE_RATE)")
println("dimensionless threshold = $threshold   (Θ columns + each dx channel normalized)")
println("ridge = $RIDGE_LAMBDA | active/uniform sampling = $(ACTIVE_SAMPLE_FRACTION)/$(1-ACTIVE_SAMPLE_FRACTION)")
println("library = $library_type")
println("Saving results → $out_dir")
println("="^72)

#------------------------------------------------
# Print equations (raw + normalized, like SINDY_new)
#------------------------------------------------
io_summary = IOBuffer()
for c in 1:C
    m = metrics[c]
    ξ = Ξ[:, c]
    ζ = Ξ_normalized[:, c]
    eq = format_equation(labels, ξ)
    println()
    println("--- dx_c$(c-1)  (R²=$(round(m.r2; digits=5))  RMSE=$(round(m.rmse; digits=6))  nnz=$(m.nnz) / $P) ---")
    println("dx_c$(c-1) ≈ ", eq)
    println()
    println("coefficients (nonzero only; raw, normalized-for-threshold):")
    for (lab, coef, coef_n) in zip(labels, ξ, ζ)
        abs(coef) == 0 && continue
        @printf("  raw=%12.6g  normalized=%12.6g  %s\n", coef, coef_n, lab)
    end

    println(io_summary)
    println(io_summary, "--- dx_c$(c-1)  (R²=$(round(m.r2; digits=5))  RMSE=$(round(m.rmse; digits=6))  nnz=$(m.nnz) / $P) ---")
    println(io_summary, "dx_c$(c-1) ≈ ", eq)
    println(io_summary, "coefficients (nonzero only; raw, normalized-for-threshold):")
    for (lab, coef, coef_n) in zip(labels, ξ, ζ)
        abs(coef) == 0 && continue
        @printf(io_summary, "  raw=%12.6g  normalized=%12.6g  %s\n", coef, coef_n, lab)
    end
end

#------------------------------------------------
# Save fit artifacts
#------------------------------------------------
summary_path = joinpath(out_dir, "summary.txt")
open(summary_path, "w") do io
    println(io, "ckpt = $ckpt_path")
    println(io, "filters = $filters")
    println(io, "channel_n = $C  visible=$(model.visible_channels)")
    println(io, "fire_rate_eval = $EVAL_FIRE_RATE  train_fr=$(model.fire_rate)")
    println(io, "max_degree = $max_degree  n_samples=$n_samples  threshold=$threshold")
    println(io, "threshold_basis = dimensionless coefficient after normalizing Θ columns and dx channel std")
    println(io, "ridge_lambda = $RIDGE_LAMBDA")
    println(io, "active_sample_fraction = $ACTIVE_SAMPLE_FRACTION")
    println(io, "active_pool_fraction = $ACTIVE_POOL_FRACTION")
    println(io, "library_column_rms = $column_rms")
    println(io, "sampling = original NCA fire_rate=1 trajectory")
    println(io, "fit_steps = $fit_steps  fit_initial_state=$ic_label")
    println(io, "library_type = $library_type")
    println(io, "n_feat = $n_feat  library_P = $P")
    println(io, "feat_names = $feat_names")
    println(io, "frame_source = $frame_source")
    USE_IMAGE_SET && println(io, "image_set = $IMAGE_SET")
    println(io)
    print(io, String(take!(io_summary)))
end

csv_path = joinpath(out_dir, "coefficients.csv")
open(csv_path, "w") do io
    print(io, "term")
    for c in 0:(C - 1)
        print(io, ",dx_c$(c)")
    end
    for c in 0:(C - 1)
        print(io, ",normalized_dx_c$(c)")
    end
    println(io)
    for j in 1:P
        print(io, labels[j])
        for c in 1:C
            @printf(io, ",%.8g", Ξ[j, c])
        end
        for c in 1:C
            @printf(io, ",%.8g", Ξ_normalized[j, c])
        end
        println(io)
    end
end

plot_path = joinpath(out_dir, "sparse_symbolic.png")
plot_sparse_symbolic(
    labels, Ξ, metrics;
    threshold = Float64(threshold), P = P, out_path = plot_path,
)

model_jld = joinpath(out_dir, "sparse_model.jld2")
jldsave(model_jld;
    Ξ = Ξ,
    Ξ_normalized = Ξ_normalized,
    labels = labels,
    exps = exps,
    feat_names = feat_names,
    filters = filters,
    channel_n = C,
    visible_channels = Int(model.visible_channels),
    max_degree = max_degree,
    library_type = library_type,
    threshold = Float64(threshold),
    ridge_lambda = RIDGE_LAMBDA,
    active_sample_fraction = ACTIVE_SAMPLE_FRACTION,
    active_pool_fraction = ACTIVE_POOL_FRACTION,
    library_column_rms = column_rms,
    fit_steps = fit_steps,
    sampling = "original NCA fire_rate=1 trajectory",
    frame_source = frame_source,
    ckpt_path = ckpt_path,
)

#------------------------------------------------
# Rollout eval vs target frames
#------------------------------------------------

vc = Int(model.visible_channels)
eval_info = eval_sparse_rollout!(
    out_dir, run_dir, model, kernel, C, vc, Ξ, exps, labels,
    target_frames, frame_source, hp;
    max_degree = max_degree, threshold = Float64(threshold),
)

# stable "latest" folder next to the timestamped one
latest_dir = joinpath(run_dir, "outputs", "sparse_symbolic")
mkpath(latest_dir)
for f in (summary_path, csv_path, plot_path, model_jld, eval_info.read_path,
          eval_info.snap_path, eval_info.re_plot_path, eval_info.re_log_path, eval_info.video_path)
    isfile(f) && cp(f, joinpath(latest_dir, basename(f)); force = true)
end

#------------------------------------------------
# Done
#------------------------------------------------

println()
println("="^72)
println("Saved fit:")
println("  $summary_path")
println("  $csv_path")
println("  $plot_path")
println("  $model_jld")
println("Saved sparse rollout @ fr=1.0:")
println("  $(eval_info.read_path)")
println("  $(joinpath(run_dir, "read_sparse_fr1.0.txt"))")
println("  $(eval_info.snap_path)")
println("  $(eval_info.re_plot_path)")
println("  $(eval_info.re_log_path)")
println("  $(eval_info.video_path)")
println("  RE TRAIN=$(eval_info.re_train) TEST=$(eval_info.re_test) FULL=$(eval_info.re_full)")
println("  RE FINAL=$(eval_info.re_final) AFTER=$(eval_info.re_after) blew_up=$(eval_info.blew_up)")
println("  (also copied to $latest_dir)")
println("Done.")
println("="^72)
