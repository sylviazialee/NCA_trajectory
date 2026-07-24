"""
Build the perception kernel as a grouped convolution.

Returns a CuArray of shape (k, k, 1, n_filters*channel_n) for use with
`NNlib.conv(...; groups=channel_n)`, where k is the spatial size shared by
all selected filters (3, 5, or 7). Each input channel is independently
convolved with the selected filters; using groups avoids the O(C²) cost of
a full dense conv (the old kernel was mostly zeros).
`filters` is a subset of the `_3`/`_5`/`_7`-suffixed keys (e.g. :id_3,
:sobel_x_5, :laplacian_7) and must all share the same suffix, since a single
grouped conv can only use one kernel size at a time.
Sobel filters are optionally rotated by `angle` (radians).
"""
function build_perception_kernel(
    channel_n::Int;
    filters::Vector{Symbol} = [:id_3, :sobel_x_3, :sobel_y_3],
    angle::Float32 = 0f0,
)
    c, s  = cos(angle), sin(angle)
    ident_3 = Float32[0 0 0; 0 1 0; 0 0 0]
    sx_3    = Float32[-1 0 1; -2 0 2; -1 0 1] ./ 8f0
    sy_3    = permutedims(sx_3)
    ident_5 = Float32[0 0 0 0 0; 0 0 0 0 0; 0 0 1 0 0; 0 0 0 0 0; 0 0 0 0 0]
    sx_5    = Float32[-1 -2 0 2 1; -4 -8 0 8 4; -6 -12 0 12 6; -4 -8 0 8 4; -1 -2 0 2 1] ./ 96f0
    sy_5    = permutedims(sx_5)
    ident_7 = Float32[0 0 0 0 0 0 0; 0 0 0 0 0 0 0; 0 0 0 0 0 0 0; 0 0 0 1 0 0 0; 0 0 0 0 0 0 0; 0 0 0 0 0 0 0; 0 0 0 0 0 0 0]
    sx_7    = Float32[-1 -4 -5 0 5 4 1; -6 -24 -30 0 30 24 6; -15 -60 -75 0 75 60 15; -20 -80 -100 0 100 80 20; -15 -60 -75 0 75 60 15; -6 -24 -30 0 30 24 6; -1 -4 -5 0 5 4 1]./1280f0
    sy_7    = permutedims(sx_7)

    all_filters = Dict{Symbol, Matrix{Float32}}(
        :id_3        => ident_3,
        :sobel_x_3   => c .* sx_3 .- s .* sy_3,
        :sobel_y_3   => s .* sx_3 .+ c .* sy_3,
        :avg_3       => fill(1f0 / 9f0, 3, 3),
        :laplacian_3 => Float32[0 1 0; 1 -4 1; 0 1 0],
        :id_5        => ident_5,
        :sobel_x_5   => c .* sx_5 .- s .* sy_5,
        :sobel_y_5   => s .* sx_5 .+ c .* sy_5,
        :avg_5       => fill(1f0 / 25f0, 5, 5),
        :laplacian_5 => Float32[2 4 4 4 2; 4 0 -8 0 4; 4 -8 -24 -8 4; 4 0 -8 0 4; 2 4 4 4 2],
        :id_7        => ident_7,
        :sobel_x_7   => c .* sx_7 .- s .* sy_7,
        :sobel_y_7   => s .* sx_7 .+ c .* sy_7,
        :avg_7       => fill(1f0 / 49f0, 7, 7),
        :laplacian_7 => Float32[2 8 14 16 14 8 2; 8 24 24 16 24 24 8; 14 24 -30 -80 -30 24 14; 16 16 -80 -160 -80 16 16; 14 24 -30 -80 -30 24 14; 8 24 24 16 24 24 8; 2 8 14 16 14 8 2],
    )

    for f in filters
        haskey(all_filters, f) || error("unknown filter :$f")
    end

    selected = [all_filters[f] for f in filters]
    n_f      = length(selected)
    ksizes   = unique(size.(selected))
    length(ksizes) == 1 ||
        error("mismatched kernel size $(ksizes) for $filters")
    k = ksizes[1][1]
    fstack = cat(selected...; dims = 3)  # (k, k, n_f)

    # Grouped kernel: (k, k, 1, n_f * channel_n)
    # Each group of n_f output channels handles one input channel with the same filter stack.
    kernel = zeros(Float32, k, k, 1, n_f * channel_n)
    for ch in 1:channel_n
        out_base = n_f * (ch - 1)
        kernel[:, :, 1, out_base+1:out_base+n_f] .= fstack
    end

    return kernel |> cu
end

function perceive(x::CuArray{Float32,4}, kernel::CuArray{Float32,4}, channel_n::Int; pad::Int = 1)
    return NNlib.conv(x, kernel; pad=pad, groups=channel_n) # vanilla zero padding
end

struct CAModel
    dense1_w::CuArray{Float32,4}
    dense1_b::CuArray{Float32,1}
    dense2_w::CuArray{Float32,4}
    dense2_b::CuArray{Float32,1}
    channel_n::Int
    fire_rate::Float32
    visible_channels::Int
    pad::Int
end

Flux.@layer CAModel

Flux.trainable(m::CAModel) = (;
    dense1_w = m.dense1_w,
    dense1_b = m.dense1_b,
    dense2_w = m.dense2_w,
    dense2_b = m.dense2_b,
)

function CAModel(hp::HParams)
    C, H, n_f = hp.channel_n, hp.hidden_n, length(hp.filters)
    dense1_w = cu(Float32.(randn(1, 1, n_f * C, H) .* sqrt(2f0 / (n_f * C))))
    dense1_b = CUDA.zeros(Float32, H)
    dense2_w = CUDA.zeros(Float32, 1, 1, H, C)
    dense2_b = CUDA.zeros(Float32, C)
    return CAModel(dense1_w, dense1_b, dense2_w, dense2_b, C, hp.fire_rate, hp.visible_channels, 1)
end

function CAModel(hp::TrajectoryHParams)
    C, H, n_f = hp.channel_n, hp.hidden_n, length(hp.filters)
    dense1_w = cu(Float32.(randn(1, 1, n_f * C, H) .* sqrt(2f0 / (n_f * C))))
    dense1_b = CUDA.zeros(Float32, H)
    dense2_w = CUDA.zeros(Float32, 1, 1, H, C)
    dense2_b = CUDA.zeros(Float32, C)
    return CAModel(dense1_w, dense1_b, dense2_w, dense2_b, C, hp.fire_rate, hp.visible_channels, hp.pad)
end

function step(
    model::CAModel,
    x::CuArray{Float32,4},
    kernel::CuArray{Float32,4};
    fire_rate::Float32 = model.fire_rate,
    step_size::Float32 = 1f0,
    update_mask = nothing,
)
    y = perceive(x, kernel, model.channel_n; pad = model.pad)

    dx = NNlib.conv(y, model.dense1_w) .+ reshape(model.dense1_b, 1, 1, :, 1)
    dx = relu.(dx)
    dx = NNlib.conv(dx, model.dense2_w) .+ reshape(model.dense2_b, 1, 1, :, 1)
    if step_size != 1f0
        dx = dx .* step_size
    end

    mask = update_mask === nothing ?
        Float32.(CUDA.rand(Float32, size(x, 1), size(x, 2), 1, size(x, 4)) .<= fire_rate) :
        update_mask
    x_new = x .+ dx .* mask

    return x_new
end

