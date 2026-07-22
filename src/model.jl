"""
Build the fixed 3x3 perception kernel as a grouped convolution.

Returns a CuArray of shape (3, 3, 1, n_filters*channel_n) for use with
`NNlib.conv(...; groups=channel_n)`. Each input channel is independently
convolved with the selected filters; using groups avoids the O(C²) cost of
a full dense conv (the old kernel was mostly zeros).
`filters` is a subset of [:id, :sobel_x, :sobel_y, :avg, :laplacian].
Sobel filters are optionally rotated by `angle` (radians).
"""
function build_perception_kernel(
    channel_n::Int;
    filters::Vector{Symbol} = [:id, :sobel_x, :sobel_y],
    angle::Float32 = 0f0,
)
    ident = Float32[0 0 0; 0 1 0; 0 0 0]
    sx    = Float32[-1 0 1; -2 0 2; -1 0 1] ./ 8f0
    sy    = permutedims(sx)
    c, s  = cos(angle), sin(angle)

    all_filters = Dict{Symbol, Matrix{Float32}}(
        :id        => ident,
        :sobel_x   => c .* sx .- s .* sy,
        :sobel_y   => s .* sx .+ c .* sy,
        :avg       => fill(1f0 / 9f0, 3, 3),
        :laplacian => Float32[0 1 0; 1 -4 1; 0 1 0],
    )

    for f in filters
        haskey(all_filters, f) || error("unknown filter :$f")
    end

    selected = [all_filters[f] for f in filters]
    n_f      = length(selected)
    fstack   = cat(selected...; dims = 3)  # (3, 3, n_f)

    # Grouped kernel: (3, 3, 1, n_f * channel_n)
    # Each group of n_f output channels handles one input channel with the same filter stack.
    kernel = zeros(Float32, 3, 3, 1, n_f * channel_n)
    for ch in 1:channel_n
        out_base = n_f * (ch - 1)
        kernel[:, :, 1, out_base+1:out_base+n_f] .= fstack
    end

    return kernel |> cu
end

function perceive(x::CuArray{Float32,4}, kernel::CuArray{Float32,4}, channel_n::Int)
    # replicate padding
    x_rep = NNlib.pad_repeat(x, 1; dims=(1, 2))
    x_cir = NNlib.pad_circular(x, 1; dims=(1, 2))
    return NNlib.conv(x_cir, kernel; pad=0, groups=channel_n)
    #return NNlib.conv(x, kernel; pad=1, groups=channel_n) # vanilla zero padding
end

struct CAModel
    dense1_w::CuArray{Float32,4}
    dense1_b::CuArray{Float32,1}
    dense2_w::CuArray{Float32,4}
    dense2_b::CuArray{Float32,1}
    channel_n::Int
    fire_rate::Float32
    visible_channels::Int
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
    return CAModel(dense1_w, dense1_b, dense2_w, dense2_b, C, hp.fire_rate, hp.visible_channels)
end

function CAModel(hp::TrajectoryHParams)
    C, H, n_f = hp.channel_n, hp.hidden_n, length(hp.filters)
    dense1_w = cu(Float32.(randn(1, 1, n_f * C, H) .* sqrt(2f0 / (n_f * C))))
    dense1_b = CUDA.zeros(Float32, H)
    dense2_w = CUDA.zeros(Float32, 1, 1, H, C)
    dense2_b = CUDA.zeros(Float32, C)
    return CAModel(dense1_w, dense1_b, dense2_w, dense2_b, C, hp.fire_rate, hp.visible_channels)
end

function step(
    model::CAModel,
    x::CuArray{Float32,4},
    kernel::CuArray{Float32,4};
    fire_rate::Float32 = model.fire_rate,
    step_size::Float32 = 1f0,
    update_mask = nothing,
)
    y = perceive(x, kernel, model.channel_n)

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

