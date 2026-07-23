function loss_fn(x::CuArray{Float32,4}, target::CuArray{Float32,3}; vc::Int = 3)
    vis  = x[:, :, 1:vc, :]
    tgt  = reshape(target[:, :, 1:vc], size(target, 1), size(target, 2), vc, 1)
    return mean((vis .- tgt) .^ 2)
end

function norm_one(target::CuArray{Float32,3}; vc::Int = 3)
    tgt  = reshape(target[:, :, 1:vc], size(target, 1), size(target, 2), vc, 1)
    return mean((tgt).^ 2)
end

function norm_vec(targets::Vector{<:CuArray{Float32, 3}}; vc::Int = 3)
    len = length(targets)
    total_norm = 0
    for i in 1:len
        total_norm += norm_one(targets[i]; vc = vc)
    end
    return total_norm/len
end

function trajectory_loss(
    xs::Vector{<:CuArray{Float32,4}},
    targets::Vector{<:CuArray{Float32,3}},
    weights::Vector{Float32} = ones(Float32, length(xs));
    vc::Int = 3,
)
    @assert length(xs) == length(targets) == length(weights)
    total = 0f0
    for (x, tgt, w) in zip(xs, targets, weights)
        vis = x[:, :, 1:vc, :]
        diff = vis .- reshape(tgt, size(tgt)..., 1)
        total += w * mean(diff .^ 2)
    end
    return total
end

#my func
#=
function loss_ex(x::CuArray{Float32,4}, target::CuArray{Float32,3}; vc::Int = 3)
    vis  = x[:, :, 1:vc, :] # top 4 visible channel layers
    tgt  = reshape(target[:, :, 1:vc], size(target, 1), size(target, 2), vc, 1) # reshape target to compare

    #add the sum of the differences between alive/dead states of predicted vs target (what to do if target does not have specific number for this tho?)
    vis_al = vis[:, :, 4, :]
    tgt_al = tgt[:, :, 4, 1] 
    #take only the alpha (aliveness) values in target and predicted
    
    loss_al = mean((vis_al .- tgt_al) .^ 2)
    loss_og = mean((vis .- tgt) .^ 2) # sum of squares of differences (magnitude squared loss func)
    c = 0.0005f0
    return c * loss_al + loss_og
end
=#
function per_sample_loss(x::CuArray{Float32,4}, target::CuArray{Float32,3}; vc::Int = 3)
    vis  = x[:, :, 1:vc, :]
    tgt  = reshape(target[:, :, 1:vc], size(target, 1), size(target, 2), vc, 1)
    return Array(vec(mean((vis .- tgt) .^ 2; dims=(1, 2, 3))))
end

function loss_alg2(
    x::Vector{<:CuArray{Float32, 4}}, #length M vector of 4d target tensors (i, j, channel vector, R = number of samples), here R=1
    y::Vector{<:CuArray{Float32, 3}}; #length M of 3d target tensors (i, j, channel vector)
    vc::Int = 3,
    )
    @assert length(x) == length(y)
    total_loss = 0
    M = size(y, 1)
    #R = size(x[1], 4)

    for m in 1:M
        #x[m-1] is 4d tensor of batch of R samples at checkpoint m-1, expected for loss_fn
        #pred_x = phi_n(phi, x[m], theta, t)
        total_loss += loss_fn(x[m], y[m]; vc = vc)
    end

    total_loss = total_loss/M
    return total_loss
end