# ============================================================
# General SINDy (own STLS implementation)
#
# Fit sparse polynomial maps
#     dx ≈ Θ(y) ξ
# from samples (Y, dX), where Θ is a monomial library of the
# feature rows in Y.
#
# No external SINDy package — only LinearAlgebra / Statistics.
# Include this file, then call `fit_sindy`.
# ============================================================

using LinearAlgebra
using Statistics: mean, std
using Printf: @sprintf

# ------------------------------------------------------------
# Polynomial library
# ------------------------------------------------------------

"""All multi-indices of total degree 1..max_degree over n_feat variables.

Each exponent vector lists 1-based feature indices with replacement,
sorted nondecreasing (e.g. [1,1,2] → feat1^2 * feat2).
"""
function monomial_exponents(n_feat::Int, max_degree::Int)
    exps = Vector{Vector{Int}}()
    function rec!(cur::Vector{Int}, start::Int, deg_left::Int)
        if deg_left == 0
            push!(exps, copy(cur))
            return
        end
        for i in start:n_feat
            push!(cur, i)
            rec!(cur, i, deg_left - 1)
            pop!(cur)
        end
    end
    for d in 1:max_degree
        rec!(Int[], 1, d)
    end
    return exps
end

function monomial_label(exp::Vector{Int}, feat_names::Vector{String})
    counts = Dict{Int,Int}()
    for i in exp
        counts[i] = get(counts, i, 0) + 1
    end
    parts = String[]
    for i in sort!(collect(keys(counts)))
        p = counts[i]
        name = feat_names[i]
        push!(parts, p == 1 ? name : "$(name)^$(p)")
    end
    return join(parts, "*")
end

"""Build Θ (N×P) from an explicit list of monomials (plus intercept)."""
function build_library(
    Y::AbstractMatrix{<:Real},
    feat_names::Vector{String},
    exps::Vector{Vector{Int}},
)
    N, n_feat = size(Y)
    length(feat_names) == n_feat || error("feature-name count mismatch")
    labels = String["1"]
    append!(labels, (monomial_label(e, feat_names) for e in exps))

    P = length(labels)
    Θ = ones(Float64, N, P)
    for (j, exp) in enumerate(exps)
        col = ones(Float64, N)
        for i in exp
            @inbounds col .*= view(Y, :, i)
        end
        Θ[:, j + 1] .= col
    end
    return Θ, labels, exps
end

"""Build Θ with all monomials of degree ≤ max_degree."""
function build_library(
    Y::AbstractMatrix{<:Real},
    feat_names::Vector{String},
    max_degree::Int,
)
    return build_library(Y, feat_names, monomial_exponents(size(Y, 2), max_degree))
end

"""Map raw ξ back to the dimensionless ζ that STLS thresholds on."""
function normalized_coefficients(
    ξ::AbstractVector{<:Real},
    b::AbstractVector{<:Real},
    column_rms::AbstractVector{<:Real},
)
    b_scale = std(b)
    (!isfinite(b_scale) || b_scale <= eps(Float64)) && (b_scale = 1.0)
    return (Float64.(ξ) .* Float64.(column_rms)) ./ b_scale
end

"""Evaluate one library row θ(y) for a single feature vector."""
function library_row(feats::AbstractVector{<:Real}, exps::Vector{Vector{Int}})
    P = 1 + length(exps)
    θ = ones(Float64, P)
    for (k, exp) in enumerate(exps)
        v = 1.0
        @inbounds for i in exp
            v *= Float64(feats[i])
        end
        θ[k + 1] = v
    end
    return θ
end

# ------------------------------------------------------------
# Ridge-STLS
# ------------------------------------------------------------

"""RMS scale of each library column; intercept stays unscaled."""
function library_column_scales(Θ::Matrix{Float64})
    P = size(Θ, 2)
    scales = ones(Float64, P)
    for j in 2:P
        s = sqrt(mean(abs2, view(Θ, :, j)))
        scales[j] = isfinite(s) && s > eps(Float64) ? s : 1.0
    end
    return scales
end

function ridge_solve(A::Matrix{Float64}, b::Vector{Float64}, ridge::Float64)
    n = size(A, 1)
    gram = (A' * A) / n
    rhs = (A' * b) / n
    # Do not penalize the intercept (column 1).
    for j in 2:size(gram, 1)
        gram[j, j] += ridge
    end
    return gram \ rhs
end

"""Sequential Thresholded Least Squares on normalized columns / target.

Thresholding acts on dimensionless ζ; returned ξ are in original units
so predictions remain Θ * ξ.
"""
function stls(
    Θ::Matrix{Float64},
    b::Vector{Float64};
    threshold::Float64,
    ridge::Float64 = 1e-4,
    n_iter::Int = 10,
)
    P = size(Θ, 2)
    scales = library_column_scales(Θ)
    Θn = Θ ./ reshape(scales, 1, :)
    b_scale = std(b)
    (!isfinite(b_scale) || b_scale <= eps(Float64)) && (b_scale = 1.0)
    bn = b ./ b_scale
    ζ = ridge_solve(Θn, bn, ridge)
    active = trues(P)
    for _ in 1:n_iter
        active .= abs.(ζ) .>= threshold
        if !any(active)
            ζ .= 0
            break
        end
        ζ_new = zeros(Float64, P)
        ζ_new[active] = ridge_solve(Θn[:, active], bn, ridge)
        ζ = ζ_new
    end
    return (b_scale .* ζ) ./ scales
end

function r2_score(y::AbstractVector, yhat::AbstractVector)
    ss_res = sum(abs2, y .- yhat)
    ss_tot = sum(abs2, y .- mean(y))
    ss_tot < 1e-30 && return ss_res < 1e-30 ? 1.0 : 0.0
    return 1.0 - ss_res / ss_tot
end

function format_equation(
    labels::Vector{String},
    ξ::Vector{Float64};
    atol = 0.0,
    max_terms::Int = typemax(Int),
)
    terms = String[]
    order = sortperm(abs.(ξ); rev = true)
    n_kept = 0
    for j in order
        c = ξ[j]
        abs(c) <= atol && continue
        lab = labels[j]
        n_kept += 1
        if n_kept > max_terms
            push!(terms, "…")
            break
        end
        if lab == "1"
            push!(terms, @sprintf("%+.3g", c))
        else
            push!(terms, @sprintf("%+.3g %s", c, lab))
        end
    end
    isempty(terms) && return "0"
    return join(terms, " ")
end

# ------------------------------------------------------------
# High-level fit
# ------------------------------------------------------------

"""Fit one sparse polynomial map per output column of dX.

Arguments
- `Y::Matrix`: N×n_feat feature samples
- `dX::Matrix`: N×C target updates
- `feat_names`: length n_feat labels for printing

Keyword args match STLS defaults.
"""
function fit_sindy(
    Y::AbstractMatrix{<:Real},
    dX::AbstractMatrix{<:Real},
    feat_names::Vector{String};
    max_degree::Int = 3,
    exps::Union{Nothing, Vector{Vector{Int}}} = nothing,
    threshold::Float64 = 1e-2,
    ridge::Float64 = 1e-4,
    n_iter::Int = 10,
)
    size(Y, 1) == size(dX, 1) || error("Y and dX must have the same number of rows")
    Y64 = Float64.(Y)
    Θ, labels, used_exps = if exps === nothing
        build_library(Y64, feat_names, max_degree)
    else
        build_library(Y64, feat_names, exps)
    end
    P = length(labels)
    C = size(dX, 2)
    column_rms = library_column_scales(Θ)
    Ξ = zeros(Float64, P, C)
    Ξ_normalized = zeros(Float64, P, C)
    metrics = Vector{NamedTuple{(:r2, :rmse, :nnz), Tuple{Float64,Float64,Int}}}(undef, C)

    for c in 1:C
        b = Float64.(dX[:, c])
        ξ = stls(Θ, b; threshold, ridge, n_iter)
        Ξ[:, c] .= ξ
        Ξ_normalized[:, c] .= normalized_coefficients(ξ, b, column_rms)
        yhat = Θ * ξ
        nnz = count(abs.(ξ) .> 0)
        r2 = r2_score(b, yhat)
        rmse = sqrt(mean(abs2, b .- yhat))
        metrics[c] = (r2 = r2, rmse = rmse, nnz = nnz)
    end

    return (;
        Ξ = Ξ,
        Ξ_normalized = Ξ_normalized,
        Θ = Θ,
        labels = labels,
        exps = used_exps,
        metrics = metrics,
        column_rms = column_rms,
    )
end

"""Predict dx = Θ(y) Ξ for a batch of feature rows Y (N×n_feat)."""
function predict_sindy(
    Y::AbstractMatrix{<:Real},
    Ξ::Matrix{Float64},
    exps::Vector{Vector{Int}},
)
    N = size(Y, 1)
    C = size(Ξ, 2)
    P = size(Ξ, 1)
    dX = zeros(Float64, N, C)
    for n in 1:N
        θ = library_row(view(Y, n, :), exps)
        @assert length(θ) == P
        for c in 1:C
            s = 0.0
            for p in 1:P
                s += θ[p] * Ξ[p, c]
            end
            dX[n, c] = s
        end
    end
    return dX
end
