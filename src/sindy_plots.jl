# ============================================================
# SINDy plotting helpers
#
# Expects `sindy_lib.jl` already included (`format_equation`).
# ============================================================

using Plots
using Colors: RGB
using Printf: @sprintf
gr()

"""Pretty-print library term for plot x-tick (compact)."""
function pretty_term(lab::AbstractString)
    lab == "1" && return "1"
    s = replace(String(lab), "*" => " ")
    return s
end

"""Signed asinh scale for symlog-like coefficient bars."""
function symlog_y(x::Real; linthresh::Float64 = 1e-2)
    return asinh(Float64(x) / linthresh)
end

function channel_colors(C::Int)
    palette = [
        RGB(0.75, 0.15, 0.18),
        RGB(0.15, 0.35, 0.70),
        RGB(0.10, 0.55, 0.30),
        RGB(0.85, 0.45, 0.10),
        RGB(0.45, 0.25, 0.55),
        RGB(0.10, 0.55, 0.55),
        RGB(0.55, 0.35, 0.15),
        RGB(0.70, 0.20, 0.50),
    ]
    return [palette[mod1(c, length(palette))] for c in 1:C]
end

"""
Paper-style figure:
  top  — c sparse update equations
  bottom — coefficient bars for terms active in any channel (c colors)
"""
function plot_sparse_symbolic(
    labels::Vector{String},
    Ξ::Matrix{Float64},
    metrics;
    threshold::Float64,
    P::Int,
    out_path::AbstractString,
    linthresh::Float64 = 1e-2,
)
    C = size(Ξ, 2)
    colors = channel_colors(C)

    active = falses(length(labels))
    for j in eachindex(labels)
        active[j] = any(abs(Ξ[j, c]) > 0 for c in 1:C)
    end
    idxs = findall(active)
    isempty(idxs) && (idxs = [1])

    function sortkey(j)
        labels[j] == "1" && return (-Inf, 0.0)
        return (0.0, -maximum(abs, view(Ξ, j, :)))
    end
    sort!(idxs; by = sortkey)

    term_labs = pretty_term.(labels[idxs])
    n_terms = length(idxs)

    eq_lines = String[]
    push!(eq_lines, "Learned sparse updates  (dimensionless threshold=$(threshold), library P=$(P))")
    for c in 1:C
        m = metrics[c]
        eq = format_equation(labels, Ξ[:, c]; max_terms = 12)
        push!(eq_lines, @sprintf(
            "dx_c%d  [R²=%.3f  RMSE=%.3g  nnz=%d]   ≈  %s",
            c - 1, m.r2, m.rmse, m.nnz, eq,
        ))
    end

    y_eq = reverse(collect(1:length(eq_lines)))
    p_eq = plot(
        xlims = (0, 1),
        ylims = (0.5, length(eq_lines) + 0.5),
        axis = false,
        grid = false,
        legend = false,
        size = (1400, 180 + 28 * C),
        left_margin = 4Plots.mm,
        right_margin = 4Plots.mm,
        top_margin = 2Plots.mm,
        bottom_margin = 0Plots.mm,
    )
    for (k, line) in enumerate(eq_lines)
        fs = k == 1 ? 11 : 8
        fw = k == 1 ? :bold : :normal
        annotate!(p_eq, 0.01, y_eq[k], text(line, fs, fw, :left, :black))
    end
    title!(p_eq, "Learned updates", titlefontsize = 13, titlelocation = :left)

    bar_w = 0.8 / C
    p_bar = plot(
        size = (1400, 520),
        legend = :outertopright,
        xlabel = "library term",
        ylabel = "coefficient  (symlog)",
        grid = true,
        gridstyle = :dot,
        gridalpha = 0.35,
        framestyle = :box,
        left_margin = 8Plots.mm,
        bottom_margin = 14Plots.mm,
        right_margin = 4Plots.mm,
        top_margin = 4Plots.mm,
        xtickfontsize = n_terms > 40 ? 5 : 7,
        ytickfontsize = 9,
        guidefontsize = 11,
        legendfontsize = 9,
    )
    hline!(p_bar, [0.0]; color = :black, lw = 0.8, label = "")

    for c in 1:C
        xs = (1:n_terms) .+ (c - (C + 1) / 2) * bar_w
        ys = [symlog_y(Ξ[j, c]; linthresh) for j in idxs]
        bar!(
            p_bar, xs, ys;
            bar_width = bar_w * 0.92,
            color = colors[c],
            linecolor = colors[c],
            label = "dx_c$(c-1)",
            alpha = 0.9,
        )
    end

    tick_vals = Float64[-1, -0.3, -0.1, -0.03, 0, 0.03, 0.1, 0.3, 1]
    tick_pos = symlog_y.(tick_vals; linthresh)
    tick_lab = ["−1", "−0.3", "−0.1", "−0.03", "0", "0.03", "0.1", "0.3", "1"]
    yticks!(p_bar, tick_pos, tick_lab)
    xticks!(p_bar, 1:n_terms, term_labs; rotation = 55)
    xlims!(p_bar, 0.3, n_terms + 0.7)
    title!(p_bar, "Learned coefficients", titlefontsize = 13, titlelocation = :left)

    fig = plot(
        p_eq, p_bar;
        layout = Plots.grid(2, 1; heights = [0.28, 0.72]),
        size = (1500, 220 + 28 * C + 520),
    )
    savefig(fig, out_path)
    return fig
end
