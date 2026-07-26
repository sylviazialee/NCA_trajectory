using Plots
using Statistics

sigma = ["id + sobel x + sobel y", "sobel x + sobel y", "id", "id + laplacian"]
x = 1:length(sigma)



re_traj = [0.115638554, 0.17839791, 0.19883321, 0.11930079]

re_final =  [0.013856956, 0.11507258, 0.1792718, 0.0102775125]

re_after = [0.033563025, 0.43391296, 0.19705887, 0.024521723]


# 2. Calculate mean and standard deviation for traj final after
traj_means = mean.(re_traj)
traj_stds = std.(re_traj)

final_means = mean.(re_final)
final_stds = std.(re_final)

after_means = mean.(re_after)
after_stds = std.(re_after)


# 3. Plot using a shaded ribbon instead of error bars
#=
ribbon = plot(x, traj_means,
        title="Loss vs Kernel Set",
        xlabel="kernels",
        ylabel="Relative Error Loss",
        label="traj",
        xticks=(x, string.(sigma)),
        markersize=6,
        color=:blue)

plot!(ribbon, x, final_means,
        label="final",
        xticks=(x, string.(sigma)),
        markersize=6,
        color=:red)

plot!(ribbon, x, after_means,
        label="after",
        xticks=(x, string.(sigma)),
        markersize=6,
        color=:cyan)

mkpath("sigma_graphs")
savefig(joinpath("sigma_graphs", "sigma_ribbon.png"))=#

# 3. Grouped bar plot with error bars (manual offset since this Plots.jl
# install has no bar_position=:group / StatsPlots.groupedbar support)
bw = 0.25

grouped_bar = bar(x .- bw, traj_means;
        yerror=traj_stds,
        bar_width=bw,
        title="Loss vs Kernel Set",
        xlabel="kernels",
        ylabel="Relative Error Loss",
        label="traj",
        color=:blue)

bar!(grouped_bar, x, final_means;
        yerror=final_stds,
        bar_width=bw,
        label="final",
        color=:red)

bar!(grouped_bar, x .+ bw, after_means;
        yerror=after_stds,
        bar_width=bw,
        label="after",
        color=:cyan)

xticks!(grouped_bar, x, string.(sigma))

mkpath("sigma_graphs")
savefig(joinpath("sigma_graphs", "sigma_bar.png"))
