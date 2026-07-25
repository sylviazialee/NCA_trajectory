using Plots
using Statistics

sigma = [0.001, 0.005, 0.01, 0.1, 0.2, 0.5]
x = 1:length(sigma)

re_final =  [0.0000156426, 0.0000112056, 0.0000101512, 0.0003198630, 0.0004640043, 0.0217955860]

re_traj = [0.0005194459, 0.0005126157, 0.0006078483, 0.0108069690, 0.0401999580, 0.2547684600]

re_after = [0.0000173781, 0.0000120668, 0.0000085514, 0.0003384795, 0.0004360254, 0.0232914820]


# 2. Calculate mean and standard deviation for traj final after
traj_means = mean.(re_traj)
traj_stds = std.(re_traj)

final_means = mean.(re_final)
final_stds = std.(re_final)

after_means = mean.(re_after)
after_stds = std.(re_after)


# 3. Plot using a shaded ribbon instead of error bars
ribbon = plot(x, traj_means,
        ribbon=traj_stds,
        fillalpha=0.2,
        title="Loss vs random noise",
        xlabel="sigma",
        ylabel="Relative Error Loss",
        label="traj",
        xticks=(x, string.(sigma)),
        markersize=6,
        color=:blue)

plot!(ribbon, x, final_means,
        ribbon=final_stds,
        fillalpha=0.2,
        label="final",
        xticks=(x, string.(sigma)),
        markersize=6,
        color=:red)

plot!(ribbon, x, after_means,
        ribbon=after_stds,
        fillalpha=0.2,
        label="after",
        xticks=(x, string.(sigma)),
        markersize=6,
        color=:cyan)

mkpath("sigma_graphs")
savefig(joinpath("sigma_graphs", "sigma_ribbon.png"))

# 3. Plot using the yerror attribute
bar = plot(x, traj_means,
        yerror=traj_stds,
        title="Loss vs timesteps",
        xlabel="Number of timesteps between checkpoint",
        ylabel="Loss",
        label="traj",
        xticks=(x, string.(sigma)),
        markersize=6,
        color=:blue)

plot!(bar, x, final_means,
        yerror=final_stds,
        title="Loss vs timesteps",
        xlabel="Number of timesteps between checkpoint",
        ylabel="Loss",
        label="final",
        xticks=(x, string.(sigma)),
        markersize=6,
        color=:red)

plot!(bar, x, after_means,
        yerror=after_stds,
        title="Loss vs timesteps",
        xlabel="Number of timesteps between checkpoint",
        ylabel="Loss",
        label="after",
        xticks=(x, string.(sigma)),
        markersize=6,
        color=:cyan)

mkpath("sigma_graphs")
savefig(joinpath("sigma_graphs", "sigma_bar.png"))
