using Plots
using Statistics

channel = [3, 4, 5, 8, 10, 12, 16]

re_traj =  [[0.03883074, 0.04104177, 0.040239368], 
            [0.03793267, 0.039607566, 0.039355233], 
            [0.03751115, 0.035683267, 0.03569843], 
            [0.033171233, 0.0358713, 0.03403997], 
            [0.03355837, 0.0334032, 0.033621676],
            [0.029920008, 0.030867772, 0.031986993],
            [0.025783608, 0.03425045, 0.02978223]]

re_final = [[0.004753494, 0.0057724672, 0.0055180592],
            [0.00456385, 0.004967892, 0.00904831],
            [0.0058269002, 0.0037707095, 0.007720054],
            [0.0077005937, 0.003144517, 0.0029432385],
            [0.002869236, 0.0050665415, 0.0036479498],
            [0.004029153, 0.006285761, 0.0034085708],
            [0.0051959986, 0.005693106, 0.0046614185]]

re_after = [[0.0059926985, 0.004129243, 0.004676947],
            [0.0070948172, 0.0045068953, 0.0031666006],
            [0.00492769, 0.00467993, 0.0037002107],
            [0.0034268054, 0.0052011022, 0.0028365809],
            [0.0030361125, 0.0042418796, 0.0041666804],
            [0.0035954372, 0.0044898037, 0.007966226],
            [0.008111016, 0.0105864275, 0.00973947]]

# 2. Calculate mean and standard deviation for traj final after
traj_means = mean.(re_traj)
traj_stds = std.(re_traj)

final_means = mean.(re_final)
final_stds = std.(re_final)

after_means = mean.(re_after)
after_stds = std.(re_after)


# 3. Plot using a shaded ribbon instead of error bars
ribbon = plot(channel, traj_means,
        ribbon=traj_stds,
        fillalpha=0.2,
        title="Loss vs number of channels",
        xlabel="Total number of channels",
        ylabel="Relative Error Loss",
        label="traj",
        xticks=channel,
        markersize=6,
        color=:blue)

plot!(ribbon, channel, final_means,
        ribbon=final_stds,
        fillalpha=0.2,
        label="final",
        markersize=6,
        color=:red)

plot!(ribbon, channel, after_means,
        ribbon=after_stds,
        fillalpha=0.2,
        label="after",
        xticks=channel,
        markersize=6,
        color=:cyan)

mkpath("channel_graphs")
savefig(joinpath("channel_graphs", "channel_ribbon.png"))

# 3. Plot using the yerror attribute
bar = plot(channel, traj_means, 
        yerror=traj_stds, 
        title="Loss vs timesteps",
        xlabel="Number of timesteps between checkpoint", 
        ylabel="Relative Error",
        label="traj",
        xticks=channel,
        markersize=6,
        color=:blue)

plot!(bar, channel, final_means, 
        yerror=final_stds, 
        title="Loss vs timesteps",
        xlabel="Number of timesteps between checkpoint", 
        ylabel="Relative Error",
        label="final",
        xticks=channel,
        markersize=6,
        color=:red)

plot!(bar, channel, after_means, 
        yerror=after_stds, 
        title="Loss vs timesteps",
        xlabel="Number of timesteps between checkpoint", 
        ylabel="Relative Error",
        label="after",
        xticks=channel,
        markersize=6,
        color=:cyan)

mkpath("channel_graphs")
savefig(joinpath("channel_graphs", "channel_bar.png"))