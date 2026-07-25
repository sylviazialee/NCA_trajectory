using Plots
using Statistics

delta = [1, 5, 10, 20, 50]

re_traj =  [[0.100934304, 0.10023865, 0.09897025], 
            [0.042359028, 0.042099558, 0.04282055], 
            [0.033534545, 0.031946305, 0.034928925], 
            [0.024493083, 0.027963394, 0.024658239], 
            [0.02371348, 0.025585482, 0.02818483]]

re_final = [[0.028495625, 0.03255327, 0.029096706], 
            [0.005834855, 0.00974589, 0.0070124688], 
            [0.008009071, 0.006698691, 0.008472141], 
            [0.0066055213, 0.006404556, 0.0043566134], 
            [0.004223203, 0.0033253876, 0.0065013394]]

re_after = [[0.03093106, 0.032690052, 0.028475713], 
            [0.028162029, 0.022932047, 0.017969396], 
            [0.035756353, 0.014194496, 0.018009257], 
            [0.00931597, 0.0074822456, 0.009501956], 
            [0.004006631, 0.003387537, 0.0052170395]]

# 2. Calculate mean and standard deviation for traj final after
traj_means = mean.(re_traj)
traj_stds = std.(re_traj)

final_means = mean.(re_final)
final_stds = std.(re_final)

after_means = mean.(re_after)
after_stds = std.(re_after)


# 3. Plot using a shaded ribbon instead of error bars
ribbon = plot(delta, traj_means,
        ribbon=traj_stds,
        fillalpha=0.2,
        title="Loss vs timesteps",
        xlabel="Number of timesteps between checkpoint",
        ylabel="Loss",
        label="traj",
        xticks=delta,
        markersize=6,
        color=:blue)

plot!(ribbon, delta, final_means,
        ribbon=final_stds,
        fillalpha=0.2,
        title="Loss vs timesteps",
        xlabel="Number of timesteps between checkpoint",
        ylabel="Loss",
        label="final",
        xticks=delta,
        markersize=6,
        color=:red)

plot!(ribbon, delta, after_means,
        ribbon=after_stds,
        fillalpha=0.2,
        title="Loss vs timesteps",
        xlabel="Number of timesteps between checkpoint",
        ylabel="Loss",
        label="after",
        xticks=delta,
        markersize=6,
        color=:cyan)

mkpath("delta_graphs")
savefig(joinpath("delta_graphs", "delta_ribbon.png"))

# 3. Plot using the yerror attribute
bar = plot(delta, traj_means, 
        yerror=traj_stds, 
        title="Loss vs timesteps",
        xlabel="Number of timesteps between checkpoint", 
        ylabel="Loss",
        label="traj",
        xticks=delta,
        markersize=6,
        color=:blue)

plot!(bar, delta, final_means, 
        yerror=final_stds, 
        title="Loss vs timesteps",
        xlabel="Number of timesteps between checkpoint", 
        ylabel="Loss",
        label="final",
        xticks=delta,
        markersize=6,
        color=:red)

plot!(bar, delta, after_means, 
        yerror=after_stds, 
        title="Loss vs timesteps",
        xlabel="Number of timesteps between checkpoint", 
        ylabel="Loss",
        label="after",
        xticks=delta,
        markersize=6,
        color=:cyan)

mkpath("delta_graphs")
savefig(joinpath("delta_graphs", "delta_bar.png"))