module NotebookSupport

using Distributions, Plots, Random

export experiment_rng, sample_distributions, plot_distributions,
    plot_estimator_convergence, plot_accumulated_reward

"""Create a fresh RNG: stream 0 generates the arms; stream 1 samples rewards."""
experiment_rng(seed; stream=0) = Xoshiro(seed + stream)

function sample_distributions(K; seed)
    rng = experiment_rng(seed)
    [truncated(Normal(rand(rng), rand(rng) + 0.1), 0, 1) for _ in 1:K]
end

function save_plot(p, output_dir, filename)
    mkpath(output_dir)
    savefig(p, joinpath(output_dir, filename))
    p
end

function plot_distributions(dists)
    panels = [
        begin
            x = range(quantile(d, 0.001), quantile(d, 0.999), length=400)
            plot(x, pdf.(Ref(d), x), legend=false)
        end
        for (a, d) in enumerate(dists)
    ]

    plot(panels...,
        layout=(1, length(dists)),
        size=(250 * length(dists), 300))
end

function plot_estimator_convergence(μ̂_history, μ, a_max, a_min; output_dir)
    T = size(μ̂_history, 1) - 1
    colors = ["#0072B2", "#D55E00"]

    p = plot(
        xlabel = "Tiradas adicionales (t)",
        ylabel = "Media de recompensa",
        title = "UCB - Convergencia de μ̂",
        legend = :outerright,
        size = (900, 450),
        margin = 5 * Plots.mm,
        dpi = 180
    )

    for (i, a) in enumerate(unique([a_max, a_min]))
        c = colors[i]
        role = a_max == a_min ? "único" : a == a_max ? "mejor" : "peor"

        plot!(p, 0:T, μ̂_history[:, a],
            color=c,
            linewidth=1.5,
            label="μ̂ $role (a=$a)")

        hline!(p, [μ[a]],
            color=c,
            linewidth=1.5,
            linestyle=:dash,
            label="μ $role (a=$a)")
    end

    save_plot(p, output_dir, "estimator_convergence.png")
end

function plot_accumulated_reward(K, best_mean, regret_history, pseudo_regret_history; output_dir)
    T = length(regret_history) - 1
    n = K .+ (0:T)
    optimal_reward = n .* best_mean
    collected_reward = optimal_reward .- regret_history

    reward_plot = plot(n, collected_reward,
        label="UCB (observada)",
        title="Recompensa acumulada",
        ylabel="Recompensa acumulada",
        color="#0072B2",
        linewidth=2,
        legend=:topleft
    )

    plot!(reward_plot, n, optimal_reward,
        label="Siempre el mejor brazo (esperada)",
        color=:black,
        linestyle=:dash,
        linewidth=2)

    regret_plot = plot(n, pseudo_regret_history,
        title="Costo esperado de las decisiones",
        ylabel="Pseudo-regret acumulado",
        color="#D55E00",
        linewidth=2,
        fillrange=0,
        fillalpha=0.1,
        legend=false)

    p = plot(reward_plot, regret_plot,
        layout=(2, 1),
        link=:x,
        xlabel="Tiradas totales (K + t)",
        size=(900, 700),
        margin=5 * Plots.mm,
        dpi=180)

    save_plot(p, output_dir, "acc_reward.png")
end

end
