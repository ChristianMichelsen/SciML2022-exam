using DifferentialEquations
using Plots
using Flux
using Optimization, OptimizationPolyalgorithms
using Zygote, DiffEqSensitivity
using Random
using PyFormattedStrings
using Measures
using DataDrivenDiffEq
using ModelingToolkit
using LinearAlgebra

using Turing
using Optim
using StatsPlots

###############################################################################

save_figures = true
make_animations = true

###############################################################################

base_plt = plot(
    size = (1000, 600),
    xtickfontsize = 14,
    ytickfontsize = 14,
    xguidefontsize = 16,
    yguidefontsize = 16,
    legendfontsize = 12,
    margin = 5mm,
)


###############################################################################


function damped_harmonic(u, p, t)
    x, v = u
    m, b, k = p


    dx = v
    dv = -1 / m * (k * x + b * v)

    [
        dx
        dv
    ]
end


u0 = [1.0; 0.0] # x, v
m0 = 1.0
p0 = [m0, 0.1, 0.1] # m, b, k
tspan = (0.0, 40.0)

prob_true = ODEProblem(damped_harmonic, u0, tspan, p0)
sol_true = solve(prob_true);


plot(sol_true, label = ["x(t)" "v(t)"], xlabel = "t", ylabel = "f(t)")
plot(sol_true, vars = (1, 2), label = "", xlabel = "x(t)", ylabel = "v(t)")


###############################################################################


function get_data(prob_true)

    sol_data = solve(prob_true, saveat = 1.0)

    noise = 0.1
    sol_u = Array(sol_data)
    sol_u += noise * randn(size(sol_u))
    sol_t = sol_data.t
    sol_x = sol_u[1, :]
    sol_v = sol_u[2, :]

    return sol_t, sol_x
end

sol_t, sol_x = get_data(prob_true)


plot_sol = deepcopy(base_plt)
plot!(plot_sol, xlabel = "t", ylabel = "x(t)")
plot!(plot_sol, sol_true; vars = (0, 1), label = "Truth", lw = 2)
scatter!(plot_sol, sol_t, sol_x; label = "Data", color = 1, markersize = 8)
save_figures && savefig(plot_sol, "figures/truth.pdf")
display(plot_sol)

###############################################################################

tt = tspan[1]:0.01:tspan[2]

Δt_extrapolate = 20
tspan_extrapolate = (tspan[1], tspan[2] + Δt_extrapolate)
tt_extrapolate = tspan[2]:0.01:tspan[2]+Δt_extrapolate

prob_true_extrapolate = ODEProblem(damped_harmonic, u0, tspan_extrapolate, p0)
sol_true_extrapolate = solve(prob_true_extrapolate, saveat = tt_extrapolate)

plot_sol_extrapolated = deepcopy(plot_sol)

plot!(
    plot_sol_extrapolated,
    sol_true_extrapolate.t,
    Array(sol_true_extrapolate)[1, :];
    line = :dash,
    # label = "Truth (extrapolated)",
    label = "",
    color = 1,
    lw = 2,
    xlims = tspan_extrapolate,
)
save_figures && savefig(plot_sol_extrapolated, "figures/truth_extrapolated.pdf")
display(plot_sol_extrapolated)

###############################################################################

Random.seed!(1)

NN_no_ode = Flux.Chain(x -> [x], Flux.Dense(1, 64, tanh), Flux.Dense(64, 1), first)
p_NN_no_ode, re_no_ode = Flux.destructure(NN_no_ode)

function loss_no_ode(p_NN_no_ode)
    NN = re_no_ode(p_NN_no_ode)
    y_hat_no_ode = [NN(t) for t in sol_t]
    y_hat_no_ode_tt = [NN(t) for t in tt]
    ℓ_no_ode = sum(abs2, y_hat_no_ode .- sol_x)
    return ℓ_no_ode, y_hat_no_ode_tt
end

function loss_no_ode(p, _)
    return loss_no_ode(p)
end

loss_no_ode(p_NN_no_ode)[1]
loss_no_ode(p_NN_no_ode)[2];



function plot_NN_no_ode!(plt::Plots.Plot, tt, y_hat_no_ode_tt)
    plot!(plt, tt, y_hat_no_ode_tt; label = "Neural Network", color = 2, lw = 3)
    return plt
end

function plot_NN_no_ode(tt, y_hat_no_ode_tt)
    plt = deepcopy(base_plt)
    plot!(plt, xlabel = "t", ylabel = "x(t)", xlim = tspan, ylim = (-0.8, 1.1))
    scatter!(plt, sol_t, sol_x, label = "Data", color = 1, markersize = 8)
    plot_NN_no_ode!(plt, tt, y_hat_no_ode_tt)
    return plt
end

function plot_NN_no_ode(tt, y_hat_no_ode_tt, iter, ℓ_no_ode)
    plt = plot_NN_no_ode(tt, y_hat_no_ode_tt)
    annotate!(plt, [(31, -0.55, (f"Iteration = #{iter}", 16, :left))])
    annotate!(plt, [(31, -0.7, (f"Loss = {ℓ_no_ode:.3f}", 16, :left))])
    return plt
end
# plt = plot_NN_no_ode(tt, y_hat_no_ode_tt, iter, ℓ_no_ode)



iter_no_ode = 0
animation_no_ode = Animation()
callback_no_ode = function (p, ℓ_no_ode, y_hat_no_ode_tt)

    if make_animations
        plt = plot_NN_no_ode(tt, y_hat_no_ode_tt, iter_no_ode, ℓ_no_ode)
        frame(animation_no_ode, plt)
        # display(plt)
    end

    if iter_no_ode % 100 == 0
        @show ℓ_no_ode, iter_no_ode
    end
    global iter_no_ode += 1

    false
    # l < noise^2 * length(sol_t)
end


optf_no_ode = OptimizationFunction(loss_no_ode, Optimization.AutoZygote())
optprob_no_ode = OptimizationProblem(optf_no_ode, p_NN_no_ode);
optsol_no_ode = solve(optprob_no_ode, PolyOpt(), callback = callback_no_ode);


if make_animations
    ℓ_no_ode, y_hat_no_ode_tt = loss_no_ode(optsol_no_ode)
    for _ = 1:400
        plt = plot_NN_no_ode(tt, y_hat_no_ode_tt, iter_no_ode, ℓ_no_ode)
        frame(animation_no_ode, plt)
    end
    gif(animation_no_ode, "figures/animation_NN_no_ode.gif", fps = 50)
end

plot_sol_no_ode = deepcopy(plot_sol)
plot_NN_no_ode!(plot_sol_no_ode, tt, y_hat_no_ode_tt)
save_figures && savefig(plot_sol_no_ode, "figures/no_ode.pdf")
display(plot_sol_no_ode)


plot_sol_no_ode_extrapolated = deepcopy(plot_sol_no_ode)

plot!(
    plot_sol_no_ode_extrapolated,
    sol_true_extrapolate.t,
    Array(sol_true_extrapolate)[1, :];
    line = :dash,
    label = "",
    color = 1,
    lw = 2,
    xlims = tspan_extrapolate,
)

y_hat_no_ode_extrapolate = [re_no_ode(optsol_no_ode)(t) for t in tt_extrapolate];
plot!(
    plot_sol_no_ode_extrapolated,
    tt_extrapolate,
    y_hat_no_ode_extrapolate;
    label = "",
    line = :dash,
    color = 2,
    lw = 2,
    legend = :topleft,
)

save_figures && savefig(plot_sol_no_ode_extrapolated, "figures/no_ode_etxrapolated.pdf")
display(plot_sol_no_ode_extrapolated)

###############################################################################

Random.seed!(1)

NN = Flux.Chain(x -> [x], Flux.Dense(1, 64, tanh), Flux.Dense(64, 1), first)
p_NN, re = Flux.destructure(NN)

# NN([5, 5])
NN(5)

function neural_ode1(u, p, t)

    x, v = u
    m = m0
    k = abs(p[1])
    p_NN = p[2:end]

    NNv = re(p_NN)(v)
    dx = v
    dy = -1 / m * (k * x + NNv)

    [
        dx
        dy
    ]

end


p1 = [1.0; p_NN]

neuralprob_initial = ODEProblem(neural_ode1, u0, tspan, p1)
neuralsol_initial = solve(neuralprob_initial, saveat = 0.1);
# plot(neuralsol_initial)


function loss_neuralode(p)
    neuralprob = remake(neuralprob_initial; p = p)
    neuralsol = solve(neuralprob, saveat = 1.0)
    ℓ = sum(abs2, Array(neuralsol)[1, :] .- sol_x)
    return ℓ, neuralprob
end

function loss_neuralode(p, _)
    return loss_neuralode(p)
end

loss_neuralode(p1)[1]
# loss_neuralode(p1)[2]

function plot_neural_prob!(plt::Plots.Plot, neuralsol)
    plot!(plt, neuralsol; vars = (0, 1), label = "Neural Network", color = 2, lw = 3)
    return plt
end

function plot_neural_prob(neuralsol)
    plt = deepcopy(base_plt)
    plot!(plt, xlabel = "t", ylabel = "x(t)", xlim = tspan, ylim = (-0.8, 1.1))
    scatter!(plt, sol_t, sol_x, label = "Data", color = 1, markersize = 8)
    plot_neural_prob!(plt, neuralsol)
    return plt
end

function plot_neural_prob(neuralsol, iter, ℓ)
    plt = plot_neural_prob(neuralsol)
    annotate!(plt, [(31, -0.55, (f"Iteration = #{iter}", 16, :left))])
    annotate!(plt, [(31, -0.7, (f"Loss = {ℓ:.3f}", 16, :left))])
    return plt

end
# plt = plot_neural_prob(neuralsol_initial, 100, 0.10)



iter = 0
animation = Animation()
callback = function (p, l, neuralprob)

    if make_animations
        neuralsol = solve(neuralprob, saveat = 0.1)
        plt = plot_neural_prob(neuralsol, iter, l)
        frame(animation, plt)
        # display(plt)
    end

    global iter += 1
    if iter % 10 == 0
        @show l, iter
    end
    false
    return l < 0.10
end

optf = OptimizationFunction(loss_neuralode, Optimization.AutoZygote())
optprob = OptimizationProblem(optf, p1);
optsol = solve(optprob, PolyOpt(); callback = callback);

if make_animations
    neuralsol = solve(remake(neuralprob_initial; p = optsol), saveat = 0.1)
    ℓ = loss_neuralode(optsol)[1]
    for _ = 1:50
        plt = plot_neural_prob(neuralsol, iter, ℓ)
        frame(animation, plt)
    end
    gif(animation, "figures/animation_NN_with_ode.gif", fps = 15)
end

neuralprob_optsol = ODEProblem(neural_ode1, u0, tspan, optsol)
neuralsol_optsol = solve(neuralprob_optsol, saveat = 0.01);

plot_sol_ode = deepcopy(plot_sol)
plot_neural_prob!(plot_sol_ode, neuralsol_optsol)
save_figures && savefig(plot_sol_ode, "figures/with_ode.pdf")
display(plot_sol_ode)


neuralprob_optsol_extrapolate = ODEProblem(neural_ode1, u0, tspan_extrapolate, optsol)
neuralsol_optsol_extrapolate = solve(neuralprob_optsol_extrapolate, saveat = tt_extrapolate);


plot_sol_ode_extrapolated = deepcopy(plot_sol_extrapolated)
plot_neural_prob!(plot_sol_ode_extrapolated, neuralsol_optsol)
plot!(
    plot_sol_ode_extrapolated,
    neuralsol_optsol_extrapolate.t,
    Array(neuralsol_optsol_extrapolate)[1, :];
    line = :dash,
    label = "",
    color = 2,
    lw = 2,
    xlim = tspan_extrapolate,
)
save_figures && savefig(plot_sol_ode_extrapolated, "figures/with_ode_extrapolated.pdf")
display(plot_sol_ode_extrapolated)

###############################################################################



NN_optsol = re(optsol[2:end])

xx = -10:0.1:10
plot_nn = deepcopy(base_plt)
plot!(
    plot_nn,
    xx,
    [NN_optsol(x) for x in xx];
    label = "",
    legend = :topleft,
    xlabel = "v",
    ylabel = "NN(v)",
    lw = 2,
)

save_figures && savefig(plot_nn, "figures/nn.pdf")
display(plot_nn)


# vv = -1:0.1:1
# surface(xx, vv, [NN_optsol([x, v]) for x in xx, v in vv])
# contour(xx, vv, [NN_optsol([x, v]) for x in xx, v in vv], xlabel = "x", ylabel = "v")


####

plot_phase_space = deepcopy(base_plt)
plot!(plot_phase_space, xlabel = "x(t)", ylabel = "v(t)")
plot!(plot_phase_space, sol_true, vars = (1, 2), label = "Truth", lw = 2)
plot!(plot_phase_space, neuralsol_optsol, vars = (1, 2), label = "NN", lw = 2)
save_figures && savefig(plot_phase_space, "figures/phase_space.pdf")
display(plot_phase_space)

##############

# f(u) = u.^2 .+ 2.0u .- 1.0
# X = randn(1, 100);
# Y = reduce(hcat, map(f, eachcol(X)));


X = hcat(solve(neuralprob_optsol, saveat = 1.0).u...)
Y = hcat([NN_optsol(u[2]) for u in eachcol(X)]...)

problem = DirectDataDrivenProblem(X, Y, name = :Test)
plot(problem)

@variables u[1:2]
u = collect(u)

h = Num[sin.(u[1]); sin.(u[2]); cos.(u[1]); cos.(u[2]); polynomial_basis(u, 5)];
basis = Basis(h, u);
println(basis)

λs = exp10.(-5:0.1:10);
# opt = ADMM(λs)
opt = STLSQ(λs)
res = solve(problem, basis, opt);
system = result(res);
params = parameters(res);

println(res)
println(system)
println(params)

plot(plot(problem), plot(res), layout = (1, 2))


###############################################################################


function ode_symbolic_regression(u, p, t)

    x, v = u

    m = m0
    k = p[1]
    b = p[2]

    dx = v
    dy = -1 / m * (k * x + b * v)

    [
        dx
        dy
    ]

end

prob_symbolic_regression = ODEProblem(ode_symbolic_regression, u0, tspan)



@model function fitlv1(sol_x, prob)
    # Prior distributions.
    σ ~ InverseGamma(2, 3)
    m = 1.0
    b0 = params[1]
    b ~ truncated(Normal(b0, 0.5), 0.0001, 10)
    k0 = optsol[1]
    k ~ truncated(Normal(k0, 0.5), 0.0001, 10)

    p = [b, k]

    predicted = solve(remake(prob, p = p), Tsit5(); saveat = sol_t, save_idxs = 1)

    # Observations.
    sol_x ~ MvNormal(predicted.u, σ^2 * I)

    return nothing
end


model1 = fitlv1(sol_x, prob_symbolic_regression);

# Generate a MAP estimate.
map_estimate1 = Turing.optimize(model1, MAP())

init_params1 = [map_estimate1.values.array for i = 1:4];
chain1 = sample(model1, NUTS(0.65), MCMCThreads(), 1000, 4; init_params = init_params1)

plot(chain1)



plot_turing = deepcopy(plot_sol)

posterior_samples1 = sample(chain1[[:b, :k]], 300; replace = false)
for p in eachrow(Array(posterior_samples1))
    sol_p = solve(prob_symbolic_regression, Tsit5(); p = p, saveat = 0.1, save_idxs = 1)
    plot!(plot_turing, sol_p; alpha = 0.2, color = "#BBBBBB", label = "")
end


p_turing_mean = mean(chain1[[:b, :k]])[:, :mean]
sol_turing_mean =
    solve(prob_symbolic_regression, Tsit5(); p = p_turing_mean, saveat = 0.1, save_idxs = 1);
plot!(
    plot_turing,
    sol_turing_mean;
    color = "black",
    label = "Turing",
    lw = 2,
    ylim = (-0.8, 1.1),
)

save_figures && savefig(plot_turing, "figures/turing.pdf")
display(plot_turing)
