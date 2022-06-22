using DifferentialEquations
using Plots
using Flux
using Optimization, OptimizationPolyalgorithms
using Zygote, DiffEqSensitivity
using Random


###############################################################################

function damped_harmonic(u, p, t)
    x, v = u
    m, b, k = p


    dx = v
    dv = -1 / m * (k * x + b * v^3)

    [
        dx
        dv
    ]
end


u0 = [1.0; 0.0] # x, v
m0 = 1.0
p0 = [m0, 1.0, 1.0] # m, b, k
tspan = (0.0, 40.0)

prob_true = ODEProblem(damped_harmonic, u0, tspan, p0)
sol_true = solve(prob_true);

plot(sol_true, label = ["x(t)" "v(t)"], xlabel = "t", ylabel = "f(t)")
plot(sol_true, vars = (1, 2), label = "", xlabel = "x(t)", ylabel = "v(t)")


###############################################################################


function get_data(prob_true)

    sol_data = solve(prob_true, saveat = 1.0)

    noise = 0.05
    sol_u = Array(sol_data)
    sol_u += noise * randn(size(sol_u))
    sol_t = sol_data.t
    sol_x = sol_u[1, :]
    sol_v = sol_u[2, :]

    return sol_t, sol_x
end

sol_t, sol_x = get_data(prob_true)

plot_sol = plot(sol_true; vars = (0, 1), label = "Data", xlabel = "t", ylabel = "x(t)")
scatter!(plot_sol, sol_t, sol_x; label = "", xlabel = "t", ylabel = "f(t)", color = 1)


###############################################################################


Δt_extrapolate = 20
tspan_extrapolate = [tspan[1], tspan[2] + Δt_extrapolate]
tt_extrapolate = tspan[2]:0.01:tspan[2]+Δt_extrapolate

prob_true_extrapolate = ODEProblem(damped_harmonic, u0, tspan_extrapolate, p0)
sol_true_extrapolate = solve(prob_true_extrapolate, saveat = tt_extrapolate)

plot!(
    plot_sol,
    sol_true_extrapolate.t,
    Array(sol_true_extrapolate)[1, :];
    line = :dash,
    label = "",
    color = 1,
    xlims = tspan_extrapolate,
)


###############################################################################

Random.seed!(1)

NN_no_ode = Flux.Chain(x -> [x], Flux.Dense(1, 64, tanh), Flux.Dense(64, 1), first)
p_NN_no_ode, re_no_ode = Flux.destructure(NN_no_ode)

function loss_no_ode(p_NN_no_ode)
    NN = re_no_ode(p_NN_no_ode)
    y_hat = [NN(t) for t in sol_t]
    return sum(abs2, y_hat .- sol_x), y_hat
end

function loss_no_ode(p, _)
    return loss_no_ode(p)
end

loss_no_ode(p_NN_no_ode)[1]
loss_no_ode(p_NN_no_ode)[2]


callback_no_ode = function (p, l, y_hat)

    # plt = plot(sol_t, y_hat)
    # scatter!(plt, sol_t, sol_x)
    # display(plt)

    @show l
    false
    # l < noise^2 * length(sol_t)
end


plot_sol_no_ode = scatter(
    sol_t,
    sol_x;
    label = "",
    xlabel = "t",
    ylabel = "f(t)",
    color = 1,
)
plot!(
    plot_sol_no_ode,
    sol_true_extrapolate.t,
    Array(sol_true_extrapolate)[1, :];
    line = :dash,
    label = "",
    color = 1,
    xlims = tspan_extrapolate,
)

optf_no_ode = OptimizationFunction(loss_no_ode, Optimization.AutoZygote())
optprob_no_ode = OptimizationProblem(optf_no_ode, p_NN_no_ode)
optsol_no_ode = solve(optprob_no_ode, PolyOpt(), callback = callback_no_ode)

loss_no_ode(optsol_no_ode)[1]


tt = tspan[1]:0.01:tspan[2]
y_hat_no_ode = [re_no_ode(optsol_no_ode)(t) for t in tt]
plot!(plot_sol_no_ode, tt, y_hat_no_ode; label = "NN_x", color = 2, lw = 3)

y_hat_no_ode_extrapolate = [re_no_ode(optsol_no_ode)(t) for t in tt_extrapolate]
plot!(
    plot_sol_no_ode,
    tt_extrapolate,
    y_hat_no_ode_extrapolate;
    label = "",
    line = :dash,
    color = 2,
    lw = 2,
)

display(plot_sol_no_ode)



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

neuralprob = ODEProblem(neural_ode1, u0, tspan, p1)
# neuralprob = ODEProblem(neural_ode2, u0, tspan, p2)
# neuralsol = solve(neuralprob, saveat = 0.1)
# plot(neuralsol)


function loss_neuralode(p)
    # neuralprob = ODEProblem(neural_ode1, u0, tspan, p)
    _neuralprob = remake(neuralprob; p = p)
    neuralsol = solve(_neuralprob, saveat = 1.0)
    loss = sum(abs2, Array(neuralsol)[1, :] .- sol_x)
    # if p[1] < 0
    #     @show "got here"
    #     loss += 1000.0
    # end
    return loss, neuralsol
end

function loss_neuralode(p, _)
    return loss_neuralode(p)
end

loss_neuralode(p1)[1]


callback = function (p, l, neuralsol)

    # plt = plot(neuralsol)
    # scatter!(plt, sol_t, sol_x)
    # display(plt)

    @show l
    false
    l < 0.11
end

optf = OptimizationFunction(loss_neuralode, Optimization.AutoZygote())
optprob = OptimizationProblem(optf, p1);
optsol = solve(optprob, PolyOpt(); callback = callback)


neuralprob_optsol = ODEProblem(neural_ode1, u0, tspan, optsol)
neuralsol_optsol = solve(neuralprob_optsol, saveat = 0.01);

plot!(
    plot_sol,
    neuralsol_optsol.t,
    Array(neuralsol_optsol)[1, :];
    label = "NN_x",
    color = 2,
    lw = 3,
)



neuralprob_optsol_extrapolate = ODEProblem(neural_ode1, u0, tspan_extrapolate, optsol)
neuralsol_optsol_extrapolate = solve(neuralprob_optsol_extrapolate, saveat = tt_extrapolate);

plot!(
    plot_sol,
    neuralsol_optsol_extrapolate.t,
    Array(neuralsol_optsol_extrapolate)[1, :];
    line = :dash,
    label = "",
    color = 2,
    lw = 2,
)


display(plot_sol)


###############################################################################

###


using LinearAlgebra

NN_optsol = re(optsol[2:end])

xx = -1:0.1:1
plot(xx, [NN_optsol(x) for x in xx])

# vv = -1:0.1:1
# surface(xx, vv, [NN_optsol([x, v]) for x in xx, v in vv])
# contour(xx, vv, [NN_optsol([x, v]) for x in xx, v in vv], xlabel = "x", ylabel = "v")


####

plot(sol_true, vars = (1, 2))
plot!(neuralsol_optsol, vars = (1, 2))

##############

using DataDrivenDiffEq
using ModelingToolkit
using LinearAlgebra

# f(u) = u.^2 .+ 2.0u .- 1.0
# X = randn(1, 100);
# Y = reduce(hcat, map(f, eachcol(X)));


X = hcat(solve(neuralprob_optsol, saveat = 1.0).u...)
Y = hcat([NN_optsol(u[2]) for u in eachcol(X)]...)

problem = DirectDataDrivenProblem(X, Y, name = :Test)
plot(problem)

# @variables u[1:2]
@variables u[1:2]
# @parameters w[1:4]
u = collect(u)
# w = collect(w)

# h = Num[sin.(w[1].*u[1]); sin.(w[2].*u[2]); cos.(w[3].*u[1]); cos.(w[4].*u[2]); polynomial_basis(u, 2)]
h = Num[sin.(u[1]); sin.(u[2]); cos.(u[1]); cos.(u[2]); polynomial_basis(u, 3)]
# h = Num[sin.(u[1]); sin.(u[2]); cos.(u[1]); cos.(u[2]); u[1]^3; u[2]^3]
basis = Basis(h, u)
println(basis)

λs = exp10.(-5:0.1:10);
# opt = STLSQ(λs)
# opt = STLSQ(exp10(-6))
opt = ADMM(λs)
res = solve(problem, basis, opt);

# res = solve(problem, basis, STLSQ())

system = result(res);
params = parameters(res);

println(res)
println(system)
println(params)

plot(plot(problem), plot(res), layout = (1, 2))


x=x


# _neuralsol_optsol = solve(neuralprob_optsol, saveat = 1);

# X = Array(_neuralsol_optsol)
# t = _neuralsol_optsol.t
# prob = ContinuousDataDrivenProblem(X, t)

# plot(prob)


# DX = Array(_neuralsol_optsol(t, Val{1}))
# scatter(t, DX', label = ["Solution" nothing], color = :red, legend = :bottomright)
# plot!(t, prob.DX', label = ["Linear Interpolation" nothing], color = :black)


# res = solve(prob, DMDSVD())



###############################################################################

using Turing
using LinearAlgebra
using Optim
using StatsPlots


function ode_symbolic_regression_1(u, p, t)

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

prob_symbolic_regression = ODEProblem(ode_symbolic_regression_1, u0, tspan)



@model function fitlv1(sol_x, prob)
    # Prior distributions.
    σ ~ InverseGamma(2, 3)
    # m ~ truncated(Normal(1.0, 0.5), 0.001, 10)
    m = 1.0
    b ~ truncated(Normal(1.0, 0.5), 0.001, 10)
    k0 = optsol[1]
    k ~ truncated(Normal(k0, 0.5), 0.001, 10)

    # u0_x ~ Normal(0.0, 1.0)
    # u0_v ~ Normal(0.0, 1.0)

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




plot_turing = plot(sol_true; vars = (0, 1), label = "Data", xlabel = "t", ylabel = "x(t)")
scatter!(plot_turing, sol_t, sol_x; label = "", xlabel = "t", ylabel = "f(t)", color = 1)


# plot(; legend = false)
posterior_samples1 = sample(chain1[[:σ, :b, :k]], 300; replace = false)
for (i, p) in enumerate(eachrow(Array(posterior_samples1)))
    σ = p[1]
    sol_p =
        solve(prob_symbolic_regression, Tsit5(); p = p[2:end], saveat = 0.1, save_idxs = 1)

    # pdf = MvNormal(sol_p.u, σ^2 * I)
    # μ = sol_p.u

    if i == 1
        label = "Turing"
    else
        label = ""
    end

    plot!(plot_turing, sol_p; alpha = 0.1, color = "#BBBBBB", label = label)
    # plot!(sol_p.t, μ; ribbon = σ, alpha = 0.01, color = "black", fillalpha=0.005)
end


display(plot_turing)