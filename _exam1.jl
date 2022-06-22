using DifferentialEquations

function damped_harmonic(u, p, t)
    x, v = u
    m, b, k = p


    dx = v
    dv = 1 / m * (-b * v - k * x)

    [
        dx
        dv
    ]
end


u0 = [1.0; 0.0]
p0 = [1.0, 0.1, 1.0]
tspan = (0.0, 20.0)

prob_true = ODEProblem(damped_harmonic, u0, tspan, p0)
sol_true = solve(prob_true)

using Plots
plot(sol_true)
plot(sol_true, vars = (1, 2))


sol_data = solve(prob_true, saveat = 1.0)
data = Array(sol_data)
tt = sol_data.t
scatter(tt, data')

using Flux

NN = Flux.Chain(x -> [x], Flux.Dense(1, 64, tanh), Flux.Dense(64, 1), first)
p_NN, re = Flux.destructure(NN)

NN(5)

function neural_ode1(u, p, t)

    x, v = u
    m, b, k = p[1:3]
    p_NN = p[4:end]

    NNu = re(p_NN)(v)
    dx = v
    dy = 1 / m * (-NNu - k * x)

    [
        dx
        dy
    ]

end


function neural_ode2(u, p, t)

    x, v = u
    m, b, k = p0
    p_NN = p

    NNu = re(p_NN)(v)
    dx = v
    dy = 1 / m * (-NNu - k * x)

    [
        dx
        dy
    ]

end

p1 = [p0; p_NN]
p2 = p_NN

# neuralprob = ODEProblem(neural_ode1, u0, tspan, p1)
neuralprob = ODEProblem(neural_ode2, u0, tspan, p2)
neuralsol = solve(neuralprob, saveat = 1.0)
plot!(neuralsol)


function loss_neuralode(p)
    neuralprob = ODEProblem(neural_ode2, u0, tspan, p)
    neuralsol = solve(neuralprob, saveat = 1.0)
    sum(abs2, Array(neuralsol) .- data), neuralsol
end

function loss_neuralode(p, _)
    return loss_neuralode(p)
end

loss_neuralode(p2)[1]

callback = function (p, l, neuralsol)

    # plt = plot(neuralsol)
    # scatter!(plt, sol_data.t, data')
    # display(plt)

    @show l

    l < 0.01
end

# # x = x

using Optimization, OptimizationPolyalgorithms
using Zygote, DiffEqSensitivity

optf = OptimizationFunction(loss_neuralode, Optimization.AutoZygote())
optprob = OptimizationProblem(optf, p2)
optsol = solve(optprob, PolyOpt(), callback = callback)


neuralprob_optsol = ODEProblem(neural_ode2, u0, tspan, optsol)
neuralsol_optsol = solve(neuralprob_optsol, saveat = 0.1)
plot!(neuralsol_optsol)


tspan_extrapolate = [tspan[1], tspan[2] + 20]

neuralprob_optsol_extrapolate = ODEProblem(neural_ode2, u0, tspan_extrapolate, optsol)
neuralsol_optsol_extrapolate = solve(neuralprob_optsol_extrapolate, saveat = 0.01)
plot!(neuralsol_optsol_extrapolate)


prob_true_extrapolate = ODEProblem(damped_harmonic, u0, tspan_extrapolate, p0)
sol_true_extrapolate = solve(prob_true_extrapolate)
plot!(sol_true_extrapolate)


###############################################################################


function loss_no_ode(p_NN)
    NN = re(p_NN)
    y_hat = [NN(t) for t in tt]
    sum(abs2, y_hat .- data[1, :])
end

function loss_no_ode(p, _)
    return loss_no_ode(p)
end

loss_no_ode(p_NN)


callback_no_ode = function (p, l)
    @show l
    l < 0.01
end


optf_no_ode = OptimizationFunction(loss_no_ode, Optimization.AutoZygote())
optprob_no_ode = OptimizationProblem(optf_no_ode, p2)
optsol_no_ode = solve(optprob_no_ode, PolyOpt(), callback = callback_no_ode)

ttt = collect(tspan[1]:0.01:tspan[2])
y_hat_no_ode = [re(optsol_no_ode)(t) for t in ttt]

plot!(ttt, y_hat_no_ode)



###

NN_optsol = re(optsol)

vv = -2:0.01:2
NN_vv = [NN_optsol(v) for v in vv];
yy_vv = [p0[2] * v for v in vv];
plot(vv, yy_vv)
plot!(vv, NN_vv)


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


# X = hcat(solve(neuralprob_optsol, saveat = 1.0).u...)[1:1, :]
X = sort(hcat(solve(neuralprob_optsol, saveat = 1.0).u...)[1:1, :], dims = 2)
Y = hcat([NN_optsol(u[1]) for u in eachcol(X)]...)

problem = DirectDataDrivenProblem(X, Y, name = :Test)

@variables u

# basis = Basis(polynomial_basis(u, 2), u)
basis = Basis(monomial_basis([u], 2), [u])
println(basis)

λs = exp10.(-5:0.1:10)
opt = STLSQ(λs)
opt = STLSQ(exp10(-6))
opt = ADMM(λs)
res = solve(problem, basis, opt)
# res = solve(problem, basis, STLSQ())

system = result(res)
params = parameters(res)

println(res)
println(system)
println(params)


plot(plot(problem), plot(res), layout = (1, 2))


# X = Array(sol)
# t = sol.t
# prob = ContinuousDataDrivenProblem(X, t)
prob = ContinuousDataDrivenProblem(X, Y)
plot(prob)

res = solve(prob, DMDSVD())
println(res)

system = result(res)
# println(system)

plot(res)

metrics(res)
parameters(res)

# Matrix(generator(system))
