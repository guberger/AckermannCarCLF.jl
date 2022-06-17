# Compute a quadratic Lyapunov function V that maximizes the ratio γ/β where γ
# is the rate of decrease on the 1-level set of V: min {∇V(x)*f(x,u) : u in U}
# is smaller than -γ for all x with V(x) = 1, and β is the largest change in V
# that can be caused by a perturbation of [0, 1] of a state on the 1-level set
# of V: {∇V(x)*[0, 1]} is smaller than β for all x with V(x) = 1.

using LinearAlgebra
using PyPlot

# Describe the car dynamics:
# r' = `v`*sin(θ)
# θ' = `v`/`L`*tan(u)
# with -`umax` ≤ u ≤ `umax`
struct CarDynamics
    v::Float64
    L::Float64
    umax::Float64
end

# Describe a quadratic form: V(r, θ) = r^2*`a` + 2*r*θ*`t` + θ^2*`b`
struct QuadForm
    a::Float64
    b::Float64
    t::Float64
end

eval_form(V::QuadForm, r, θ) = r^2*V.a + 2*r*θ*V.t + θ^2*V.b

# Find that ratio γ/β by sampling `npoint` on the 1-level set of `V`
function max_steering_rate(V::QuadForm, car::CarDynamics, npoint::Int)
    cmax = abs(car.v/car.L*tan(car.umax))
    rate = Inf
    for α in range(0, π, length=npoint)
        r = cos(α)
        θ = sin(α)
        Vx = sqrt(eval_form(V, r, θ))
        r = r/Vx
        θ = θ/Vx
        βx = abs(r*V.t + θ*V.b) # ∇V(x)*[0, 1]
        δx = -(r*V.a + θ*V.t)*car.v*sin(θ)
        # γx = max {δx + βx*v/L*tan(u) : u ∈ U}
        rate = min(rate, δx/βx + cmax)
    end
    return rate
end

# Compute a quadratic clf `V` that maximizes the steering rate and satisfies
# that |r| ≤ `rmax` for all x with `V`(x) ≤ 1.
# This is done by sampling `nangle` angle and `nratio` to define the 1-level set
# of `V`.
function compute_clf(
        car::CarDynamics, rmax::Float64, ϵ::Float64,
        nangle::Int, nratio::Int, npoint::Int
    )
    αs = range(0, π, length=nangle)
    ratios = range(ϵ, 1, length=nratio)
    rate_opt = -Inf
    V_opt = QuadForm(NaN, NaN, NaN)
    for (α, ratio) in Iterators.product(αs, ratios)
        # U = [cos(α) -sin(α)/ratio; sin(α) cos(α)/ratio]
        # P = U'*U/γ
        γ = rmax^2/((cos(α))^2 + (sin(α)*ratio)^2)
        a = ((cos(α))^2 + (sin(α)/ratio)^2)/γ
        b = ((sin(α))^2 + (cos(α)/ratio)^2)/γ
        t = (cos(α)*sin(α)*(1-1/ratio^2))/γ
        V = QuadForm(a, b, t)
        rate = max_steering_rate(V, car, npoint)
        if rate > rate_opt
            rate_opt = rate
            V_opt = V
        end
    end
    return V_opt, rate_opt
end

V, rate = compute_clf(car, 0.2, 0.5, 10, 10, 10)
display(V)
display(rate)

fig = figure(0)
ax = fig.add_subplot()

x1_ = range(-1, 1, length=200)
x2_ = range(-1, 1, length=200)
X_ = Iterators.product(x1_, x2_)
X1 = getindex.(X_, 1)
X2 = getindex.(X_, 2)
VX = map(x -> eval_form(V, x...), X_)
ax.contour(X1, X2, VX, levels=(1,))

function optimal_next_point(car::CarDynamics, V::QuadForm, r, θ, dt, us)
    rnext_opt = NaN
    θnext_opt = NaN
    Vxnext_opt = Inf
    for u in us
        rnext = r + dt*car.v*sin(θ)
        θnext = θ + dt*car.v/car.L*tan(u)
        Vxnext = eval_form(V, rnext, θnext)
        if Vxnext < Vxnext_opt
            rnext_opt = rnext
            θnext_opt = θnext
            Vxnext_opt = Vxnext
        end
    end
    return rnext_opt, θnext_opt    
end

function simulate_straight_line(
        car::CarDynamics, V::QuadForm, r0, θ0, dt, nstep, ninput
    )
    x_list = Vector{Vector{Float64}}(undef, nstep)
    x_list[1] = [r0, θ0]
    us = range(-car.umax, car.umax, length=ninput)
    for i = 2:nstep
        r, θ = x_list[i - 1]
        rnext, θnext = optimal_next_point(car, V, r, θ, dt, us)
        x_list[i] = [rnext, θnext]
    end
    return x_list
end

r0 = 0.3
θ0 = 0.0
dt = 0.01
tmax = 10.0
nstep = ceil(Int, tmax/dt)
x_list = simulate_straight_line(car, V, r0, θ0, dt, nstep, 10)

fig = figure(1)
ax = fig.add_subplot()
tdom = range(0, tmax, length=nstep)
ax.plot(tdom, getindex.(x_list, 1))
ax.plot(tdom, getindex.(x_list, 2))