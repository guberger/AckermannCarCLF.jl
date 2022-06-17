module TestMain

using LinearAlgebra
using PyPlot

"""
    QuadForm

Describe a quadratic form: `V(δ,α) = a*δ^2 + 2*t*δ*α + b*α^2`
"""
struct QuadForm
    a::Float64
    b::Float64
    t::Float64
end

_eval(V::QuadForm, δ, α) = V.a*δ^2 + V.t*2*δ*α + V.b*α^2

function level_roa(V::QuadForm, δmax, αmax)
    level1 = (-(V.t*δmax)^2 + V.a*δmax^2*V.b)/V.b
    level2 = (-(V.t*αmax)^2 + V.b*αmax^2*V.a)/V.a
    return min(level1, level2)
end

function level_inner(V::QuadForm, δreq, αreq)
    Δ = (V.t*δreq*αreq)^2 - V.a*V.b*(δreq*αreq)^2
    S = V.a*δreq^2 + V.b*αreq^2
    return (S+sqrt(S^2+4*Δ))/2
end

δmax = 0.3
αmax = π/4
δreq = 0.1
αreq = π/8
V = QuadForm(1.0/δmax^2, 1.1/αmax^2, 0.4/(δmax*αmax))
ν = level_roa(V, δmax, αmax)
μ = level_inner(V, δreq, αreq)

display(ν)
display(μ)

fig = figure(0)
ax = fig.add_subplot()

x1_ = range(-δmax, δmax, length=200)
x2_ = range(-αmax, αmax, length=200)
X_ = Iterators.product(x1_, x2_)
X1 = getindex.(X_, 1)
X2 = getindex.(X_, 2)
VX = map(x -> _eval(V, x...), X_)
ax.contour(X1, X2, VX, levels=(ν,))
ax.contour(X1, X2, VX, levels=(μ,))
inner_circle = [
    (δreq*cos(η), αreq*sin(η)) for η in range(0, 2π, length=100)
]
ax.plot(getindex.(inner_circle, 1), getindex.(inner_circle, 2))

function _sinαmax(α)
    α < -π/2 && return -1.0
    α < 0 && return sin(α)
    α < π && return 0.0
    α < 3*π/2 && return sin(α)
    return -1.0
end

function _steering(V::QuadForm, ν, δ, α, ωub)
    dVdδ = V.a*δ + V.t*α
    dVdα = abs(V.t*δ + V.b*α)
    val = _eval(V, δ, α)
    αs = α*sqrt(ν/val)
    s = sign(dVdδ)
    σ = s*_sinαmax(s*αs)
    t = dVdδ*σ
    t > ωub*dVdα && return ωub
    return t/dVdα
end

function steering_opt(V::QuadForm, ν, δmax, αmax, ndisc, ωub)
    ωopt = -Inf
    for η in range(0, π, ndisc)
        δ, α = δmax*cos(η), αmax*sin(η)
        ωopt = max(ωopt, _steering(V, ν, δ, α, ωub))
    end
    return ωopt
end

# function rate_opt(V::QuadForm, ν, ωmax, δmax, αmax, ndisc)
#     γopt = Inf
#     for η in range(0, π, ndisc)
#         δ, α = δmax*cos(η), αmax*sin(η)
#         val = _eval(V, δ, α)
#         δ, α = (δ, α).*sqrt(ν/val)
#         term1 = abs(V.t*δ + V.b*α)*ωmax
#         term2 = max(0, V.a*δ*sin(α) + V.t*α*sin(α))
#         γopt = min(γopt, (term1 - term2)/ν)
#     end
#     return γopt
# end

function compute_clf(ωmax, δmax, αmax, δreq, αreq, γreq, sϵ, nη, ndisc)
    ϵ = 1.0
    topt = Inf
    Vopt = QuadForm(NaN, NaN, NaN)
    while true
        feasible = false
        for η in range(-π/2, π/2, length=nη)
            a = ((cos(η))^2 + (sin(η)*ϵ)^2)/δmax^2
            b = ((sin(η))^2 + (cos(η)*ϵ)^2)/αmax^2
            t = cos(η)*sin(η)*(1-ϵ^2)/(δmax*αmax)
            V = QuadForm(a, b, t)
            ν = level_roa(V, δmax, αmax)
            μ = level_inner(V, δreq, αreq)
            feasible = ν ≥ μ
            !feasible && continue
            # γ = rate_opt(V, ν, ωmax, δmax, αmax, ndisc)
            # γ < γreq && continue
            t = steering_opt(V, ν, δmax, αmax, ndisc, ωmax)
            if t < topt
                topt = t
                Vopt = V
            end
        end
        !feasible && break
        ϵ *= sϵ
    end
    return Vopt, topt
end

δreq = δmax/2
αreq = αmax/2
γreq = 0.2
sϵ = 1.01
nη = 100
ndisc = 100
L = 0.34
ωmax = 1/L
V, t = compute_clf(ωmax, δmax, αmax, δreq, αreq, γreq, sϵ, nη, ndisc)

display(V)
display(t)

ν = level_roa(V, δmax, αmax)
μ = level_inner(V, δreq, αreq)

fig = figure(1)
ax = fig.add_subplot()

x1_ = range(-δmax, δmax, length=200)
x2_ = range(-αmax, αmax, length=200)
X_ = Iterators.product(x1_, x2_)
X1 = getindex.(X_, 1)
X2 = getindex.(X_, 2)
VX = map(x -> _eval(V, x...), X_)
ax.contour(X1, X2, VX, levels=(ν,))
ax.contour(X1, X2, VX, levels=(μ,))
inner_circle = [
    (δreq*cos(η), αreq*sin(η)) for η in range(0, 2π, length=100)
]
ax.plot(getindex.(inner_circle, 1), getindex.(inner_circle, 2))

#=
function steering_opt(V::QuadForm, ν, δmax, αmax, npoint::Int)
    ω = -Inf
    for r in range(-δmax, δmax, length=npoint)
        
    end
    for θ in range(-αmax, αmax, length=npoint)
        num = V.a*δmax*sin(θ) + V.t*θ*sin(θ)
        den = abs(V.t*δmax + V.b*θ)
        ω = max(ω, num/den)
    end
    return ω
end

function rate_opt(V::QuadForm, δmax, αmax, npoint::Int)
    ω = -Inf
    for r in range(-δmax, δmax, length=npoint)
        num = V.a*r*sin(αmax) + V.t*αmax*sin(αmax)
        den = abs(V.t*r + V.b*αmax)
        ω = max(ω, num/den)
    end
    for θ in range(-αmax, αmax, length=npoint)
        num = V.a*δmax*sin(θ) + V.t*θ*sin(θ)
        den = abs(V.t*δmax + V.b*θ)
        ω = max(ω, num/den)
    end
    return ω
end

# Compute a quadratic clf `V` with required curvatue bounded by `ωmax` that
# minimizes the eccentricity
function compute_clf(δmax, αmax, ωmax, ϵmax, nangle, nϵ, npoint)
    αs = range(0, π, length=nangle)
    for ϵ in range(1, ϵmax, length=nϵ)
        for α in αs
            # U = [cos(α) -sin(α)/ϵ; sin(α) cos(α)/ϵ]
            # P = U'*U/γ
            γ1 = ((cos(α))^2 + (sin(α)*ϵ)^2)
            γ2 = ((sin(α))^2 + (cos(α)*ϵ)^2)
            γ = max(γ1, γ2)
            a = ((cos(α))^2 + (sin(α)/ϵ)^2)*γ/δmax^2
            b = ((sin(α))^2 + (cos(α)/ϵ)^2)*γ/αmax^2
            t = (cos(α)*sin(α)*(1-1/ϵ^2))*γ/(δmax*αmax)
            V = QuadForm(a, b, t)
            ω = required_curvature(V, δmax, αmax, npoint)
            ω ≤ ωmax && return V, ω
        end
    end
    return QuadForm(NaN, NaN, NaN), Inf
end

δmax = 0.3
αmax = π/4
ωmax = 0.36
ϵmax = 10.0
V, ω = compute_clf(δmax, αmax, ωmax, ϵmax, 100, 100, 50)
display(V)
display(ω)

fig = figure(0)
ax = fig.add_subplot()

x1_ = range(-δmax, δmax, length=200)
x2_ = range(-αmax, αmax, length=200)
X_ = Iterators.product(x1_, x2_)
X1 = getindex.(X_, 1)
X2 = getindex.(X_, 2)
VX = map(x -> _eval(V, x...), X_)
ax.contour(X1, X2, VX, levels=(1,))

# Describe the car dynamics:
# r' = `v`*sin(θ)
# θ' = `v`/`L`*tan(u)
struct CarDynamics
    v::Float64
    L::Float64
end

function optimal_input(car::CarDynamics, V::QuadForm, r, θ, dt, inputs)
    input_opt = NaN
    Vxnext_opt = Inf
    for input in inputs
        rnext = r + dt*car.v*sin(θ)
        θnext = θ + dt*input
        Vxnext = _eval(V, rnext, θnext)
        if Vxnext < Vxnext_opt
            input_opt = input
            Vxnext_opt = Vxnext
        end
    end
    return input_opt    
end

function simulate_line(
        car::CarDynamics, V::QuadForm, r0, θ0, dt, nstep, umax, ninput
    )
    x_list = Vector{Vector{Float64}}(undef, nstep)
    x_list[1] = [r0, θ0]
    input_max = car.v/car.L*tan(umax)
    inputs = range(-input_max, input_max, length=ninput)
    for i = 2:nstep
        r, θ = x_list[i - 1]
        input = optimal_input(car, V, r, θ, dt, inputs)
        rnext = r + dt*car.v*sin(θ)
        θnext = θ + dt*input
        x_list[i] = [rnext, θnext]
    end
    return x_list
end

car = CarDynamics(1.0, 0.3)
r0, θ0 = 1.0, 1.0
n0 = sqrt(_eval(V, r0, θ0))
r0, θ0 = r0/n0, θ0/n0
dt = 0.01
tmax = 100.0
nstep = ceil(Int, tmax/dt)
umax = atan(ωmax*car.L)
ninput = 10
x_list = simulate_line(car, V, r0, θ0, dt, nstep, umax, ninput)

fig = figure(1)
ax_ = fig.subplots(3)
tdom = range(0, tmax, length=nstep)
ax_[1].plot(tdom, getindex.(x_list, 1))
ax_[2].plot(tdom, getindex.(x_list, 2))
ax_[3].plot(tdom, map(x -> _eval(V, x...), x_list))

function simulate_circle(
        car::CarDynamics, V::QuadForm, R, p10, p20, δ0, dt, nstep, umax, ninput
    )
    x_list = Vector{Vector{Float64}}(undef, nstep)
    x_list[1] = [p10, p20, δ0]
    input_max = car.v/car.L*tan(umax)
    steering_ref = car.v/R
    inputs = range(
        -steering_ref - input_max, input_max - steering_ref, length=ninput
    )
    for i = 2:nstep
        p1, p2, δ = x_list[i - 1]
        r = R - sqrt(p1^2 + p2^2)
        θ = mod(δ - atan(p2, p1) + π/2, 2π) - π
        # display(δ - atan(p2, p1) - π/2)
        @assert -π/4 < θ < π/4
        input = optimal_input(car, V, r, θ, dt, inputs)
        p1next = p1 + dt*car.v*cos(δ)
        p2next = p2 + dt*car.v*sin(δ)
        δnext = δ + dt*(steering_ref + input)
        x_list[i] = [p1next, p2next, δnext]
    end
    return x_list
end

car = CarDynamics(1.0, 0.3)
R = 0.34
display(1/(1/car.L - ωmax))
p10, p20, δ0 = R + 0.2, 0.0, π/3 
n0 = sqrt(_eval(V, r0, θ0))
r0, θ0 = r0/n0, θ0/n0
dt = 0.001
tmax = 1000.0
nstep = ceil(Int, tmax/dt)
umax = π/4
ninput = 10
x_list = simulate_circle(car, V, R, p10, p20, δ0, dt, nstep, umax, ninput)

fig = figure(2)
ax = fig.add_subplot()
circle = [R*[cos(α), sin(α)] for α in range(0, 2π, length=200)]
ax.plot(getindex.(circle, 1), getindex.(circle, 2))
ax.plot(getindex.(x_list, 1), getindex.(x_list, 2))
=#

end # module