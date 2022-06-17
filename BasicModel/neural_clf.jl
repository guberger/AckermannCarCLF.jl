using Symbolics
using PyPlot

@variables r θ

V = tanh(
    0.42504429817199707 - 1.041951060295105*tanh(
        (-0.84225171804428101 - 0.017523517832159996*r + 0.19419780373573303*θ)
    ) + 0.50664299726486206*tanh(
        (-0.49960985779762268 + 1.3034340143203735*r - 0.31105217337608337*θ)
    ) - 1.2314140796661377*tanh(
        (0.53244763612747192 + 0.84670799970626831*r + 1.2621999979019165*θ)
    ) - 0.80811703205108643*tanh(
        (0.65992987155914307 - 1.3125202655792236*r - 1.5518209934234619*θ)
    ) - 0.76920902729034424*tanh(
        (0.71334123611450195 + 0.6927187442779541*r - 1.5707864761352539*θ)
    ) + 0.9385685920715332*tanh(
        (1.0288400650024414 + 0.035149402916431427*r - 0.13390231132507324*θ)
    )
)

V_expr = build_function(V, [r, θ])
V_ = eval(V_expr)

DV = Symbolics.jacobian([V], [r, θ])

DVr_expr = build_function(DV[1], [r, θ])
DVr_ = eval(DVr_expr)
DVθ_expr = build_function(DV[2], [r, θ])
DVθ_ = eval(DVθ_expr)

radinner = 0.2
radouter = 0.8
nrad = 1000
nangle = 1000

# Max value on boundary of safe region:
function compute_max_val(V_, radouter, nangle)
    Vmax = Inf
    for α in range(0, 2π, length=nangle)
        x = (cos(α)*radouter, sin(α)*radouter)
        Vx = V_(x)
        Vmax = min(Vmax, Vx)
    end
    return Vmax
end

Vmax = compute_max_val(V_, radouter, nangle)
display(Vmax)
# 0.44958110066792134

function compute_max_steering(DVr_, DVθ_, radinner, radouter, nrad, nangle)
    ω = -Inf
    for α in range(0, 2π, length=nangle)
        for r in range(radinner, radouter, length=nrad)
            r, θ = cos(α)*radouter, sin(α)*radouter
            num = DVr_((r, θ))*sin(θ)
            den = abs(DVθ_((r, θ)))
            ω = max(ω, num/den)
        end
    end
    return ω
end

ωmax = compute_max_steering(DVr_, DVθ_, radinner, radouter, nrad, nangle)
display(ωmax)
# 0.49623607315111723

################################################################################
fig = figure(0)
ax = fig.add_subplot()
np = 200
αs = range(0, 2π, length=np)
c_inner = [(radinner*cos(α), radinner*sin(α)) for α in αs]
c_outer = [(radouter*cos(α), radouter*sin(α)) for α in αs]
ax.plot(getindex.(c_inner, 1), getindex.(c_inner, 2))
ax.plot(getindex.(c_outer, 1), getindex.(c_outer, 2))

x1_ = range(-radouter, radouter, length=200)
x2_ = range(-radouter, radouter, length=200)
X_ = Iterators.product(x1_, x2_)
X1 = getindex.(X_, 1)
X2 = getindex.(X_, 2)
VX = map(x -> V_(x), X_)
ax.contour(X1, X2, VX, levels=(Vmax*0.99,))

# Describe the car dynamics:
# r' = `v`*sin(θ)
# θ' = `v`/`L`*tan(u)
struct CarDynamics
    v::Float64
    L::Float64
end

function optimal_input(car::CarDynamics, V_, r, θ, dt, inputs)
    input_opt = NaN
    Vxnext_opt = Inf
    for input in inputs
        rnext = r + dt*car.v*sin(θ)
        θnext = θ + dt*input
        Vxnext = V_((rnext, θnext))
        if Vxnext < Vxnext_opt
            input_opt = input
            Vxnext_opt = Vxnext
        end
    end
    return input_opt    
end

function simulate_circle(
        car::CarDynamics, V_, R, p10, p20, δ0, dt, nstep, umax, ninput
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
        input = optimal_input(car, V_, r, θ, dt, inputs)
        p1next = p1 + dt*car.v*cos(δ)
        p2next = p2 + dt*car.v*sin(δ)
        δnext = δ + dt*(steering_ref + input)
        x_list[i] = [p1next, p2next, δnext]
    end
    return x_list
end

car = CarDynamics(1.0, 0.3)
R = 1.1/(1/car.L - ωmax)
# R = 2.0
display(R)
# minimal radius: 0.3524729356425989
r0, θ0 = rmax/2, θmax/2
p10, p20, δ0 = R + r0, 0.0, π/2 - θ0
p10, p20, δ0 = R + 0.2, 0.0, π/3
dt = 0.001
tmax = 100.0
nstep = ceil(Int, tmax/dt)
umax = π/4
ninput = 10
x_list = simulate_circle(car, V_, R, p10, p20, δ0, dt, nstep, umax, ninput)

fig = figure(2)
ax = fig.add_subplot()
circle = [R*[cos(α), sin(α)] for α in range(0, 2π, length=200)]
ax.plot(getindex.(circle, 1), getindex.(circle, 2))
ax.plot(getindex.(x_list, 1), getindex.(x_list, 2))
