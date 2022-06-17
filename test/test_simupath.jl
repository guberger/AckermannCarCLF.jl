module TestSimuPath

using PyPlot

include("../src/AckermannCarCLF.jl")
ACC = AckermannCarCLF

δmax = 0.3
αmax = π/4
car = ACC.CarDynamics(1.0, 0.3)
umax = π/4
ωmax = tan(umax)/car.L

ndisc = 100
V = ACC.QuadForm(12.12531344656796, 6.38051268779618, 2.1970361790544004)
ν = ACC.level_roa(V, δmax, αmax)
ωopt = ACC.steering_opt(V, ν, δmax, αmax, ndisc, ωmax)

display(ωopt)

κmax = (ωmax - ωopt)*car.L/(car.L + δmax)
display(κmax)

x_path(s) = [(4 + cos(4*s))*cos(s), (3 + cos(4*s))*sin(s)]
nsample = 250
waypoints = [x_path(s) for s in range(0, 2π, length=nsample)]

fig = figure(0)
ax = fig.add_subplot()

ax.set_xlabel("x", fontsize=15)
ax.set_ylabel("y", fontsize=15)

ax.plot(
    getindex.(waypoints, 1), getindex.(waypoints, 2), marker=".", 
)

car = ACC.CarDynamics(1.0, 0.3)
dt = 0.01
tmax = 100.0
nstep = ceil(Int, tmax/dt)
umax = π/4
ninp = 10
horiz = 20
x0, y0 = waypoints[1] .+ (0.3, 0.0)
θ0 = π/2
state_list = ACC.simulate_path(
    car, V, waypoints, x0, y0, θ0, dt, nstep, umax, ninp, horiz
)

ax.plot(getindex.(state_list, 1), getindex.(state_list, 2))

fig.savefig(
    "./figs/test_simupath.png",
    dpi=200, transparent=false, bbox_inches="tight"
)


end # module