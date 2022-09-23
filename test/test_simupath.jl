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
nsample = 500
waypoints = [x_path(s) for s in range(0, 2π, length=nsample)]

fig = figure(0)
ax = fig.add_subplot()

ax.set_xticks(())
ax.set_yticks(())
# ax.set_xlabel("x", fontsize=15)
# ax.set_ylabel("y", fontsize=15)

ax.plot(
    getindex.(waypoints, 1), getindex.(waypoints, 2), marker=".", 
)

car = ACC.CarDynamics(1.0, 0.3)
dt = 0.01
tmax = 30.0
nstep = ceil(Int, tmax/dt)
umax = π/4
ninp = 31
horiz = 20
x0, y0 = waypoints[1] .+ (0.9, 0.0)
θ0 = π/2
state_list = ACC.simulate_path(
    car, V, waypoints, x0, y0, θ0, dt, nstep, umax, ninp, horiz
)

function _draw_car(L, x, y, θ)
    xp = (1, 1, -1, -1, 1).*L
    yp = (-1, 1, 1, -1, -1).*(L/2)
    xs = x .+ cos(θ).*xp .- sin(θ).*yp
    ys = y .+ sin(θ).*xp .+ cos(θ).*yp
    return xs, ys
end

hcar = ax.plot((), ())[1]
hcar.set_linestyle("-")
hcar.set_color("r")
hcar.set_linewidth(2)
hpoint = ax.plot((), ())[1]
hpoint.set_linestyle("none")
hpoint.set_color("r")
hpoint.set_marker(".")
hpoint.set_markersize(10)
verts_ = (
    ((-1, -1), (-1, 1), (1, 1), (1, -1), (-1, -1)),
    ((-7, -5), (-7, -3), (-3, -3), (-3, -5), (-7, -5)),
    ((7, 5), (7, 3), (3, 3), (3, 5), (7, 5)),
    ((-7, 5), (-7, 3), (-3, 3), (-3, 5), (-7, 5)),
    ((7, -5), (7, -3), (3, -3), (3, -5), (7, -5))
)
polylist = matplotlib.collections.PolyCollection(verts_)
polylist.set_facecolor("r")
polylist.set_edgecolor("r")
polylist.set_linewidth(1.0)
ax.add_collection(polylist)

iter = 0

for state in state_list
    global iter += 1
    mod(iter - 1, 20) != 0 && continue
    x, y, θ, i = state
    xs, ys = _draw_car(car.L, x, y, θ)
    hcar.set_xdata(xs)
    hcar.set_ydata(ys)
    hpoint.set_xdata((waypoints[i][1],))
    hpoint.set_ydata((waypoints[i][2],))
    fig.canvas.draw()
    fig.canvas.flush_events()
    filename = string("./figs/animation_simupath/frame_", iter, ".png")
    # fig.savefig(filename, dpi=100)
end

ax.plot(getindex.(state_list, 1), getindex.(state_list, 2))

fig.savefig(
    "./figs/test_simupath.png",
    dpi=200, transparent=false, bbox_inches="tight"
)

end # module