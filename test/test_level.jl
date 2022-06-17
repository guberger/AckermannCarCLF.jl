module TestLevel

using PyPlot

include("../src/AckermannCarCLF.jl")
ACC = AckermannCarCLF

δmax = 0.3
αmax = π/4
δreq = 0.1
αreq = π/8

V = ACC.QuadForm(1.0/δmax^2, 1.1/αmax^2, 0.4/(δmax*αmax))
ν = ACC.level_roa(V, δmax, αmax)
μ = ACC.level_inner(V, δreq, αreq)

display(ν)
display(μ)

fig = figure(0)
ax = fig.add_subplot()

x1_ = range(-δmax, δmax, length=200)
x2_ = range(-αmax, αmax, length=200)
X_ = Iterators.product(x1_, x2_)
X1 = getindex.(X_, 1)
X2 = getindex.(X_, 2)
VX = map(x -> ACC._eval(V, x...), X_)
ax.contour(X1, X2, VX, levels=(ν,))
ax.contour(X1, X2, VX, levels=(μ,))
inner_circle = [
    (δreq*cos(η), αreq*sin(η)) for η in range(0, 2π, length=100)
]
ax.plot(getindex.(inner_circle, 1), getindex.(inner_circle, 2))

end # module