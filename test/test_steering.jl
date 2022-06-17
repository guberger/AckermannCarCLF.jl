module TestSteering

using PyPlot

include("../src/AckermannCarCLF.jl")
ACC = AckermannCarCLF

δmax = 0.3
αmax = 2.1*π

V = ACC.QuadForm(1.0/δmax^2, 1.1/αmax^2, 0.4/(δmax*αmax))
ν = ACC.level_roa(V, δmax, αmax)*0.8

display(ν)

fig = figure(0)
ax = fig.add_subplot()

x1_ = range(-δmax, δmax, length=200)
x2_ = range(-αmax, αmax, length=200)
X_ = Iterators.product(x1_, x2_)
X1 = getindex.(X_, 1)
X2 = getindex.(X_, 2)
VX = map(x -> ACC._eval(V, x...), X_)
ax.contour(X1, X2, VX, levels=(ν,))

ndisc = 100
ωopt = ACC.steering_opt(V, ν, δmax, αmax, ndisc, 100)

display(ωopt)

ΩX = Matrix{Float64}(undef, size(X_))

for (i, x) in enumerate(X_)
    δ, α = x
    dVdδ = V.a*δ + V.t*α
    dVdα = abs(V.t*δ + V.b*α)
    ΩX[i] = dVdδ*sin(α) - dVdα*ωopt
end

ax.contour(X1, X2, ΩX, levels=(0.0,))

end # module