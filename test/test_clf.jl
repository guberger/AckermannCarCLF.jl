module TestCLF

using PyPlot

include("../src/AckermannCarCLF.jl")
ACC = AckermannCarCLF

δmax = 0.3
αmax = π/4
δreq = δmax/2
αreq = αmax/2
γreq = 0.2
sϵ = 1.01
nη = 100
ndisc = 100
L = 0.34
ωmax = 1/L
V, t = ACC.compute_clf(ωmax, δmax, αmax, δreq, αreq, γreq, sϵ, nη, ndisc)

display(V)
display(t)

ν = ACC.level_roa(V, δmax, αmax)
μ = ACC.level_inner(V, δreq, αreq)

fig = figure(0)
ax = fig.add_subplot()

x1_ = range(-δmax, δmax, length=600)
x2_ = range(-αmax, αmax, length=600)
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

ax.set_xlabel("δ", fontsize=15)
ax.set_ylabel("α", fontsize=15)

ωopt = ACC.steering_opt(V, ν, δmax, αmax, ndisc, ωmax)
γopt = ACC.rate_opt(V, ν, ωmax, δmax, αmax, ndisc)

fig.savefig(
    "./figs/test_clf.png",
    dpi=200, transparent=false, bbox_inches="tight"
)

display(ωopt)
display(sqrt(γopt))

# ΩX = Matrix{Float64}(undef, size(X_))
# ΓX = Matrix{Float64}(undef, size(X_))

# for (i, x) in enumerate(X_)
#     δ, α = x
#     dVdδ = V.a*δ + V.t*α
#     dVdα = abs(V.t*δ + V.b*α)
#     ΩX[i] = dVdδ*sin(α) - dVdα*ωopt
#     ΓX[i] = VX[i] < μ/5 ? NaN : max(-1, (dVdδ*sin(α) - dVdα*ωmax)/VX[i])
# end

# ax.contour(X1, X2, ΩX, levels=(0.0,))
# h = ax.contourf(X1, X2, ΓX)
# colorbar(h)

end # module