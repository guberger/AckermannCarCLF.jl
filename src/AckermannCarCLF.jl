module AckermannCarCLF

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

function _sinαmax(α)
    α > π/2 && return 1.0
    α > 0 && return sin(α)
    α > -π && return 0.0
    α > -3*π/2 && return sin(α)
    return 1.0
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

function _rate(V::QuadForm, ν, ωmax, δ, α)
    dVdδ = V.a*δ + V.t*α
    dVdα = abs(V.t*δ + V.b*α)
    val = _eval(V, δ, α)
    scale = sqrt(ν/val)
    t = dVdδ*sin(α*scale)
    return (dVdα*ωmax - t)*scale/ν
end

function rate_opt(V::QuadForm, ν, ωmax, δmax, αmax, ndisc)
    γopt = Inf
    for η in range(0, π, ndisc)
        δ, α = δmax*cos(η), αmax*sin(η)
        γopt = min(γopt, _rate(V, ν, ωmax, δ, α))
    end
    return γopt
end

function compute_clf(ωmax, δmax, αmax, δreq, αreq, γreq, sϵ, nη, ndisc)
    ϵ = 1.0
    ωopt = Inf
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
            !(feasible = ν ≥ μ) && continue
            γ = rate_opt(V, ν, ωmax, δmax, αmax, ndisc)
            γ < γreq && continue
            ω = steering_opt(V, ν, δmax, αmax, ndisc, ωmax*2)
            if ω < ωopt
                ωopt = ω
                Vopt = V
            end
        end
        !feasible && break
        ϵ *= sϵ
    end
    return Vopt, ωopt
end

"""
    CarDynamics

Describe the car dynamics:\n
`\\dot{δ} = v*sin(α)`\n
`\\dot{α} = v/L*tan(u)`
"""
struct CarDynamics
    v::Float64
    L::Float64
end

_dist(x, y, xp, yp) = sqrt((x - xp)^2 + (y - yp)^2)

function _index_waypoint(waypoints, x, y, i, npoint, horiz)
    i_opt = 0
    D_opt = Inf
    irun = mod(i - 1, npoint) + 1
    for k = 1:min(npoint, horiz)
        D = _dist(x, y, waypoints[irun]...)
        if D < D_opt
            i_opt = irun
            D_opt = D
        end
        irun = irun == npoint ? 1 : irun + 1
    end
    return i_opt
end

function _dev(x, y, θ, xp, yp, xpn, ypn)
    dx = xpn - xp
    dy = ypn - yp
    nd = sqrt(dx^2 + dy^2)
    δ = -(dy*(x - xp) - dx*(y - yp))/nd
    θp = atan(dy, dx)
    α = mod(θ - θp + π, 2π) - π
    return δ, α
end

function _dev_waypoint(waypoints, x, y, θ, i, npoint)
    xp, yp = waypoints[i]
    in = mod(i, npoint) + 1
    xpn, ypn = waypoints[in]
    return _dev(x, y, θ, xp, yp, xpn, ypn)
end

function _optimal_update(
        car, V, waypoints, x, y, θ, i, npoint, horiz, dt, inps
    )
    xnext_opt = NaN
    ynext_opt = NaN
    θnext_opt = NaN
    inext_opt = 0
    Vnext_opt = Inf
    for inp in inps
        xnext = x + dt*car.v*cos(θ)
        ynext = y + dt*car.v*sin(θ)
        θnext = θ + dt*car.v*inp
        inext = _index_waypoint(waypoints, xnext, ynext, i, npoint, horiz)
        δnext, αnext = _dev_waypoint(
            waypoints, xnext, ynext, θnext, inext, npoint
        )
        Vnext = _eval(V, δnext, αnext)
        if Vnext < Vnext_opt
            xnext_opt, ynext_opt, θnext_opt = xnext, ynext, θnext
            inext_opt = inext
            Vnext_opt = Vnext
        end
    end
    return xnext_opt, ynext_opt, θnext_opt, inext_opt
end

function simulate_path(
        car, V, waypoints, x0, y0, θ0, dt, nstep, umax, ninp, horiz
    )
    npoint = length(waypoints)
    inp_max = tan(umax)/car.L
    inps = range(-inp_max, inp_max, length=ninp)
    i0 = _index_waypoint(waypoints, x0, y0, 1, npoint, length(waypoints))
    state_list = Vector{Tuple{Float64,Float64,Float64,Int}}(undef, nstep)
    state_list[1] = (x0, y0, θ0, i0)
    for k = 2:nstep
        x, y, θ, i = state_list[k - 1]
        state_list[k] = _optimal_update(
            car, V, waypoints, x, y, θ, i, npoint, horiz, dt, inps
        )
    end
    return state_list
end

end # module
