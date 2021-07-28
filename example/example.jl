
# NONLEPUS CONTROL
#integrator = DiscretizeRelax(prob, DynamicBoundspODEsDiscrete.LohnerContractor{4}(), h = 0.01, skip_step2 = false, relax = use_relax)
#integrator = DiscretizeRelax(prob, DynamicBoundspODEsDiscrete.LohnerContractor{4}(), h = 0.025, skip_step2 = false, relax = use_relax)

# LEPUS CONTROL
#integrator = DiscretizeRelax(prob, DynamicBoundspODEsDiscrete.LohnerContractor{6}(), h = 0.02,
#                             repeat_limit = 1, skip_step2 = false, step_limit = 5, relax = use_relax)

# TODO: need to adjust coefficient from hoe test...

#using Revise
using IntervalArithmetic, TaylorSeries
setrounding(Interval, :none)
import Base: literal_pow, ^
import IntervalArithmetic.pow
function ^(x::Interval{Float64}, n::Integer)  # fast integer power
    if n < 0
        return 1/IntervalArithmetic.pow(x, -n)
    end
    isempty(x) && return x
    if iseven(n) && 0 ∈ x
        return IntervalArithmetic.hull(zero(x),
                    hull(Base.power_by_squaring(Interval(mig(x)), n),
                        Base.power_by_squaring(Interval(mag(x)), n))
            )
    else
      return IntervalArithmetic.hull( Base.power_by_squaring(Interval(x.lo), n),
                    Base.power_by_squaring(Interval(x.hi), n) )
    end
end

using DynamicBoundsBase, Plots, DifferentialEquations#, Cthulhu
using DynamicBoundspODEsDiscrete, BenchmarkTools

println(" ")
println(" ------------------------------------------------------------ ")
println(" ------------- PACKAGE EXAMPLE       ------------------------ ")
println(" ------------------------------------------------------------ ")

use_relax = false
lohners_type = 2
prob_num = 1
ticks = 50.0
steps = 50.0
tend = 0.02*steps/ticks # lo 7.6100

if prob_num == 1
    x0(p) = [9.0]
    function f!(dx, x, p, t)
        dx[1] = p[1] - x[1]^2 #x[1]*x[1]
        nothing
    end
    tspan = (0.0, tend)
    pL = [-1.0]
    pU = [1.0]

elseif prob_num == 2
    x0(p) = [1.2; 1.1]
    function f!(dx, x, p, t)
        dx[1] = p[1]*x[1]*(one(typeof(p[1])) - x[2])
        dx[2] = p[1]*x[2]*(x[1] - one(typeof(p[1])))
        nothing
    end
    tspan = (0.0, tend)
    pL = [2.95]
    pU = [3.05]
end

prob = DynamicBoundsBase.ODERelaxProb(f!, tspan, x0, pL, pU)
tol = 1E-5

if lohners_type == 1
    integrator = DiscretizeRelax(prob, DynamicBoundspODEsDiscrete.LohnerContractor{5}(), h = 1/ticks,
                                 repeat_limit = 1, skip_step2 = false, step_limit = steps, relax = use_relax, tol= tol)
elseif lohners_type == 2
    integrator = DiscretizeRelax(prob, DynamicBoundspODEsDiscrete.HermiteObreschkoff(3, 3), h = 1/ticks,
                             repeat_limit = 1, skip_step2 = false, step_limit = steps, relax = use_relax, tol= tol)
elseif lohners_type == 3
    function iJx!(dx, x, p, t)
        dx[1] = -2.0*x[1]
        nothing
    end
    function iJp!(dx, x, p, t)
        dx[1] = one(p[1])
        nothing
    end
    integrator = DiscretizeRelax(prob, DynamicBoundspODEsDiscrete.AdamsMoulton(2), h = 1/ticks,
                                 repeat_limit = 1, step_limit = steps, skip_step2 = false,
                                 relax = false, Jx! = iJx!, Jp! = iJp!, tol= tol)
end

ratio = rand(1)
pstar = pL.*ratio .+ pU.*(1.0 .- ratio)
setall!(integrator, ParameterValue(), [0.0])
DynamicBoundsBase.relax!(integrator)
integrate!(integrator)

t_vec = integrator.time
if !use_relax
    lo_vec = getfield.(getindex.(integrator.storage[:],1), :lo)
    hi_vec = getfield.(getindex.(integrator.storage[:],1), :hi)
else
    lo_vec = getfield.(getfield.(getindex.(integrator.storage[:],1), :Intv), :lo)
    hi_vec = getfield.(getfield.(getindex.(integrator.storage[:],1), :Intv), :hi)
end

plt = plot(t_vec , lo_vec, label="Interval Bounds 0.0", marker = (:hexagon, 2, 0.6, :green), linealpha = 0.0, legend=:bottomleft)
plot!(plt, t_vec , hi_vec, label="", linealpha = 0.0, marker = (:hexagon, 2, 0.6, :green))

prob = ODEProblem(f!, [9.0], tspan, [-1.0])
sol = solve(prob, Tsit5(), reltol=1e-8, abstol=1e-8)
plot!(plt, sol.t , sol[1,:], label="", linecolor = :red, linestyle = :solid, lw=1.5)

prob = ODEProblem(f!, [9.0], tspan,[1.0])
sol = solve(prob, Tsit5(), reltol=1e-8, abstol=1e-8)
plot!(plt, sol.t , sol[1,:], label="", linecolor = :red, linestyle = :solid, lw=1.5)

ylabel!("x[1] (M)")
xlabel!("Time (seconds)")
display(plt)