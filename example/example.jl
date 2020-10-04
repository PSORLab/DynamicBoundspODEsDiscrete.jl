
# NONLEPUS CONTROL
#integrator = DiscretizeRelax(prob, DynamicBoundspODEsDiscrete.LohnerContractor{4}(), h = 0.01, skip_step2 = false, relax = use_relax)
#integrator = DiscretizeRelax(prob, DynamicBoundspODEsDiscrete.LohnerContractor{4}(), h = 0.025, skip_step2 = false, relax = use_relax)

# LEPUS CONTROL
#integrator = DiscretizeRelax(prob, DynamicBoundspODEsDiscrete.LohnerContractor{6}(), h = 0.02,
#                             repeat_limit = 1, skip_step2 = false, step_limit = 5, relax = use_relax)


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
using DynamicBoundspODEsDiscrete

println(" ")
println(" ------------------------------------------------------------ ")
println(" ------------- PACKAGE EXAMPLE       ------------------------ ")
println(" ------------------------------------------------------------ ")

use_relax = false
lohners_type = 1
prob_num = 1
ticks = 100.0
steps = 100.0
tend = steps/ticks

if prob_num == 1
    x0(p) = [9.0]
    function f!(dx, x, p, t)
        dx[1] = p[1] - x[1]^2
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

if lohners_type == 1
    integrator = DiscretizeRelax(prob, DynamicBoundspODEsDiscrete.LohnerContractor{7}(),
                                 repeat_limit = 1, skip_step2 = false, step_limit = steps, relax = false)
elseif lohners_type == 2
    integrator = DiscretizeRelax(prob, DynamicBoundspODEsDiscrete.HermiteObreschkoff(3, 3), h = 1/ticks,
                             repeat_limit = 1, skip_step2 = false, step_limit = steps, relax = use_relax)
elseif lohners_type == 3
    function iJx!(dx, x, p, t)
        dx[1] = -2.0*x[1]
        nothing
    end
    function iJp!(dx, x, p, t)
        dx[1] = one(p[1])
        nothing
    end
    integrator = DiscretizeRelax(prob, DynamicBoundspODEsDiscrete.AdamsMoulton(4), h = 1/ticks,
                                 skip_step2 = false, relax = false, Jx! = iJx!, Jp! = iJp!)
end

#=
integrator = DiscretizeRelax(prob, DynamicBoundspODEsDiscrete.AdamsMoulton(4), h = 0.01,
                             repeat_limit = 1, skip_step2 = false, step_limit = 3, relax = use_relax)

integrator = DiscretizeRelax(prob, DynamicBoundspODEsDiscrete.HermiteObreschkoff(3, 3), h = 0.01,
                             repeat_limit = 1, skip_step2 = false, step_limit = 5, relax = use_relax)
=#
#=
function iJx!(dx, x, p, t)
    dx[1] = -2.0*x[1]
    nothing
end
function iJp!(dx, x, p, t)
    dx[1] = one(p[1])
    nothing
end
integrator = DiscretizeRelax(prob, PLMS(4, AdamsMoulton()), h = 0.01, skip_step2 = false, relax = false, Jx! = iJx!, Jp! = iJp!)
=#

ratio = rand(1)
pstar = pL.*ratio .+ pU.*(1.0 .- ratio)
setall!(integrator, ParameterValue(), [0.0])
DynamicBoundsBase.relax!(integrator)
#println("alloc_num: $(alloc_num)")

d = integrator
#@code_warntype DynamicBoundspODEsPILMS.single_step!(d.step_result, d.step_params, d.method_f!, d.set_tf!, d.Δ, d.A, d.P, d.rP, d.p)
#=
method_f! = d.method_f!
@code_warntype method_f!(d.step_result.steps, d.step_result.times, d.step_result.unique_result.X, d.step_result.Xⱼ,
                            d.step_result.xⱼ, d.A, d.Δ, d.P, d.rP, d.p, d.step_result.unique_result.fk)
                            =#
#using BenchmarkTools
#@btime DynamicBoundsBase.relax!($integrator)

t_vec = integrator.time
if !use_relax
    lo_vec = getfield.(getindex.(integrator.storage[:],1), :lo)
    hi_vec = getfield.(getindex.(integrator.storage[:],1), :hi)
else
    lo_vec = getfield.(getfield.(getindex.(integrator.storage[:],1), :Intv), :lo)
    hi_vec = getfield.(getfield.(getindex.(integrator.storage[:],1), :Intv), :hi)
    #lo_vec = getfield.(getindex.(integrator.storage[:],1), :cv)
    #hi_vec = getfield.(getindex.(integrator.storage[:],1), :cc)
end

plt = plot(t_vec , lo_vec, label="Interval Bounds 0.0", marker = (:hexagon, 2, 0.6, :green), linealpha = 0.0, legend=:bottomleft)
plot!(plt, t_vec , hi_vec, label="", linealpha = 0.0, marker = (:hexagon, 2, 0.6, :green))
#=
prob = DynamicBoundsBase.ODERelaxProb(f!, tspan, x0, pL, pU)
integrator = DiscretizeRelax(prob, h = 0.01, skip_step2 = false, k = kval)
ratio = rand(1)
pstar = pL.*ratio .+ pU.*(1.0 .- ratio)
setall!(integrator, ParameterValue(), [1.0])
DynamicBoundsBase.relax!(integrator)

t_vec = integrator.time
lo_vec = getfield.(getindex.(integrator.storage[:],1), :lo)
hi_vec = getfield.(getindex.(integrator.storage[:],1), :hi)
plot!(plt, t_vec , lo_vec, label="Interval Bounds 1.0", linecolor = :green, linestyle = :dash,
           lw=1.5, legend=:bottomleft)
plot!(plt, t_vec , hi_vec, label="", linecolor = :green, linestyle = :dash, lw=1.5)
=#
prob = ODEProblem(f!, [9.0], tspan, [-1.0])
sol = solve(prob, Tsit5(), reltol=1e-8, abstol=1e-8)
plot!(plt, sol.t , sol[1,:], label="", linecolor = :red, linestyle = :solid, lw=1.5)

prob = ODEProblem(f!, [9.0], tspan,[1.0])
sol = solve(prob, Tsit5(), reltol=1e-8, abstol=1e-8)
plot!(plt, sol.t , sol[1,:], label="", linecolor = :red, linestyle = :solid, lw=1.5)

ylabel!("x[1] (M)")
xlabel!("Time (seconds)")
display(plt)

status_code = get(integrator, TerminationStatus())
println("status code: $(status_code)")
d = integrator
t = 2.3

#@code_warntype DynamicBoundspODEsPILMS.single_step!(d.step_result, d.step_params, d.method_f!, d.set_tf!, d.Δ, d.A, d.P, d.rP, d.p, t)
#@btime DynamicBoundspODEsPILMS.single_step!($(d.step_result), $(d.step_params), $(d.method_f!),
#                    $(d.set_tf!), $(d.Δ), $(d.A), $(d.P), $(d.rP), $(d.p), $t)


#println("INTERNAL TESTS 1!")
#=
@code_warntype DynamicBoundspODEsPILMS.existence_uniqueness!(integrator.step_result,
                                                             integrator.set_tf!,
                                                             integrator.step_params.hmin,
                                                             integrator.P,
                                                             t)
=#
#=
println("INTERNAL TESTS 2!")
out = d.step_result
lf = integrator.method_f!
=#
#=
@code_warntype lf(out.hj, out.unique_result.X, out.Xⱼ, out.xⱼ, d.A, d.Δ,
                  d.P, d.rP, d.p, t)
                  =#
#=
hj = out.hj
urX = out.unique_result.X
Xⱼ = out.Xⱼ
xⱼ = out.xⱼ
A = d.A
del = d.Δ
P = d.P
rP = d.rP
p = d.p
=#
#@btime ($lf)($hj, $urX, $Xⱼ, $xⱼ, $A, $del, $P, $rP, $p, $t)

#println("INTERNAL TESTS 3!")
#jacfunc = lf.jac_tf!
#@code_warntype DynamicBoundspODEsPILMS.set_JxJp!(jacfunc, out.Xⱼ, d.P, t)
#@btime DynamicBoundspODEsPILMS.set_JxJp!($jacfunc, $(out.Xⱼ), $(d.P), $t)

#@code_warntype DynamicBoundspODEsPILMS.jacobian_taylor_coeffs!(jacfunc, out.Xⱼ, d.P, t)
#@btime DynamicBoundspODEsPILMS.jacobian_taylor_coeffs!($jacfunc, $(out.Xⱼ), $(d.P), $t)
#=
r = jacfunc.result
aot = jacfunc.out
yot = jacfunc.y
cfg = jacfunc.cfg

println(" ")
println(" ")
println("jacobian stuff")
=#
#@code_warntype ForwardDiff.jacobian!(r, jacfunc, aot, yot, compare_config)
#@btime ForwardDiff.jacobian!($r, $jacfunc, $aot, $yot, $compare_config)

#println(" ")
#println(" ")
#println("jacobian functor stuff")
#outy = [integrator.method_f!.jac_tf!.x[1] for i in 1:21]
#iny = [integrator.method_f!.jac_tf!.x[1], integrator.method_f!.jac_tf!.p[1]]
#@code_warntype jacfunc(outy, iny)
#@btime ($jacfunc)($outy, $iny)

#=
order = 20
val = Val(21-1)
xtaylor = [STaylor1(integrator.method_f!.jac_tf!.x[1], val)]
xaux = deepcopy(xtaylor)
dx = deepcopy(xtaylor)
eqdiffs = jacfunc.g!
P = integrator.method_f!.jac_tf!.p[1]
println("jetcoeffs stuff")
vnxt = fill(0, 1)
fnxt = fill(0.0, 1)
=#
#@code_warntype DynamicBoundspODEsPILMS.jetcoeffs!(eqdiffs, t, xtaylor, xaux, dx, order, P, vnxt, fnxt)
#@btime DynamicBoundspODEsPILMS.jetcoeffs!($eqdiffs, $t, $xtaylor, $xaux, $dx, $order, $P, $vnxt, $fnxt)

#println("recurse taylor")
#@code_warntype DynamicBoundspODEsPILMS.recurse_taylor!(dx, xtaylor, vnxt)
#@btime DynamicBoundspODEsPILMS.recurse_taylor!($dx, $xtaylor, $vnxt)

println(" copy recurse")
#sdx = dx[1]
#sxtaylor = xtaylor[1]
#cflt = 1.0
# @code_warntype DynamicBoundspODEsPILMS.copy_recurse(sdx, sxtaylor, 1, cflt)
# @btime DynamicBoundspODEsPILMS.copy_recurse($sdx, $sxtaylor, 1, $cflt)
#=
s = integrator.step_result
@code_warntype DynamicBoundspODEsPILMS.existence_uniqueness!(s.unique_result, integrator.set_tf!, s.Xⱼ, s.hj,
                                              integrator.step_params.hmin, s.f, s.∂f∂x,
                                              s.∂f∂p, integrator.P, s.h, t)
=#
#=
ratio = rand(6)
pstar = pL.*ratio .+ pU.*(1.0 .- ratio)
setall!(integrator, ParameterValue(), pstar)
integrate!(integrator)
plot!(plt, integrator.local_problem_storage.integrator_t,
integrator.local_problem_storage.pode_x[1,:], label="Trajectories", linecolor = :green,
markershape = :+, markercolor = :green, linestyle = :dash, markersize = 2, lw=0.75)
=#


# seed!(xdual, x, cfg.seeds)
