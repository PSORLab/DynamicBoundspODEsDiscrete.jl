#  START of function ... (HOE,yadda yadda)
# p[1] = Interval{Float64}
# x[1] = STaylor1{6,Interval{Float64}}

#

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

#=
function Base.literal_pow(::typeof(^), x::Interval{T}, ::Val{N}) where {N,T<:AbstractFloat}
    IntervalArithmetic.pow(x, N)
end
=#
using DynamicBoundsBase, DynamicBoundspODEsPILMS, Plots, DifferentialEquations#, Cthulhu

#pyplot()
println(" ")
println(" ------------------------------------------------------------ ")
println(" ------------------------------------------------------------ ")
println(" ------------------------------------------------------------ ")

# TODO: Fix Lohner interval test?
# TODO: Get McCormick operator version working.
# TODO: Fix variable step size hoe routine...

x0(p) = [9.0]
function f!(dx, x, p, t)
    #dx[1] = -x[1]^2 + p[1]
    #dx[1] = p[1] - x[1]^2
    dx[1] = p[1] - x[1]^2
    nothing
end

tspan = (0.0,1.00)
#pL = [0.2; 0.1]
#pU = 10.0*pL
pL = [-1.0]
pU = [1.0]
kval = 5


#=
DynamicBoundsBase.relax!(integrator)
t_vec = integrator.time
lo_vec = getfield.(getindex.(integrator.storage[:],1), :lo)
hi_vec = getfield.(getindex.(integrator.storage[:],1), :hi)
plt = plot(t_vec , lo_vec, label="Interval Bounds -1.0", linecolor = :blue, linestyle = :dashdot,
           lw=1.5, legend=:bottomleft)
plot!(plt, t_vec , hi_vec, label="", linecolor = :blue, linestyle = :dashdot, lw=1.5)
=#
prob = DynamicBoundsBase.ODERelaxProb(f!, tspan, x0, pL, pU)
#integrator = DiscretizeRelax(prob, DynamicBoundspODEsPILMS.LohnerContractor{4}(), h = 0.01, skip_step2 = false, relax = true)
integrator = DiscretizeRelax(prob, HermiteObreschkoff(2,2), h = 0.01, skip_step2 = false, relax = false)
ratio = rand(1)
pstar = pL.*ratio .+ pU.*(1.0 .- ratio)
setall!(integrator, ParameterValue(), [0.1])
DynamicBoundsBase.relax!(integrator)

using BenchmarkTools
@btime DynamicBoundsBase.relax!($integrator)

t_vec = integrator.time
lo_vec = getfield.(getindex.(integrator.storage[:],1), :lo)
hi_vec = getfield.(getindex.(integrator.storage[:],1), :hi)
#lo_vec = getfield.(getfield.(getindex.(integrator.storage[:],1), :Intv), :lo)
#hi_vec = getfield.(getfield.(getindex.(integrator.storage[:],1), :Intv), :hi)

#lo_vec = getfield.(getindex.(integrator.storage[:],1), :cv)
#hi_vec = getfield.(getindex.(integrator.storage[:],1), :cc)

plt = plot(t_vec , lo_vec, label="Interval Bounds 0.0", linecolor = :black, linestyle = :dot,
           lw=1.5, legend=:bottomleft)
plot!(plt, t_vec , hi_vec, label="", linecolor = :black, linestyle = :dot, lw=1.5)
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
