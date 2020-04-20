using Revise

using DynamicBoundsBase, DynamicBoundspODEsPILMS, Plots, DifferentialEquations
pyplot()
println(" ")
println(" ------------------------------------------------------------ ")
println(" ------------------------------------------------------------ ")
println(" ------------------------------------------------------------ ")

# TODO: Fix Lohner interval test?
# TODO: Get McCormick operator version working.
# TODO: Fix variable step size hoe routine...

x0(p) = [9.0]
function f!(dx,x,p,t)
    #dx[1] = -x[1]^2 + p[1]
    dx[1] = x[1] #-x[1]*x[1] + p[1]
    nothing
end
tspan = (0.0,1.0)
#pL = [0.2; 0.1]
#pU = 10.0*pL
pL = [-1.0]
pU = [1.0]

prob = DynamicBoundsBase.ODERelaxProb(f!, tspan, x0, pL, pU)
integrator = DiscretizeRelax(prob, h = 0.01, skip_step2 = false, k = 20)
setall!(integrator, ParameterValue(), [-1.0])

using BenchmarkTools
@btime DynamicBoundsBase.relax!($integrator)
@code_warntype DynamicBoundsBase.relax!(integrator)

DynamicBoundsBase.relax!(integrator)
t_vec = integrator.time
lo_vec = getfield.(getindex.(integrator.storage[:],1), :lo)
hi_vec = getfield.(getindex.(integrator.storage[:],1), :hi)
plt = plot(t_vec , lo_vec, label="Interval Bounds -1.0", linecolor = :blue, linestyle = :dashdot,
           lw=1.5, legend=:bottomleft)
plot!(plt, t_vec , hi_vec, label="", linecolor = :blue, linestyle = :dashdot, lw=1.5)

prob = DynamicBoundsBase.ODERelaxProb(f!, tspan, x0, pL, pU)
integrator = DiscretizeRelax(prob, h = 0.01, skip_step2 = false, k = 20)
ratio = rand(1)
pstar = pL.*ratio .+ pU.*(1.0 .- ratio)
setall!(integrator, ParameterValue(), [0.0])
DynamicBoundsBase.relax!(integrator)

t_vec = integrator.time
lo_vec = getfield.(getindex.(integrator.storage[:],1), :lo)
hi_vec = getfield.(getindex.(integrator.storage[:],1), :hi)
plot!(plt, t_vec , lo_vec, label="Interval Bounds 0.0", linecolor = :black, linestyle = :dot,
           lw=1.5, legend=:bottomleft)
plot!(plt, t_vec , hi_vec, label="", linecolor = :black, linestyle = :dot, lw=1.5)

prob = DynamicBoundsBase.ODERelaxProb(f!, tspan, x0, pL, pU)
integrator = DiscretizeRelax(prob, h = 0.01, skip_step2 = false, k = 20)
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

#=
ratio = rand(6)
pstar = pL.*ratio .+ pU.*(1.0 .- ratio)
setall!(integrator, ParameterValue(), pstar)
integrate!(integrator)
plot!(plt, integrator.local_problem_storage.integrator_t,
integrator.local_problem_storage.pode_x[1,:], label="Trajectories", linecolor = :green,
markershape = :+, markercolor = :green, linestyle = :dash, markersize = 2, lw=0.75)
=#
