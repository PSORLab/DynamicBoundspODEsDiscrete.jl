using Revise

using DynamicBoundsBase, DynamicBoundspODEsPILMS, Plots, DifferentialEquations
pyplot()
println(" ")
println(" ------------------------------------------------------------ ")
println(" ------------------------------------------------------------ ")
println(" ------------------------------------------------------------ ")

# TODO: Get multiple fixed step sizes working?
# TODO: Why is uniqueness test failing?
# TODO: Get McCormick operator version working.

x0(p) = [9.0]
function f!(dx,x,p,t)
    dx[1] = -x[1] + p[1]
#    dx[2] = x[2]
    nothing
end
tspan = (0.0,1.0)
#pL = [0.2; 0.1]
#pU = 10.0*pL
pL = [-1.0]
pU = [1.0]

prob = DynamicBoundsBase.ODERelaxProb(f!, tspan, x0, pL, pU)
integrator = DiscretizeRelax(prob, h = 0.01)
ratio = rand(1)
pstar = pL.*ratio .+ pU.*(1.0 .- ratio)
setall!(integrator, ParameterValue(), pstar)
DynamicBoundsBase.relax!(integrator)

t_vec = integrator.time
#lo_vec = getfield.(integrator.storage[1,:], :lo)
#hi_vec = getfield.(integrator.storage[1,:], :hi)
lo_vec = getfield.(integrator.storage[:], :lo)
hi_vec = getfield.(integrator.storage[:], :hi)
plt = plot(t_vec , lo_vec, label="Interval Bounds", linecolor = :darkblue, linestyle = :dash,
           lw=1.5, legend=:bottomleft)
plot!(plt, t_vec , hi_vec, label="", linecolor = :darkblue, linestyle = :dash, lw=1.5)
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
