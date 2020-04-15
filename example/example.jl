using Revise

using DynamicBoundsBase, DynamicBoundspODEsPILMS, Plots
pyplot()

x0(p) = [2.9; 1.0]
function f!(dx,x,p,t)
    dx[1] = -x[1]
    dx[2] = x[2]
    nothing
end
tspan = (0.0,18.0e-5*12000)
pL = [0.2; 0.1]
pU = 10.0*pL

prob = DynamicBoundsBase.ODERelaxProb(f!, tspan, x0, pL, pU)
integrator = DiscretizeRelax(prob)
ratio = rand(2)
pstar = pL.*ratio .+ pU.*(1.0 .- ratio)
setall!(integrator, ParameterValue(), pstar)
DynamicBoundsBase.relax!(integrator)

t_vec = integrator.time
lo_vec = getfield.(integrator.storage[1,:], :lo)
hi_vec = getfield.(integrator.storage[1,:], :hi)
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
