using Revise

using DynamicBoundsBase, DynamicBoundspODEsPILMS

x0(p) = [0.1; 1.0]
function f!(dx,x,p,t)
    dx[1] = x[1]^2 + p[2]
    dx[2] = x[2] + p[1]^2
    nothing
end
tspan = (0.0,18.0e-5*12000)
pL = [0.2; 0.1]
pU = 10.0*pL

prob = DynamicBoundsBase.ODERelaxProb(f!, tspan, x0, pL, pU)
integrator = DiscretizeRelax(prob)
integrator.p .= 0.5*(pL + pU)
#pstar .= 0.5*(pL + pU)
#setall!(integrator, ParameterValue(), pstar)

DynamicBoundspODEsPILMS.relax!(integrator)
