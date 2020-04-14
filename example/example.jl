using Revise

using DynamicBoundsBase, DynamicBoundspODEsPILMS

x0(p) = [34.0; 20.0]
function f!(du, u, p, t)
    du[1] = -p[1]*u[1]*u[2] + p[2]*u[1]
    du[2] = -p[1]*u[1]*u[2] + p[2]*u[2]
    return
end
tspan = (0.0,18.0e-5*50)
pL = [0.1; 0.033]
pU = 10.0*pL

prob = DynamicBoundsBase.ODERelaxProb(f!, tspan, x0, pL, pU)
integrator = DiscretizeRelax(prob)
integrator.p .= 0.5*(pL + pU)
DynamicBoundspODEsPILMS.relax!(integrator)

#=
function single_step!(out::StepResult{S}, params::StepParams, lf::LohnersFunctor,
                      stf!::TaylorFunctor!, A::CircularBuffer{QRDenseStorage},
                      Yⱼ::Vector{S}, Δ::CircularBuffer{Vector{S}}) where {S <: Real}
=#
