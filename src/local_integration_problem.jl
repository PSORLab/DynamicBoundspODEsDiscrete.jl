# Copyright (c) 2020: Matthew Wilhelm & Matthew Stuber.
# This work is licensed under the Creative Commons Attribution-NonCommercial-
# ShareAlike 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/ or send a letter to Creative
# Commons, PO Box 1866, Mountain View, CA 94042, USA.
#############################################################################
# DynamicBoundspODEIneq.jl
# See https://github.com/PSORLab/DynamicBoundspODEIneq.jl
#############################################################################
# src/integrate.jl
# Code used to integrate but not relax the underlying parametric ODE system.
#############################################################################

struct RelaxDualTag end

mutable struct LocalProblemStorage{PRB, N}
    pode_problem
    pode_x::ElasticArray{Float64,2}
    pode_dxdp::Vector{ElasticArray{Float64,2}}
    x0local::Vector{Float64}
    pduals::Vector{Dual{RelaxDualTag,Float64,N}}
    x0duals::Vector{Dual{RelaxDualTag,Float64,N}}
    integator
    user_t::Vector{Float64}
    integrator_t::Vector{Float64}
    abstol::Float64
    reltol::Float64
end
problem_type(x::LocalProblemStorage{N}) where {N} = typeof(pode_problem)

function seed_duals(x::AbstractArray{V}, ::Chunk{N} = Chunk(x)) where {V,N}
  seeds = construct_seeds(Partials{N,V})
  duals = [Dual{RelaxDualTag}(x[i],seeds[i]) for i in eachindex(x)]
end

function LocalProblemStorage(d::ODERelaxProb, integator, user_t::Vector{Float64})
    np = length(d.p)
    pode_problem = ODEForwardSensitivityProblem(d.f, zeros(Float64, d.nx), d.tspan, d.p)
    pode_x = zeros(Float64, d.nx, length(d.tsupports))
    pode_dxdp = Array{Float64,2}[zeros(Float64, d.nx, length(d.tsupports)) for i=1:np]
    pduals = seed_duals(d.p)
    sing_seed = single_seed(Partials{np, Float64}, Val(1))
    x0duals = fill(Dual{RelaxDualTag}(0.0, sing_seed), (d.nx,))
    x0local = zeros((np+1)*d.nx)
    local_problem_storage = LocalProblemStorage{typeof(pode_problem), np}(pode_problem, pode_x, pode_dxdp,
                                                    x0local, pduals, x0duals, integator, user_t, Float64[],
                                                    1E-9, 1E-8)
    return local_problem_storage
end
