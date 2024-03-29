# Copyright (c) 2020: Matthew Wilhelm & Matthew Stuber.
# This work is licensed under the Creative Commons Attribution-NonCommercial-
# ShareAlike 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/ or send a letter to Creative
# Commons, PO Box 1866, Mountain View, CA 94042, USA.
#############################################################################
# Dynamic Bounds - pODEs Discrete
# A package for discretize and relax methods for bounding pODEs.
# See https://github.com/PSORLab/DynamicBoundspODEsDiscrete.jl
#############################################################################
# src/DiscretizeRelax/utilities/access_functions.jl
# Defines methods to access attributes via DynamicBoundsBase interface.
#############################################################################

DBB.supports(::DiscretizeRelax, ::DBB.IntegratorName) = true
DBB.supports(::DiscretizeRelax, ::DBB.Subgradient{T}) where {T <: AbstractBoundLoc} = true
DBB.supports(::DiscretizeRelax, ::DBB.Bound{T}) where {T <: AbstractBoundLoc} = true
DBB.supports(::DiscretizeRelax, ::DBB.Relaxation{T}) where {T <: AbstractBoundLoc} = true
DBB.supports(::DiscretizeRelax, ::DBB.IsNumeric) = true
DBB.supports(::DiscretizeRelax, ::DBB.IsSolutionSet) = true
DBB.supports(::DiscretizeRelax, ::DBB.TerminationStatus) = true
DBB.supports(::DiscretizeRelax, ::DBB.Value) = true
DBB.supports(::DiscretizeRelax, ::DBB.ParameterBound{Lower}) = true
DBB.supports(::DiscretizeRelax, ::DBB.ParameterBound{Upper}) = true
DBB.supports(::DiscretizeRelax, ::DBB.ParameterValue) = true
DBB.supports(::DiscretizeRelax, ::DBB.SupportSet) = true
DBB.supports(::DiscretizeRelax, ::DBB.ParameterNumber) = true
DBB.supports(::DiscretizeRelax, ::DBB.StateNumber) = true
DBB.supports(::DiscretizeRelax, ::DBB.SupportNumber) = true
DBB.supports(t::DiscretizeRelax, ::DBB.LocalSensitivityOn) = t.calculate_local_sensitivity

function get_val_loc(t::DiscretizeRelax, index::Int64, time::Float64)
    (index <= 0 && time == -Inf) && error("Must set either index or time.")
    if index > 0
        return t.relax_t_dict_indx[index]
    end
    t.relax_t_dict_flt[time]
end

function get_val_loc_local(t::DiscretizeRelax, index::Int64, time::Float64)
    (index <= 0 && time == -Inf) && error("Must set either index or time.")
    if index > 0
        return t.local_t_dict_indx[index]
    end
    t.local_t_dict_flt[time]
end

DBB.get(t::DiscretizeRelax, v::DBB.IntegratorName) = "Discretize & Relax Integrator" # TO DO... FIX ME
DBB.get(t::DiscretizeRelax, v::DBB.IsNumeric) = false # TO DO... FIX ME
DBB.get(t::DiscretizeRelax, v::DBB.IsSolutionSet) = true
DBB.get(t::DiscretizeRelax, v::DBB.TerminationStatus) = t.error_code
DBB.get(t::DiscretizeRelax, v::DBB.SupportSet) = DBB.SupportSet(t.tsupports)
DBB.get(t::DiscretizeRelax, v::DBB.ParameterNumber) = t.np
DBB.get(t::DiscretizeRelax, v::DBB.StateNumber) = t.nx
DBB.get(t::DiscretizeRelax, v::DBB.SupportNumber) = length(t.tsupports)
DBB.get(t::DiscretizeRelax, v::DBB.LocalSensitivityOn) = t.calculate_local_sensitivity

DBB.getall(t::DiscretizeRelax, v::DBB.ParameterBound{Lower}) = t.pL
DBB.getall(t::DiscretizeRelax, v::DBB.ParameterBound{Upper}) = t.pU

function DBB.set!(t::DiscretizeRelax, v::DBB.LocalSensitivityOn, q::Bool)
    t.calculate_local_sensitivity = q
    return
end

function DBB.set!(t::DiscretizeRelax, v::DBB.ConstantStateBounds)
    t.constant_state_bounds = v
    set_constant_state_bounds!(t.method_f!, v)
    t.exist_result.constant_state_bounds = v
    return
end

function DBB.set!(t::DiscretizeRelax, v::DBB.SupportSet)
    t.time = v.s
    t.tsupports = v.s
    nothing
end

## Setting problem attributes
function DBB.setall!(t::DiscretizeRelax, v::ParameterBound{Lower}, value::Vector{Float64})
    t.new_decision_box = true
    @inbounds for i in 1:t.np
        t.pL[i] = value[i]
    end
    return
end

function DBB.setall!(t::DiscretizeRelax, v::ParameterBound{Upper}, value::Vector{Float64})
    t.new_decision_box = true
    @inbounds for i in 1:t.np
        t.pU[i] = value[i]
    end
    return
end

function DBB.setall!(t::DiscretizeRelax, v::ParameterValue, value::Vector{Float64})
    t.new_decision_pnt = true
    @inbounds for i in 1:t.np
        t.p[i] = value[i]
    end
    return
end

function DBB.getall!(out::Vector{Float64}, t::DiscretizeRelax, v::DBB.ParameterValue)
    out .= t.p[1:t.np]
    return
end

function DBB.getall!(out::Array{Float64,2}, t::DiscretizeRelax, v::DBB.Value)
    copyto!(out, t.local_problem_storage.pode_x)
    return
end

function DBB.get(out::Vector{Float64}, t::DiscretizeRelax, v::DBB.Value)
    val_loc = get_val_loc_local(t, v.index, v.time)
    out .= t.local_problem_storage.pode_x[:, val_loc]
    return
end

## Inplace integrator acccess functions
function DBB.getall!(out::Vector{Matrix{Float64}}, t::DiscretizeRelax{X,T}, ::DBB.Subgradient{Lower}) where {X, T <: AbstractInterval}
    for i in 1:t.np
        fill!(out[i], 0.0)
    end
    return
end
function DBB.getall!(out::Vector{Matrix{Float64}}, t::DiscretizeRelax{X,T}, ::DBB.Subgradient{Lower}) where {X, T <: MC}
    for i = 1:length(t.storage)
        for j = 1:t.np
            for k = 1:t.nx
                out[j][k,i]= t.storage[i][k].cv_grad[j]
            end
        end
    end
    return
end

function DBB.getall!(out::Vector{Matrix{Float64}}, t::DiscretizeRelax{X,T}, ::DBB.Subgradient{Upper}) where {X, T <: AbstractInterval}
    for i in 1:t.np
        fill!(out[i], 0.0)
    end
    return
end

function DBB.getall!(out::Vector{Matrix{Float64}}, t::DiscretizeRelax{X,T}, ::DBB.Subgradient{Upper}) where {X, T <: MC}
    for i = 1:length(t.storage)
        for j = 1:t.np
            for k = 1:t.nx
                out[j][k,i]= t.storage[i][k].cc_grad[j]
            end
        end
    end
    return
end

function DBB.get(out::Matrix{Float64}, t::DiscretizeRelax{X,T}, v::DBB.Subgradient{Lower}) where {X, T <: AbstractInterval}
    fill!(out, 0.0)
    return
end

function DBB.get(out::Matrix{Float64}, t::DiscretizeRelax{X,T}, v::DBB.Subgradient{Upper}) where {X, T <: AbstractInterval}
    fill!(out, 0.0)
    return
end

function DBB.get(out::Matrix{Float64}, t::DiscretizeRelax{X,T}, v::DBB.Subgradient{Lower}) where {X, T <: MC}
    val_loc = get_val_loc(t, v.index, v.time)
    for i = 1:t.np
        for j = 1:t.nx
            out[j,i] = t.relax_cv_grad[j,val_loc][i]
        end
    end
    return
end

function DBB.get(out::Matrix{Float64}, t::DiscretizeRelax{X,T}, v::DBB.Subgradient{Upper}) where {X, T <: MC}
    val_loc = get_val_loc(t, v.index, v.time)
    for i = 1:t.np
        for j = 1:t.nx
            out[j,i] = t.relax_cc_grad[j,val_loc][i]
        end
    end
    return
end

function DBB.getall!(out::Vector{Float64}, t::DiscretizeRelax{X,T}, ::DBB.Bound{Lower}) where {X, T <: AbstractInterval}
    @inbounds for j in eachindex(out)
        out[j] = t.storage[j][1].lo
    end
    return
end

function DBB.getall!(out::Matrix{Float64}, t::DiscretizeRelax{X,T}, ::DBB.Bound{Lower}) where {X, T <: AbstractInterval}
    dims = length(t.storage), length(t.storage[1])
    (length(dims) == 1) && (dims = (dims[1],1))
    for i = 1:dims[1]
        for j = 1:dims[2]
            out[j,i] = t.storage[i][j].lo
        end
    end
    return
end

function DBB.getall!(out::Vector{Float64}, t::DiscretizeRelax{X,T}, ::DBB.Bound{Upper}) where {X, T <: AbstractInterval}
    @inbounds for j in eachindex(out)
        out[j] = t.storage[j][1].hi
    end
    return
end
function DBB.getall!(out::Matrix{Float64}, t::DiscretizeRelax{X,T}, v::DBB.Bound{Upper}) where {X, T <: AbstractInterval}
    dims = length(t.storage), length(t.storage[1])
    (length(dims) == 1) && (dims = (dims[1],1))
    for i = 1:dims[1]
        for j = 1:dims[2]
            out[j, i] = t.storage[i][j].hi
        end
    end
    return
end

function DBB.getall!(out::Vector{Float64}, t::DiscretizeRelax{X,T}, ::DBB.Bound{Lower}) where {X, T <: MC}
    @inbounds for j in eachindex(out)
        out[j] = t.storage[j][1].Intv.lo
    end
    return
end

function DBB.getall!(out::Matrix{Float64}, t::DiscretizeRelax{X,T}, ::DBB.Bound{Lower}) where {X, T <: MC}
    dims = length(t.storage), length(t.storage[1])
    (length(dims) == 1) && (dims = (dims[1],1))
    for i = 1:dims[1]
        for j = 1:dims[2]
            out[j, i] = t.storage[i][j].Intv.lo
        end
    end
    return
end

function DBB.getall!(out::Vector{Float64}, t::DiscretizeRelax{X,T}, ::DBB.Bound{Upper}) where {X, T <: MC}
    @inbounds for j in eachindex(out)
        out[j] = t.storage[j][1].Intv.hi
    end
    return
end
function DBB.getall!(out::Matrix{Float64}, t::DiscretizeRelax{X,T}, v::DBB.Bound{Upper}) where {X, T <: MC}
    dims = length(t.storage), length(t.storage[1])
    (length(dims) == 1) && (dims = (dims[1],1))
    for i = 1:dims[1]
        for j = 1:dims[2]
            out[j, i] = t.storage[i][j].Intv.hi
        end
    end
    return
end

function DBB.getall!(out::Matrix{Float64}, t::DiscretizeRelax{X,T}, ::DBB.Relaxation{Lower}) where {X, T <: AbstractInterval}
    dims = length(t.storage), length(t.storage[1])
    (length(dims) == 1) && (dims = (dims[1],1))
    for i = 1:dims[1]
        for j = 1:dims[2]
            out[j, i] = t.storage[i][j].lo
        end
    end
    return
end

function DBB.getall!(out::Matrix{Float64}, t::DiscretizeRelax{X,T}, ::DBB.Relaxation{Lower}) where {X, T <: MC}
    dims = length(t.storage), length(t.storage[1])
    (length(dims) == 1) && (dims = (dims[1],1))
    for i = 1:dims[1]
        for j = 1:dims[2]
            out[j, i] = t.storage[i][j].cv
        end
    end
    return
end

function DBB.getall!(out::Vector{Float64}, t::DiscretizeRelax{X,T}, ::DBB.Relaxation{Lower}) where {X, T <: AbstractInterval}
    @inbounds for j in eachindex(out)
        out[j] = t.storage[j][1].lo
    end
    return
end

function DBB.getall!(out::Vector{Float64}, t::DiscretizeRelax{X,T}, ::DBB.Relaxation{Lower}) where {X, T <: MC}
    @inbounds for j in eachindex(out)
        out[j] = t.storage[j][1].cv
    end
    return
end

function DBB.getall!(out::Matrix{Float64}, t::DiscretizeRelax{X,T}, ::DBB.Relaxation{Upper}) where {X, T <: AbstractInterval}
    dims = length(t.storage), length(t.storage[1])
    (length(dims) == 1) && (dims = (dims[1],1))
    for i = 1:dims[1]
        for j = 1:dims[2]
            out[j, i] = t.storage[i][j].hi
        end
    end
    return
end

function DBB.getall!(out::Matrix{Float64}, t::DiscretizeRelax{X,T}, ::DBB.Relaxation{Upper}) where {X, T <: MC}
    dims = length(t.storage), length(t.storage[1])
    (length(dims) == 1) && (dims = (dims[1],1))
    for i = 1:dims[1]
        for j = 1:dims[2]
            out[j, i] = t.storage[i][j].cc
        end
    end
    return
end

function DBB.getall!(out::Vector{Float64}, t::DiscretizeRelax{X,T}, ::DBB.Relaxation{Upper}) where {X, T <: AbstractInterval}
    @inbounds for j in eachindex(out)
        out[j] = t.storage[j][1].hi
    end
    return
end

function DBB.getall!(out::Vector{Float64}, t::DiscretizeRelax{X,T}, ::DBB.Relaxation{Upper}) where {X, T <: MC}
    @inbounds for j in eachindex(out)
        out[j] = t.storage[j][1].cc
    end
    return
end

## Out of place integrator acccess functions
function DBB.getall(t::DiscretizeRelax{X,T}, ::DBB.Subgradient{Lower}) where {X, T <: AbstractInterval}
    dims = length(t.storage), length(t.storage[1])
    (length(dims) == 1) && (dims = (dims[1],1))
    out = Matrix{Float64}[]
    for i = 1:t.np
        push!(out, zeros(Float64, dims[2], dims[1]))
    end
    out
end
function DBB.getall(t::DiscretizeRelax{X,T}, ::DBB.Subgradient{Lower}) where {X, T <: MC}
    dims = length(t.storage), length(t.storage[1])
    (length(dims) == 1) && (dims = (dims[1],1))
    out = Matrix{Float64}[]
    for i = 1:t.np
        push!(out, zeros(Float64, dims[2], dims[1]))
        for j = 1:dims[1]
            for k = 1:dims[2]
                out[i][k, j] = t.storage[j][k].cv_grad[i]
            end
        end
    end
    out
end

function DBB.getall(t::DiscretizeRelax{X,T}, ::DBB.Subgradient{Upper}) where {X, T <: AbstractInterval}
    dims = length(t.storage), length(t.storage[1])
    (length(dims) == 1) && (dims = (dims[1],1))
    out = Matrix{Float64}[]
    for i = 1:t.np
        push!(out, zeros(Float64, dims[2], dims[1]))
    end
    out
end

function DBB.getall(t::DiscretizeRelax{X,T}, ::DBB.Subgradient{Upper}) where {X, T <: MC}
    dims = length(t.storage), length(t.storage[1])
    (length(dims) == 1) && (dims = (dims[1],1))
    out = Matrix{Float64}[]
    for i = 1:t.np
        push!(out, zeros(Float64, dims[2], dims[1]))
        for j = 1:dims[1]
            for k = 1:dims[2]
                out[i][k, j] = t.storage[j][k].cc_grad[i]
            end
        end
    end
    out
end

function DBB.get!(out::Vector{Float64}, t::DiscretizeRelax{X,T}, v::DBB.Bound{Lower}) where {X, T <: AbstractInterval}
    val_loc = get_val_loc(t, v.index, v.time)
    @__dot__ out = lo(t.storage[val_loc])
    return
end

function DBB.get!(out::Vector{Float64}, t::DiscretizeRelax{X,T}, v::DBB.Bound{Upper}) where {X, T <: AbstractInterval}
    val_loc = get_val_loc(t, v.index, v.time)
    @__dot__ out = hi(t.storage[val_loc])
    return
end

function DBB.getall(t::DiscretizeRelax{X,T}, ::DBB.Bound{Lower}) where {X, T <: AbstractInterval}
    dims = length(t.storage), length(t.storage[1])
    (length(dims) == 1) && (dims = (dims[1],1))
    out = zeros(Float64, dims[2], dims[1])
    for i = 1:dims[1]
        for j = 1:dims[2]
            out[j, i] = t.storage[i][j].lo
        end
    end
    out
end

function DBB.getall(t::DiscretizeRelax{X,T}, ::DBB.Bound{Upper}) where {X, T <: AbstractInterval}
    dims = length(t.storage), length(t.storage[1])
    (length(dims) == 1) && (dims = (dims[1],1))
    out = zeros(Float64, dims[2], dims[1])
    for i = 1:dims[1]
        for j = 1:dims[2]
            out[j, i] = t.storage[i][j].hi
        end
    end
    out
end

function DBB.get(out::Vector{Float64}, t::DiscretizeRelax{X,T}, v::DBB.Bound{Lower}) where {X, T <: MC}
    val_loc = get_val_loc_local(t, v.index, v.time)
    @__dot__ out = lo(t.storage[val_loc])
    return
end

function DBB.get(out::Vector{Float64}, t::DiscretizeRelax{X,T}, v::DBB.Bound{Upper}) where {X, T <: MC}
    val_loc = get_val_loc_local(t, v.index, v.time)
    @__dot__ out = hi(t.storage[val_loc])
    return
end

function DBB.getall(t::DiscretizeRelax{X,T}, ::DBB.Bound{Lower}) where {X, T <: MC}
    dims = length(t.storage), length(t.storage[1])
    (length(dims) == 1) && (dims = (dims[1],1))
    out = zeros(Float64, dims[2], dims[1])
    for i = 1:dims[1]
        for j = 1:dims[2]
            out[j, i] = t.storage[i][j].Intv.lo
        end
    end
    out
end

function DBB.getall(t::DiscretizeRelax{X,T}, ::DBB.Bound{Upper}) where {X, T <: MC}
    dims = length(t.storage), length(t.storage[1])
    (length(dims) == 1) && (dims = (dims[1],1))
    out = zeros(Float64, dims[2], dims[1])
    for i = 1:dims[1]
        for j = 1:dims[2]
            out[j, i] = t.storage[i][j].Intv.hi
        end
    end
    out
end

function DBB.get(out::Vector{Float64}, t::DiscretizeRelax{X,T}, v::DBB.Relaxation{Lower}) where {X, T <: MC}
    val_loc = get_val_loc(t, v.index, v.time)
    @__dot__ out = cv(t.storage[val_loc])
    return
end

function DBB.get(out::Vector{Float64}, t::DiscretizeRelax{X,T}, v::DBB.Relaxation{Upper}) where {X, T <: MC}
    val_loc = get_val_loc(t, v.index, v.time)
    @__dot__ out = cc(t.storage[val_loc])
    return
end

function DBB.get(out::Vector{Float64}, t::DiscretizeRelax{X,T}, v::DBB.Relaxation{Lower}) where {X, T <: AbstractInterval}
    val_loc = get_val_loc(t, v.index, v.time)
    @__dot__ out = lo(t.storage[val_loc])
    return
end

function DBB.get(out::Vector{Float64}, t::DiscretizeRelax{X,T}, v::DBB.Relaxation{Upper}) where {X, T <: AbstractInterval}
    val_loc = get_val_loc(t, v.index, v.time)
    @__dot__ out = hi(t.storage[val_loc])
    return
end

function DBB.getall(t::DiscretizeRelax{X,T}, ::DBB.Relaxation{Lower}) where {X, T <: AbstractInterval}
    dims = length(t.storage), length(t.storage[1])
    (length(dims) == 1) && (dims = (dims[1],1))
    out = zeros(Float64, dims[2], dims[1])
    for i = 1:dims[1]
        for j = 1:dims[2]
            out[j, i] = t.storage[i][j].lo
        end
    end
    out
end

function DBB.getall(t::DiscretizeRelax{X,T}, ::DBB.Relaxation{Lower}) where {X, T <: MC}
    dims = length(t.storage), length(t.storage[1])
    (length(dims) == 1) && (dims = (dims[1],1))
    out = zeros(Float64, dims[2], dims[1])
    for i = 1:dims[1]
        for j = 1:dims[2]
            out[j, i] = t.storage[i][j].cv
        end
    end
    out
end

function DBB.getall(t::DiscretizeRelax{X,T}, ::DBB.Relaxation{Upper}) where {X, T <: AbstractInterval}
    dims = length(t.storage), length(t.storage[1])
    (length(dims) == 1) && (dims = (dims[1],1))
    out = zeros(Float64, dims[2], dims[1])
    for i = 1:dims[1]
        for j = 1:dims[2]
            out[j, i] = t.storage[i][j].hi
        end
    end
    out
end
function DBB.getall(t::DiscretizeRelax{X,T}, ::DBB.Relaxation{Upper}) where {X, T <: MC}
    dims = length(t.storage), length(t.storage[1])
    (length(dims) == 1) && (dims = (dims[1],1))
    out = zeros(Float64,  dims[2], dims[1])
    for i = 1:dims[1]
        for j = 1:dims[2]
            out[j, i] = t.storage[i][j].cc
        end
    end
    out
end

function DBB.get(t::DiscretizeRelax{X,T}, v::DBB.LocalIntegrator) where {X, T}
    return t.local_problem_storage
end

DBB.get(t::DiscretizeRelax{X,T}, v::DBB.AttachedProblem) where {X, T} = t.prob
