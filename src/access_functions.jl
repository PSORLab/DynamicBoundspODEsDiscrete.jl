
## Setting problem attributes
function setall!(t::DiscretizeRelax, v::ParameterBound{Lower}, value::Vector{Float64})
    t.new_decision_box = true
    @inbounds for i in 1:t.np
        t.pL[i] = value[i]
    end
    return
end

function setall!(t::DiscretizeRelax, v::ParameterBound{Upper}, value::Vector{Float64})
    t.new_decision_box = true
    @inbounds for i in 1:t.np
        t.pU[i] = value[i]
    end
    return
end

function setall!(t::DiscretizeRelax, v::ParameterValue, value::Vector{Float64})
    t.new_decision_pnt = true
    @inbounds for i in 1:t.np
        t.p[i] = value[i]
    end
    return
end


## Inplace integrator acccess functions
function getall!(out::Vector{Array{Float64,2}}, t::DiscretizeRelax{X,T}, g::Subgradient{Lower}) where {X, T <: AbstractInterval}
    for i in 1:t.np
        fill!(out[i], 0.0)
    end
    return
end
function getall!(out::Vector{Array{Float64,2}}, t::DiscretizeRelax{X,T}, g::Subgradient{Lower}) where {X, T <: MC}
    for i in 1:t.np
        @inbounds for j in eachindex(out[i])
            out[i][j] = t.storage[j].cv_grad[j]
        end
    end
    return
end

function getall!(out::Vector{Array{Float64,2}}, t::DiscretizeRelax{X,T}, g::Subgradient{Upper}) where {X, T <: AbstractInterval}
    for i in 1:t.np
        fill!(out[i], 0.0)
    end
    return
end

function getall!(out::Vector{Array{Float64,2}}, t::DiscretizeRelax{X,T} g::Subgradient{Upper}) where {X, T <: MC}
    for i in 1:t.np
        @inbounds for j in eachindex(out[i])
            out[i][j] = t.storage[j].cc_grad[i]
        end
    end
    return
end

function getall!(out::Union{Vector{Float64}, Matrix{Float64}}, t::DiscretizeRelax, v::Bound{Lower})
    @inbounds for j in eachindex(out)
        out[j] = t.storage[j].Intv.lo
    end
    return
end

function getall!(out::Union{Vector{Float64}, Matrix{Float64}}, t::DiscretizeRelax, v::Bound{Upper})
    @inbounds for j in eachindex(out)
        out[j] = t.storage[j].Intv.hi
    end
    return
end

function getall!(out::Union{Vector{Float64}, Array{Float64,2}}, t::DiscretizeRelax{X,T}, v::Relaxation{Lower}) where {X, T <: AbstractInterval}
    @inbounds for i in eachindex(out)
        out[i] = t.storage[i].Intv.lo
    end
    return
end

function getall!(out::Union{Vector{Float64}, Array{Float64,2}}, t::DiscretizeRelax{X,T}, v::Relaxation{Lower}) where {X, T <: MC}
    @inbounds for i in eachindex(out)
        out[i] = t.storage[i].cv
    end
    return
end


function getall!(out::Union{Vector{Float64}, Array{Float64,2}}, t::DiscretizeRelax{X,T}, v::Relaxation{Upper}) where {X, T <: AbstractInterval}
    @inbounds for i in eachindex(out)
        out[i] = t.storage[i].Intv.hi
    end
    return
end
function getall!(out::Union{Vector{Float64}, Array{Float64,2}}, t::DiscretizeRelax{X,T}, v::Relaxation{Upper}) where {X, T <: MC}
    @inbounds for i in eachindex(out)
        out[i] = t.storage[i].cc
    end
    return
end

## Out of place integrator acccess functions
function getall(t::DiscretizeRelax{X,T}, g::Subgradient{Lower}) where {X, T <: AbstractInterval}
    dim1, dim2 = size(t.storage)
    out = zeros(t.np, zeros(Float64, dim1, dim2))
    for i in 1:t.np
        fill!(out[i], 0.0)
    end
    out
end
function getall(t::DiscretizeRelax{X,T}, g::Subgradient{Lower}) where {X, T <: MC}
    dim1, dim2 = size(t.storage)
    out = zeros(t.np, zeros(Float64, dim1, dim2))
    for i in 1:t.np
        @inbounds for j in eachindex(out[i])
            out[i][j] = t.storage[j].cv_grad[j]
        end
    end
    out
end

function getall(t::DiscretizeRelax{X,T}, g::Subgradient{Upper}) where {X, T <: AbstractInterval}
    dim1, dim2 = size(t.storage)
    out = zeros(t.np, zeros(Float64, dim1, dim2))
    for i in 1:t.np
        fill!(out[i], 0.0)
    end
    out
end

function getall(t::DiscretizeRelax{X,T} g::Subgradient{Upper}) where {X, T <: MC}
    dim1, dim2 = size(t.storage)
    out = zeros(t.np, zeros(Float64, dim1, dim2))
    for i in 1:t.np
        @inbounds for j in eachindex(out[i])
            out[i][j] = t.storage[j].cc_grad[i]
        end
    end
    out
end

function getall(t::DiscretizeRelax, v::Bound{Lower})
    dim1, dim2 = size(t.storage)
    out = zeros(Float64, dim1, dim2)
    @inbounds for j in eachindex(out)
        out[j] = t.storage[j].Intv.lo
    end
    out
end

function getall(t::DiscretizeRelax, v::Bound{Upper})
    dim1, dim2 = size(t.storage)
    out = zeros(Float64, dim1, dim2)
    @inbounds for j in eachindex(out)
        out[j] = t.storage[j].Intv.hi
    end
    out
end

function getall(t::DiscretizeRelax{X,T}, v::Relaxation{Lower}) where {X, T <: AbstractInterval}
    dim1, dim2 = size(t.storage)
    out = zeros(Float64, dim1, dim2)
    @inbounds for i in eachindex(out)
        out[i] = t.storage[i].Intv.lo
    end
    out
end

function getall(t::DiscretizeRelax{X,T}, v::Relaxation{Lower}) where {X, T <: MC}
    dim1, dim2 = size(t.storage)
    out = zeros(Float64, dim1, dim2)
    @inbounds for i in eachindex(out)
        out[i] = t.storage[i].cv
    end
    out
end


function getall(t::DiscretizeRelax{X,T}, v::Relaxation{Upper}) where {X, T <: AbstractInterval}
    dim1, dim2 = size(t.storage)
    out = zeros(Float64, dim1, dim2)
    @inbounds for i in eachindex(out)
        out[i] = t.storage[i].Intv.hi
    end
    out
end
function getall(t::DiscretizeRelax{X,T}, v::Relaxation{Upper}) where {X, T <: MC}
    dim1, dim2 = size(t.storage)
    out = zeros(Float64, dim1, dim2)
    @inbounds for i in eachindex(out)
        out[i] = t.storage[i].cc
    end
    out
end
