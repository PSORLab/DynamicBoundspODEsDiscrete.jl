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

function getall!(out::Vector{Array{Float64,2}}, t::DiscretizeRelax{X,T}, g::Subgradient{Lower}) where {X, T < AbstractInterval}
    for i in 1:t.np
        fill!(out[i], 0.0)
    end
    return
end
function getall!(out::Vector{Array{Float64,2}}, t::DiscretizeRelax{X,T}, g::Subgradient{Lower}) where {X, T < MC}
    for i in 1:t.np
        @inbounds for j in eachindex(out[i])
            out[i][j] = t.storage[j].cv_grad[j]
        end
    end
    return
end

function getall!(out::Vector{Array{Float64,2}}, t::DiscretizeRelax{X,T}, g::Subgradient{Upper}) where {X, T < AbstractInterval}
    for i in 1:t.np
        fill!(out[i], 0.0)
    end
    return
end

function getall!(out::Vector{Array{Float64,2}}, t::DiscretizeRelax{X,T} g::Subgradient{Upper}) where {X, T < MC}
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

function getall!(out::Union{Vector{Float64}, Matrix{Float64}},, t::DiscretizeRelax, v::Bound{Upper})
    @inbounds for j in eachindex(out)
        out[j] = t.storage[j].Intv.hi
    end
    return
end

function getall!(out::Array{Float64,2}, t::DiscretizeRelax, v::Relaxation{Lower})
    if t.evaluate_interval
        @inbounds for i in eachindex(out)
            out[i] = t.relax_lo[i]
        end
    else
        @inbounds for i in eachindex(out)
            out[i] = t.state_relax[i].cv
        end
    end
    return
end
function getall!(out::Vector{Float64}, t::DiscretizeRelax, v::Relaxation{Lower})
    if t.evaluate_interval
        @inbounds for i in eachindex(out)
            out[i] = t.X[i].lo
        end
    else
        @inbounds for i in eachindex(out)
            out[i] = t.state_relax[i].cv
        end
    end
    return
end

function getall!(out::Array{Float64,2}, t::DiscretizeRelax, v::Relaxation{Upper})
    if t.evaluate_interval
        @inbounds for i in eachindex(out)
            out[i] = t.X[i].hi
        end
    else
        @inbounds for i in eachindex(out)
            out[i] = t.state_relax[i].cc
        end
    end
    return
end
function getall!(out::Vector{Float64}, t::DiscretizeRelax, v::Relaxation{Upper})
    if t.evaluate_interval
        @inbounds for i in eachindex(out)
            out[i] = t.X[i].hi
        end
    else
        @inbounds for i in eachindex(out)
            out[i] = t.state_relax[i].cc
        end
    end
    return
end
