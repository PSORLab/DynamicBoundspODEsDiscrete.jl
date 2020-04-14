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

function getall!(out::Vector{Array{Float64,2}}, t::DiscretizeRelax, g::Gradient{Nominal})
    for i in 1:t.np
        @inbounds for j in eachindex(out[i])
            out[i][j] = t.local_problem_storage.pode_dxdp[i][j]
        end
    end
    return
end

function getall!(out::Vector{Array{Float64,2}}, t::DiscretizeRelax, g::Gradient{Lower})
    if ~t.differentiable
        error("Integrator does not generate differential relaxations. Set the
               differentiable_flag field to true and reintegrate.")
    end
    for i in 1:t.np
        if t.evaluate_interval
            fill!(out[i], 0.0)
        else
            @inbounds for j in eachindex(out[i])
                out[i][j] = t.relax_cv_grad[j][i]
            end
        end
    end
    return
end
function getall!(out::Vector{Array{Float64,2}}, t::DiscretizeRelax, g::Gradient{Upper})
    if ~t.differentiable_flag
        error("Integrator does not generate differential relaxations. Set the
               differentiable_flag field to true and reintegrate.")
    end
    for i in 1:t.np
        if t.evaluate_interval
            fill!(out[i], 0.0)
        else
            @inbounds for j in eachindex(out[i])
                out[i][j] = t.relax_cc_grad[j][i]
            end
        end
    end
    return
end

function getall!(out::Vector{Array{Float64,2}}, t::DiscretizeRelax, g::Subgradient{Lower})
    for i in 1:t.np
        if t.evaluate_interval
            fill!(out[i], 0.0)
        else
            @inbounds for j in eachindex(out[i])
                out[i][j] = t.relax_cv_grad[j][i]
            end
        end
    end
    return
end
function getall!(out::Vector{Array{Float64,2}}, t::DiscretizeRelax, g::Subgradient{Upper})
    for i in 1:t.np
        if t.evaluate_interval
            fill!(out[i], 0.0)
        else
            @inbounds for j in eachindex(out[i])
                out[i][j] = t.relax_cc_grad[j][i]
            end
        end
    end
    return
end

function getall!(out::Array{Float64,2}, t::DiscretizeRelax, v::Bound{Lower})
    out .= t.relax_lo
    return
end

function getall!(out::Vector{Float64}, t::DiscretizeRelax, v::Bound{Lower})
    out[:] = t.relax_lo[1,:]
    return
end

function getall!(out::Array{Float64,2}, t::DiscretizeRelax, v::Bound{Upper})
    out .= t.relax_hi
    return
end

function getall!(out::Vector{Float64}, t::DiscretizeRelax, v::Bound{Upper})
    out[:] = t.relax_hi[1,:]
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
